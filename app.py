# app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import numpy as np
from hvac_app import preprocessing_app, cop_calculation_app, regression_app
from hvac_app.dynamic_plot_app import MultiFluidDynamicPlot
from hvac_app.enhance_app import align_to_match_baseline

# ---------------- Session State ---------------- #
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'true_original_data' not in st.session_state:
    st.session_state.true_original_data = {}
if 'aligned_data' not in st.session_state:
    st.session_state.aligned_data = {}
if 'current_view' not in st.session_state:
    st.session_state.current_view = "original"

# Set page title and favicon (browser tab / URL bar)
st.set_page_config(
    page_title="HVAC Performance Analysis",
    layout="wide",
    page_icon="htms_logo.jpg"
)

col1, col2 = st.columns([5,1])
with col1:
    st.title("Mini-HVAC Performance Analysis")  # page title text
with col2:
    st.image("htms_logo.jpg", width=150)

COLUMNS = ['Date', 'Time', 'Energy1', 'Energy2', 'T1', 'T2', 'T3', 'T4']
CHOSEN_BIN_SIZE = 1


def save_uploaded_files(uploaded_files):
    paths = []
    for f in uploaded_files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
        tmp.write(f.read())
        tmp.close()
        paths.append(Path(tmp.name))
    return paths


def load_fluid_files(txt_files, columns_to_keep):
    all_dfs = []
    for file_path in txt_files:
        try:
            df = pd.read_csv(file_path, sep='\t', decimal=',', skiprows=1, header=None)
            df.columns = columns_to_keep
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {file_path}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def process_fluid(fluid_name, file_paths, st_container=None):
    if st_container is None:
        st_container = st
    st_container.info(f"Loading files for {fluid_name}...")
    df_raw = load_fluid_files(file_paths, COLUMNS)
    if df_raw.empty:
        st_container.warning(f"No data loaded for {fluid_name}")
        return None, None, None
    numeric_cols = ['Energy1','Energy2','T1','T2','T3','T4']
    for col in numeric_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw = df_raw.sort_values('Date').reset_index(drop=True)
    df_raw = preprocessing_app.iqr_filter(df_raw, 'T3')
    df = preprocessing_app.add_datetime(df_raw.copy())
    df = preprocessing_app.resample_to_minute(df)
    df['time_bin_mark'] = df['1_min_helper'].dt.floor('15min')
    grouped = df.groupby('time_bin_mark')[['Energy1','Energy2','T3']].agg({
        'Energy1':['min','max'],
        'Energy2':['min','max'],
        'T3':'mean'
    })
    grouped.columns = ['min_energy1','max_energy1','min_energy2','max_energy2','avg_oat']
    grouped = grouped.reset_index()
    grouped = cop_calculation_app.compute_delta_energy(grouped)
    for col in ['delta_energy1','delta_energy2']:
        grouped = preprocessing_app.iqr_filter(grouped, col)
    grouped['cop_15min'] = grouped['delta_energy2'] / grouped['delta_energy1']
    grouped['cop_15min_smooth'] = grouped['cop_15min'].rolling(window=3, center=True, min_periods=1).mean()
    grouped = preprocessing_app.remove_outliers_zscore(grouped, 'cop_15min_smooth')
    st_container.success(f"{fluid_name} processing complete! {len(grouped)} valid 15-min points")
    return grouped, df_raw, df


def analyze_bins(grouped, df_raw, st_container=None):
    bin_results = {}
    if st_container is None:
        st_container = st

    grouped = grouped.copy()
    grouped['avg_oat'] = pd.to_numeric(grouped['avg_oat'], errors='coerce')
    grouped['cop_15min_smooth'] = pd.to_numeric(grouped['cop_15min_smooth'], errors='coerce')
    grouped = grouped.dropna(subset=['avg_oat', 'cop_15min_smooth'])
    
    temp = cop_calculation_app.oat_binning(grouped.copy(), bin_size=CHOSEN_BIN_SIZE)
    cop_grouped_smooth = temp.groupby('oat_interval')['cop_15min_smooth'].mean().reset_index()
    
    df_raw_binned = cop_calculation_app.oat_binning(df_raw.copy(), bin_size=CHOSEN_BIN_SIZE, temp_col='T3')
    raw_counts = df_raw_binned.groupby('oat_interval').size().reset_index(name='n_samples_raw')
    cop_grouped_smooth = cop_grouped_smooth.merge(raw_counts, on='oat_interval', how='left')

    if len(cop_grouped_smooth) >= 1:
        X = []
        for interval in cop_grouped_smooth['oat_interval']:
            if isinstance(interval, str) and '-' in interval:
                low, high = map(float, interval.split('-'))
                X.append(int(low))
            else:
                X.append(float(interval))

        X = np.array(X)
        y = cop_grouped_smooth['cop_15min_smooth'].values
        
        X = pd.to_numeric(X, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        mask = ~(np.isnan(X) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 1:
            st_container.warning("Not enough valid data points for regression after cleaning")
            bin_results[CHOSEN_BIN_SIZE] = {
                "X": np.array([]), "y": np.array([]), "y_pred": np.array([]), 
                "r2": None, "slope": None
            }
            return bin_results
        
        X_2d = X.reshape(-1, 1)
        y_pred, y_smooth, r2, slope = regression_app.smoothed_linear_regression(X_2d, y, window_length=5, polyorder=2)

        bin_results[CHOSEN_BIN_SIZE] = {
            "X": X,
            "y": y_smooth, 
            "y_pred": y_pred, 
            "r2": r2, 
            "slope": slope
        }

    else:
        bin_results[CHOSEN_BIN_SIZE] = {
            "X": np.array([]), "y": np.array([]), "y_pred": np.array([]), 
            "r2": None, "slope": None, "n_samples": np.array([])
        }
    return bin_results


# ---------------- Phase 2 ---------------- #
def align_phase():
    st.subheader("Data Enhancement using Alignment")
    
    if 'results_dict' not in st.session_state or 'baseline_name' not in st.session_state:
        st.warning("Please run Phase 1 analysis first to load fluid data.")
        return

    base_fluid_names = list(st.session_state.true_original_data.keys())
    
    # 1. Select the baseline fluid to match its slope
    baseline_fluid = st.selectbox(
        "1. Select the **Baseline Fluid** to match its regression slope:",
        options=[st.session_state.baseline_name] if st.session_state.baseline_name in base_fluid_names else base_fluid_names,
        index=0 if st.session_state.baseline_name in base_fluid_names else 0
    )
    
    # 2. Select the product fluid to be squished
    product_options = [name for name in base_fluid_names if name != baseline_fluid]
    if not product_options:
        st.error("Please upload at least two fluids in Phase 1 to perform comparison.")
        return
        
    product_fluid = st.selectbox(
        "2. Select the **Product Fluid** to apply Alignment to:",
        options=product_options
    )
    
    aligned_name_key = next((name for name in st.session_state.results_dict if name.startswith(product_fluid) and 'aligned' in name), None)

    # --- Action Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        aligned_button = st.button("Run Alignment", type="primary")

    # --- Running Aligning ---
    if aligned_button:
        
        # Get baseline slope
        baseline_data = st.session_state.results_dict.get(baseline_fluid)
        product_data = st.session_state.true_original_data.get(product_fluid)

        if not baseline_data or not product_data:
            st.error("Required data not found for baseline or product fluid.")
            return

        baseline_slope = baseline_data['bin_results'].get('slope')
        if baseline_slope is None:
            st.error(f"Cannot perform alignment: Baseline fluid '{baseline_fluid}' has no valid regression slope. Check Phase 1 results.")
            return

        with st.spinner(f"Running alignment on {product_fluid} to match {baseline_fluid} slope ({baseline_slope:.6f})..."):
            
            # 1. Run the alignment
            aligned_name, aligned_data = align_to_match_baseline(
                product_fluid,
                product_data,
                baseline_slope,
                baseline_fluid
            )

            if aligned_name is None or aligned_data is None:
                st.error(
                    f"Alignment aborted: '{product_fluid}' does not have enough valid data for Alignment.")
                return

            if aligned_data:
                # 2. Store the new aligned result
                st.session_state.results_dict[aligned_name] = aligned_data

                st.success(f"Alignment Complete. Plot now shows **{baseline_fluid}** and **{aligned_name}**.")
            else:
                st.error("Alignment failed. See logs above.")

    # Display Baseline Slope
    if st.session_state.get('results_dict') and baseline_fluid in st.session_state.true_original_data:
        # Get the slope from the original (or currently displayed) baseline data
        current_baseline_data = st.session_state.results_dict.get(baseline_fluid, st.session_state.true_original_data[baseline_fluid])
        baseline_slope = current_baseline_data['bin_results'].get('slope')

    aligned_name_key = next((name for name in st.session_state.results_dict if name.startswith(product_fluid) and 'aligned' in name), None)

    if 'results_dict' in st.session_state and 'baseline_name' in st.session_state:
        plotter = MultiFluidDynamicPlot(
            st.session_state.results_dict,
            default_baseline=st.session_state.baseline_name
        )
        
# ---------------- Main Application Flow ---------------- #
# Phase selection
phase = st.radio("Select Phase:", ["Phase 1: Analysis", "Phase 2: Alignment"], horizontal=True)

if phase == "Phase 1: Analysis":
    st.subheader("Upload Data for Analysis")
    
    # Baseline Name & Files
    baseline_name = st.text_input("Enter Baseline Name", value=st.session_state.get('baseline_name_input', ''))
    st.session_state['baseline_name_input'] = baseline_name

    baseline_files = st.file_uploader("Upload Baseline Data", accept_multiple_files=True, key="baseline_files_uploader")
    if baseline_files:
        st.session_state['baseline_paths'] = save_uploaded_files(baseline_files)

    # Product Fluid Name & Files
    fluid_name = st.text_input("Enter Product Name", value=st.session_state.get('fluid_name_input', ''))
    st.session_state['fluid_name_input'] = fluid_name

    fluid_files = st.file_uploader("Upload Product Data", accept_multiple_files=True, key="fluid_files_uploader")
    if fluid_files:
        st.session_state['fluid_paths'] = save_uploaded_files(fluid_files)

    run_button = st.button("Analyse")

    if run_button:
        st.session_state.removed_points = {}
        st.session_state.removed_history = {}

        # Get paths from session
        baseline_paths = st.session_state.get('baseline_paths', [])
        fluid_paths = st.session_state.get('fluid_paths', [])

        if baseline_files:
            baseline_paths = st.session_state['baseline_paths']
        if fluid_files:
            fluid_paths = st.session_state['fluid_paths']

        if not baseline_name or not baseline_paths or not fluid_name or not fluid_paths:
            st.error("Provide names and files for both fluids.")
            st.session_state.analysis_run = False
        else:
            st.session_state.analysis_run = True

            # --- Processing ---
            col_b, col_f = st.columns(2)
            with col_b:
                grouped_base, df_raw_base, df_base = process_fluid(baseline_name, baseline_paths, st.container())
            with col_f:
                grouped_fluid, df_raw_fluid, df_fluid = process_fluid(fluid_name, fluid_paths, st.container())

            # --- Analysis ---
            if grouped_base is None or grouped_fluid is None:
                st.error("Analysis stopped due to data loading/processing error.")
                st.session_state.analysis_run = False
            else:
                bin_base = analyze_bins(grouped_base, df_raw_base)
                bin_fluid = analyze_bins(grouped_fluid, df_raw_fluid)

                # Store results
                st.session_state.results_dict = {
                    baseline_name: {"grouped": grouped_base, "df_raw": df_raw_base, "bin_results": bin_base[CHOSEN_BIN_SIZE]},
                    fluid_name: {"grouped": grouped_fluid, "df_raw": df_raw_fluid, "bin_results": bin_fluid[CHOSEN_BIN_SIZE]}
                }

                st.session_state.true_original_data = {
                    baseline_name: {"grouped": grouped_base.copy(), "df_raw": df_raw_base.copy(), "bin_results": bin_base[CHOSEN_BIN_SIZE].copy()},
                    fluid_name: {"grouped": grouped_fluid.copy(), "df_raw": df_raw_fluid.copy(), "bin_results": bin_fluid[CHOSEN_BIN_SIZE].copy()}
                }

                st.session_state.baseline_name = baseline_name
                st.session_state.analysis_run = True
                plotter = MultiFluidDynamicPlot({baseline_name: st.session_state.results_dict[baseline_name],fluid_name: st.session_state.results_dict[fluid_name]},default_baseline=baseline_name)

    elif st.session_state.get('analysis_run', False):
        st.info("Analysis Complete")
        if 'results_dict' in st.session_state and 'baseline_name' in st.session_state:
            plotter = MultiFluidDynamicPlot(st.session_state.results_dict, default_baseline=st.session_state.baseline_name)


if phase == "Phase 2: Alignment":
    align_phase()