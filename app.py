# app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import numpy as np
from hvac_app import preprocessing_app, cop_calculation_app, regression_app
from hvac_app.dynamic_plot_app import MultiFluidDynamicPlot
from hvac_app.lstm_app import enhance_fluid_data

# ---------------- Session State ---------------- #
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'true_original_data' not in st.session_state:
    st.session_state.true_original_data = {}

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
        
        # Ensure numeric types
        X = pd.to_numeric(X, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Remove any NaN values
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


def enhancement_phase():
    st.subheader("Data Enhancement using LSTM")
    
    if 'results_dict' not in st.session_state:
        st.warning("Please run Phase 1 analysis first to enhance fluids.")
        return
    
    #st.session_state.removed_points = {}
    #st.session_state.removed_history = {}
    
    fluid_names = list(st.session_state.results_dict.keys())
    base_fluid_names = [name for name in fluid_names if "Enhanced" not in name]
    
    # Fluid selection
    selected_fluid = st.selectbox(
        "Select fluid to enhance:",
        options=base_fluid_names
    )
    
    # Check if this fluid is already enhanced
    is_currently_enhanced = st.session_state.results_dict.get(selected_fluid, {}).get('is_enhanced', False)
    
    # Enhancement parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        percent_points = st.slider(
            "Percent of points to generate:",
            min_value=1,
            max_value=1000,
            value=100,
            step=1
        )
    
    fluid_data = st.session_state.results_dict.get(selected_fluid, {})
    original_points = len(fluid_data.get('grouped', []))
    requested_points = int(original_points * percent_points / 100)
    
    # Show helpful warnings
    validation_ok = True
    if original_points > 0 and requested_points == 0:
        min_percent = max(1, (100 // original_points) + 1)
        st.error(f"❌ Cannot enhance: {percent_points}% of {original_points} points = 0 points. Try at least {min_percent}%.")
        validation_ok = False
    
    if original_points < 2:
        st.error(f"❌ Cannot enhance: Need at least 2 data points, but {selected_fluid} only has {original_points}.")
        validation_ok = False
    
    with col2:
        # Only enable if validation passes
        if validation_ok and not is_currently_enhanced:
            enhance_button = st.button("🎯 Generate Enhanced Data", type="primary")
        elif validation_ok and is_currently_enhanced:
            enhance_button = st.button("🔄 Re-generate Enhanced Data", type="primary")
        else:
            enhance_button = False
            if is_currently_enhanced:
                st.button("🔄 Re-generate Enhanced Data", disabled=True)
            else:
                st.button("🎯 Generate Enhanced Data", disabled=True)
    
    with col3:
        if is_currently_enhanced:
            undo_button = st.button("⏪ Remove Enhancement", type="secondary")
        else:
        # Show why no undo is available
            if selected_fluid in st.session_state.results_dict:
                st.write("📝 No enhancement to remove")
            else:
                st.write("⚠️ No fluid selected")

    if 'undo_button' in locals() and undo_button and selected_fluid:
        if selected_fluid in st.session_state.true_original_data:
            # Restore from true original data
            true_original = st.session_state.true_original_data[selected_fluid]
            st.session_state.results_dict[selected_fluid] = {
            "grouped": true_original['grouped'].copy(),
            "df_raw": true_original['df_raw'].copy(),
            "bin_results": analyze_bins(true_original['grouped'].copy(), true_original['df_raw'])[CHOSEN_BIN_SIZE],
            "is_enhanced": False 
                }
            # Also remove any "Enhanced" version if it exists
            enhanced_name = f"{selected_fluid} Enhanced"
            if enhanced_name in st.session_state.results_dict:
                del st.session_state.results_dict[enhanced_name]
        
            st.success(f"✅ Enhancement removed from {selected_fluid}")
            st.rerun()
        else:
            st.error("Cannot undo - true original data not found")

    # Handle enhancement
    if enhance_button and selected_fluid and original_points >= 2 and requested_points > 0:
        with st.spinner(f"Generating enhanced data for {selected_fluid}..."):
            try:
                fluid_data = st.session_state.results_dict[selected_fluid]
                
                # ALWAYS use the TRUE ORIGINAL data (not enhanced versions)
                if 'true_original_data' in st.session_state:
                    true_original_data = st.session_state.true_original_data.get(selected_fluid)
                    if true_original_data:
                        grouped_original = true_original_data.get('grouped')
                        df_raw = true_original_data.get('df_raw')
                    else:
                        # Fallback to current data and store as true original
                        grouped_original = fluid_data.get('grouped')
                        df_raw = fluid_data.get('df_raw')
                        st.session_state.true_original_data[selected_fluid] = {
                            "grouped": grouped_original.copy(),
                            "df_raw": df_raw.copy()
                        }
                else:
                    # First time - store true original separately
                    grouped_original = fluid_data.get('grouped')
                    df_raw = fluid_data.get('df_raw')
                    st.session_state.true_original_data = {
                        selected_fluid: {
                            "grouped": grouped_original.copy(),
                            "df_raw": df_raw.copy()
                        }
                    }
                
                if grouped_original is None or grouped_original.empty:
                    st.error(f"No data available for {selected_fluid}")
                    return
                
                # Ensure clean numeric data
                clean_grouped = grouped_original.copy()
                
                # Force convert to numeric and handle any conversion issues
                clean_grouped['avg_oat'] = pd.to_numeric(clean_grouped['avg_oat'], errors='coerce')
                clean_grouped['cop_15min_smooth'] = pd.to_numeric(clean_grouped['cop_15min_smooth'], errors='coerce')
                
                # Remove any rows with non-numeric values
                clean_grouped = clean_grouped.dropna(subset=['avg_oat', 'cop_15min_smooth'])
                
                # Ensure the LSTM has the column it expects with proper numeric type
                clean_grouped['cop_15min'] = pd.to_numeric(clean_grouped['cop_15min_smooth'], errors='coerce')
                
                # Final validation
                if clean_grouped['cop_15min'].dtype != 'float64':
                    st.error(f"COP data is still not numeric! Current dtype: {clean_grouped['cop_15min'].dtype}")
                    return
                if len(clean_grouped) < 2:
                    st.error("Not enough valid data points after cleaning")
                    return
                
                # Generate enhanced data using CLEAN data
                enhanced_grouped_data, synthetic_df = enhance_fluid_data(
                    clean_grouped,
                    percent_points=percent_points
                )
                
                if enhanced_grouped_data is None or enhanced_grouped_data.empty:
                    st.error("Enhancement failed - no data generated")
                    return
                
                # Keep only required columns
                required_cols = ['time_bin_mark', 'cop_15min_smooth', 'avg_oat']
                enhanced_grouped_data = enhanced_grouped_data[required_cols]

                # Ensure numeric data types in enhanced data
                enhanced_grouped_data['avg_oat'] = pd.to_numeric(enhanced_grouped_data['avg_oat'], errors='coerce')
                enhanced_grouped_data['cop_15min_smooth'] = pd.to_numeric(enhanced_grouped_data['cop_15min_smooth'], errors='coerce')
                enhanced_grouped_data = enhanced_grouped_data.dropna(subset=['avg_oat', 'cop_15min_smooth'])

                # Analyze enhanced bins
                enhanced_results = analyze_bins(
                    enhanced_grouped_data.copy(),
                    df_raw,
                    f"{selected_fluid} Enhanced"
                )
                
                # Replace the original fluid with enhanced version
                st.session_state.results_dict[selected_fluid] = {
                    "bin_results": enhanced_results[CHOSEN_BIN_SIZE],
                    "grouped": enhanced_grouped_data,
                    "df_raw": df_raw,
                    "is_enhanced": True,  # Flag to track enhancement
                    "original_data": fluid_data  # Store original for undo
                }

                st.success(f"✅ {selected_fluid} successfully enhanced! {len(synthetic_df)} synthetic points added.")
                st.rerun()

                # Show enhancement stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Points", len(grouped_original))
                with col2:
                    st.metric("Synthetic Points", len(synthetic_df))
                with col3:
                    st.metric("Total Points", len(enhanced_grouped_data))
                
            except Exception as e:
                st.error(f"Enhancement failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Enhancement status
    st.subheader("Enhancement Status")
    enhanced_fluids = [name for name in base_fluid_names 
                      if st.session_state.results_dict.get(name, {}).get('is_enhanced')]
    
    if enhanced_fluids:
        st.write("**Enhanced Fluids:**")
        for fluid in enhanced_fluids:
            enhanced_data = st.session_state.results_dict[fluid]
            original_points = len(enhanced_data.get('original_data', {}).get('grouped', []))
            enhanced_points = len(enhanced_data.get('grouped', []))
            st.write(f"• **{fluid}**: {original_points} original points → {enhanced_points} total points")
    else:
        st.info("No fluids are currently enhanced")
    
    # Show the plot with current data (enhanced or original)
    if 'results_dict' in st.session_state:
        plotter = MultiFluidDynamicPlot(
            st.session_state.results_dict,  # Use the full dict
            default_baseline=st.session_state.get('baseline_name')
        )


# Phase selection
phase = st.radio("Select Phase:", ["Phase 1: Analysis", "Phase 2: Enhancement"], horizontal=True)


if phase == "Phase 1: Analysis":
    st.subheader("Upload Data for Analysis")
    
    baseline_name = st.text_input("Enter Baseline Name")
    baseline_files = st.file_uploader("Upload Baseline Data", accept_multiple_files=True)
    fluid_name = st.text_input("Enter Product Name") 
    fluid_files = st.file_uploader("Upload Product Data", accept_multiple_files=True)
    run_button = st.button("Analyse")

    if run_button:
        st.session_state.removed_points = {}
        st.session_state.removed_history = {}

    if run_button:
        if not baseline_name or not baseline_files or not fluid_name or not fluid_files:
            st.error("Provide names and files for both fluids.")
            st.session_state.analysis_run = False
        else:
            st.session_state.analysis_run = True
            baseline_paths = save_uploaded_files(baseline_files)
            fluid_paths = save_uploaded_files(fluid_files)
            grouped_base, df_raw_base, df_base = process_fluid(baseline_name, baseline_paths)
            grouped_fluid, df_raw_fluid, df_fluid = process_fluid(fluid_name, fluid_paths)
            bin_base = analyze_bins(grouped_base, df_raw_base)
            bin_fluid = analyze_bins(grouped_fluid, df_raw_fluid)
        
        # Store results
            st.session_state.results_dict = {
            baseline_name: {"grouped": grouped_base, "df_raw": df_raw_base, "bin_results": bin_base[CHOSEN_BIN_SIZE]},
            fluid_name: {"grouped": grouped_fluid, "df_raw": df_raw_fluid, "bin_results": bin_fluid[CHOSEN_BIN_SIZE]}
            }
        
        # Store TRUE original data for enhancement
            st.session_state.true_original_data = {
            baseline_name: {
                "grouped": grouped_base.copy(),
                "df_raw": df_raw_base.copy()
            },
            fluid_name: {
                "grouped": grouped_fluid.copy(), 
                "df_raw": df_raw_fluid.copy()
            }
        }
        
            st.session_state.baseline_name = baseline_name
            st.session_state.analysis_run = True
            plotter = MultiFluidDynamicPlot(st.session_state.results_dict, default_baseline=baseline_name)

    elif st.session_state.get('analysis_run', False):
        st.info("Analysis completed. Use the interactive plot to explore the data.")
        if 'results_dict' in st.session_state and 'baseline_name' in st.session_state:
            plotter = MultiFluidDynamicPlot(st.session_state.results_dict, default_baseline=st.session_state.baseline_name)


elif phase == "Phase 2: Enhancement":
    enhancement_phase()