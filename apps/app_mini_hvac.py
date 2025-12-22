# app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import numpy as np
from hvac_app import preprocessing_app, cop_calculation_app, regression_app
from hvac_app.dynamic_plot_app import MultiFluidDynamicPlot
from hvac_app.enhance_mini_hvac_app import align_to_match_baseline


def run():

    # ---------------- Session State ---------------- #
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'true_original_data' not in st.session_state:
        st.session_state.true_original_data = {}
    if 'aligned_data' not in st.session_state:
        st.session_state.aligned_data = {}
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "original"

    # ---------------- Config ---------------- #
    st.set_page_config(
        page_title="Mini-HVAC",
        layout="wide",
        page_icon="htms_logo.jpg"
    )

    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("Mini-HVAC Performance Analysis")
    with col2:
        st.image("htms_logo.jpg", width=150)

    COLUMNS = ['Date', 'Time', 'Energy1', 'Energy2', 'T1', 'T2', 'T3', 'T4']
    CHOSEN_BIN_SIZE = 1

    # ---------------- Helpers ---------------- #
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

        numeric_cols = ['Energy1', 'Energy2', 'T1', 'T2', 'T3', 'T4']
        for col in numeric_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        df_raw = df_raw.sort_values('Date').reset_index(drop=True)
        df_raw = preprocessing_app.iqr_filter(df_raw, 'T3')
        df = preprocessing_app.add_datetime(df_raw.copy())
        df = preprocessing_app.resample_to_minute(df)
        df['time_bin_mark'] = df['1_min_helper'].dt.floor('15min')

        grouped = df.groupby('time_bin_mark')[['Energy1', 'Energy2', 'T3']].agg({
            'Energy1': ['min', 'max'],
            'Energy2': ['min', 'max'],
            'T3': 'mean'
        })
        grouped.columns = ['min_energy1', 'max_energy1', 'min_energy2', 'max_energy2', 'avg_oat']
        grouped = grouped.reset_index()

        grouped = cop_calculation_app.compute_delta_energy(grouped)
        for col in ['delta_energy1', 'delta_energy2']:
            grouped = preprocessing_app.iqr_filter(grouped, col)

        grouped['cop_15min'] = grouped['delta_energy2'] / grouped['delta_energy1']
        grouped['cop_15min_smooth'] = grouped['cop_15min'].rolling(window=3, center=True, min_periods=1).mean()
        grouped = preprocessing_app.remove_outliers_zscore(grouped, 'cop_15min_smooth')

        st_container.success(f"{fluid_name} processing complete! {len(grouped)} valid 15-min points")
        return grouped, df_raw, df

    def analyze_bins(grouped, df_raw, st_container=None):
        if st_container is None:
            st_container = st

        grouped = grouped.copy()
        grouped['avg_oat'] = pd.to_numeric(grouped['avg_oat'], errors='coerce')
        grouped['cop_15min_smooth'] = pd.to_numeric(grouped['cop_15min_smooth'], errors='coerce')
        grouped = grouped.dropna(subset=['avg_oat', 'cop_15min_smooth'])

        temp = cop_calculation_app.oat_binning(grouped.copy(), bin_size=CHOSEN_BIN_SIZE)
        cop_grouped = temp.groupby('oat_interval')['cop_15min_smooth'].mean().reset_index()

        X = []
        for interval in cop_grouped['oat_interval']:
            if isinstance(interval, str) and '-' in interval:
                low, _ = map(float, interval.split('-'))
                X.append(int(low))
            else:
                X.append(float(interval))

        X = np.array(X)
        y = cop_grouped['cop_15min_smooth'].values

        mask = ~(np.isnan(X) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 1:
            return {CHOSEN_BIN_SIZE: {"X": np.array([]), "y": np.array([]), "y_pred": np.array([]), "r2": None, "slope": None}}

        X_2d = X.reshape(-1, 1)
        y_pred, y_smooth, r2, slope = regression_app.smoothed_linear_regression(
            X_2d, y, window_length=5, polyorder=2
        )

        return {
            CHOSEN_BIN_SIZE: {
                "X": X,
                "y": y_smooth,
                "y_pred": y_pred,
                "r2": r2,
                "slope": slope
            }
        }

    # ---------------- Phase Selection ---------------- #
    phase = st.radio("Select Phase:", ["Phase 1: Analysis", "Phase 2: Alignment"], horizontal=True)

    if phase == "Phase 1: Analysis":
        st.subheader("Upload Data for Analysis")

        baseline_name = st.text_input("Enter Baseline Name", value=st.session_state.get('baseline_name_input', ''))
        st.session_state['baseline_name_input'] = baseline_name
        baseline_files = st.file_uploader("Upload Baseline Data", accept_multiple_files=True)

        fluid_name = st.text_input("Enter Product Name", value=st.session_state.get('fluid_name_input', ''))
        st.session_state['fluid_name_input'] = fluid_name
        fluid_files = st.file_uploader("Upload Product Data", accept_multiple_files=True)

        if st.button("Analyse"):
            if not baseline_name or not fluid_name or not baseline_files or not fluid_files:
                st.error("Provide names and files for both fluids.")
                return

            baseline_paths = save_uploaded_files(baseline_files)
            fluid_paths = save_uploaded_files(fluid_files)

            col_b, col_f = st.columns(2)
            with col_b:
                grouped_base, df_raw_base, _ = process_fluid(baseline_name, baseline_paths, st.container())
            with col_f:
                grouped_fluid, df_raw_fluid, _ = process_fluid(fluid_name, fluid_paths, st.container())

            bin_base = analyze_bins(grouped_base, df_raw_base)
            bin_fluid = analyze_bins(grouped_fluid, df_raw_fluid)

            st.session_state.results_dict = {
                baseline_name: {"grouped": grouped_base, "df_raw": df_raw_base, "bin_results": bin_base[CHOSEN_BIN_SIZE]},
                fluid_name: {"grouped": grouped_fluid, "df_raw": df_raw_fluid, "bin_results": bin_fluid[CHOSEN_BIN_SIZE]}
            }

            st.session_state.baseline_name = baseline_name
            MultiFluidDynamicPlot(st.session_state.results_dict, default_baseline=baseline_name)

    if phase == "Phase 2: Alignment":
        align_phase()
