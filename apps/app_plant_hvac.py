# app_plant_hvac.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from hvac_app.dynamic_plot_plant_hvac_app import PlantMultiFluidDynamicPlot
from hvac_app.enhance_plant_hvac_app import align_to_match_baseline_plant
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import re

TONS_CONSTANT = 3.51685
WINDOW_LENGTH = 5
POLYORDER = 2

def save_uploaded_files(uploaded_files):
    paths = []
    for f in uploaded_files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
        tmp.write(f.read())
        tmp.close()
        paths.append(Path(tmp.name))
    return paths

def iqr_filter(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series >= Q1 - 1.5*IQR) & (series <= Q3 + 1.5*IQR)

def _normalize(col):
    return re.sub(r'[^a-z0-9]', '', col.lower())

def detect_fluid_in_out(df):
    norm_cols = {col: _normalize(col) for col in df.columns}
    IN_KEYS  = ['fluidin', 'entering', 'inlet', 'tin', 'tempin']
    OUT_KEYS = ['fluidout', 'leaving', 'outlet', 'tout', 'tempout']
    fluid_in, fluid_out = None, None
    for col, ncol in norm_cols.items():
        if any(k in ncol for k in IN_KEYS): fluid_in = col
        if any(k in ncol for k in OUT_KEYS): fluid_out = col
    if fluid_in and fluid_out: return fluid_in, fluid_out
    raise ValueError("Fluid in/out columns not detected")

def detect_timestamp(df):
    for col in df.columns:
        if "DATE" in col.upper() or "TIME" in col.upper():
            return col
    return df.columns[0]

def detect_state_on(df):
    for col in df.select_dtypes(include=np.number).columns:
        if set(df[col].dropna().unique()).issubset({0,1}):
            return col
    raise ValueError("No binary ON/OFF column found")

def detect_oat(df):
    for col in df.columns:
        if 'OAT' in col.upper(): return col
    raise ValueError("OAT column not found")

def detect_flow_and_power(df, state_col, exclude_cols):
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude_cols]
    corrs = {c: df[c].corr(df[state_col]) for c in numeric_cols}
    flow_col = max(corrs, key=lambda x: abs(corrs[x]))
    power_cols = [c for c in numeric_cols if c != flow_col]
    if len(power_cols) != 1: raise ValueError(f"Expected 1 numeric column for power, got {power_cols}")
    return flow_col, power_cols[0]

def oat_density_table(df, oat_col="oat", bin_size=1):
    df = df.copy()
    df["oat_bin"] = np.floor(df[oat_col] / bin_size) * bin_size
    counts = df.groupby("oat_bin").size().sort_index()
    return counts

def build_horizontal_oat_table(density_dict):
    all_bins = sorted(set().union(*[s.index for s in density_dict.values()]))
    table_data = []
    for name, series in density_dict.items():
        row = {"Fluid": name}
        for b in all_bins:
            row[f"{int(b)}°C"] = f"{int(series.get(b, 0)):,}"
        table_data.append(row)
    return pd.DataFrame(table_data)

def filter_by_trend(df, oat_col="oat", cop_col="cop", keep_fraction=0.9):
    X = df[oat_col].values.reshape(-1, 1)
    y = df[cop_col].values
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    residuals = np.abs(y - y_pred)
    threshold = np.percentile(residuals, keep_fraction * 100)
    mask = residuals <= threshold
    df_filtered = df[mask].copy()
    X_f = df_filtered[oat_col].values.reshape(-1, 1)
    y_f = df_filtered[cop_col].values
    lr_filtered = LinearRegression().fit(X_f, y_f)
    df_filtered["cop_trend"] = lr_filtered.predict(X_f)
    return df_filtered, lr_filtered

def process_fluid(file_paths, fluid_name, baseline_name, st_container=None):
    if st_container is None: st_container = st
    st_container.info(f"Loading files for {fluid_name}...")
    all_dfs = []

    for f in file_paths:
        df = pd.read_csv(f).dropna()
        timestamp_col = detect_timestamp(df)
        state_col = detect_state_on(df)
        fluid_in_col, fluid_out_col = detect_fluid_in_out(df)
        oat_col = detect_oat(df)
        flow_col, power_col = detect_flow_and_power(df, state_col, exclude_cols=[fluid_in_col, fluid_out_col, oat_col, state_col])

        df = df.rename(columns={
            timestamp_col: "timestamp",
            state_col: "state_on",
            fluid_in_col: "fluid_in",
            fluid_out_col: "fluid_out",
            oat_col: "oat",
            flow_col: "flow_rate",
            power_col: "power_kw"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="raise")
        df = df[df["state_on"]==1].copy()
        df["tons"] = (df["fluid_in"] - df["fluid_out"]) * (df["flow_rate"]/24)
        df["cop"] = (df["tons"] * TONS_CONSTANT) / df["power_kw"]
        df = df[df["power_kw"] >= 20].copy()
        df = df[iqr_filter(df["oat"]) & iqr_filter(df["cop"])].copy()

        if fluid_name == baseline_name:
            X_temp = df['oat'].values.reshape(-1, 1)
            y_temp = df['cop'].values
            lr_temp = LinearRegression().fit(X_temp, y_temp)
            y_pred_temp = lr_temp.predict(X_temp)
            r2_temp = r2_score(y_temp, y_pred_temp)
            if r2_temp < 0.94:
                df_trend_filtered, lr_filtered = filter_by_trend(df, keep_fraction=0.87)
            else:
                df_trend_filtered = df
                lr_filtered = lr_temp
        else:
            df_trend_filtered, lr_filtered = filter_by_trend(df)

        df_trend_filtered["oat_binned"] = np.floor(df_trend_filtered["oat"]).astype(int)
        cop_by_oat = df_trend_filtered.groupby("oat_binned")["cop"].mean().reset_index().sort_values("oat_binned")

        cop_values = cop_by_oat["cop"].values
        downward_mask = np.zeros_like(cop_values, dtype=bool)
        diffs = np.diff(cop_values)
        for i in range(len(cop_values)-1):
            downward_mask[i] = diffs[i] <= 0
        downward_mask[-1] = diffs[-1] <= 0

        cop_down = cop_values[downward_mask]
        oat_down = cop_by_oat["oat_binned"].values[downward_mask]

        if len(cop_down) < 2:
            st_container.warning(f"Not enough valid COP points for {fluid_name}")
            continue

        window_length_adj = min(WINDOW_LENGTH, len(cop_down))
        if window_length_adj % 2 == 0: window_length_adj += 1
        cop_smooth = savgol_filter(cop_down, window_length=window_length_adj, polyorder=POLYORDER)
        slope, intercept = np.polyfit(oat_down, cop_smooth, 1)
        cop_linear = slope * oat_down + intercept
        r2 = r2_score(cop_smooth, cop_linear)

        all_dfs.append({
            "oat": oat_down,
            "cop": cop_smooth,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "name": fluid_name,
            "bin_results": {
                'X': np.array(oat_down),
                'y': np.array(cop_smooth),
                'y_pred': cop_linear,
                'slope': slope,
                'r2': r2
            }
        })
    st_container.success(f"{fluid_name} processing complete! {len(df_trend_filtered)} valid 1-minute COP points")
    return all_dfs, df_trend_filtered

def run():
    # ---------------- Session State ---------------- #
    for key, default in [('analysis_run', False), ('true_original_data', {}), ('aligned_data', {}),
                         ('current_view', 'original'), ('extend_lines', False), ('extend_by', 5),
                         ('plotter', None), ('results_dict', None), ('plot_data_ready', False)]:
        if key not in st.session_state:
            st.session_state[key] = default

    #st.set_page_config(page_title="Plant-HVAC", layout="wide", page_icon="htms_logo.jpg")
    #col1, col2 = st.columns([5,1])
    #with col1: st.title("Plant-HVAC Performance Analysis")
    #with col2: st.image("htms_logo.jpg", width=150)

    phase = st.radio("Select Phase:", ["Phase 1: Analysis", "Phase 2: Alignment"], horizontal=True)

    # -------- Phase 1 --------
    if phase == "Phase 1: Analysis":
        st.subheader("Upload Baseline and Product Fluids")
        baseline_name = st.text_input("Enter Baseline Name", value=st.session_state.get('baseline_name_input', ''))
        st.session_state['baseline_name_input'] = baseline_name
        baseline_files = st.file_uploader("Upload Baseline Data", accept_multiple_files=True, key="baseline_uploader")
        if baseline_files: st.session_state['baseline_paths'] = save_uploaded_files(baseline_files)

        fluid_name = st.text_input("Enter Product Name", value=st.session_state.get('fluid_name_input', ''))
        st.session_state['fluid_name_input'] = fluid_name
        fluid_files = st.file_uploader("Upload Product Data", accept_multiple_files=True, key="product_uploader")
        if fluid_files: st.session_state['fluid_paths'] = save_uploaded_files(fluid_files)

        run_button = st.button("Analyse")
        if run_button:
            baseline_results, baseline_df_filtered = process_fluid(st.session_state.get('baseline_paths', []), baseline_name, baseline_name)
            fluid_results, fluid_df_filtered = process_fluid(st.session_state.get('fluid_paths', []), fluid_name, baseline_name)
            combined_results = {
                baseline_name: baseline_results[0],
                fluid_name: fluid_results[0]
            }
            st.session_state.results_dict = combined_results
            st.session_state.plot_data_ready = True

            density = {baseline_name: oat_density_table(baseline_df_filtered),
                       fluid_name: oat_density_table(fluid_df_filtered)}
            st.subheader("1-Minute COP Points per OAT Bin")
            st.dataframe(build_horizontal_oat_table(density), width="stretch")

            st.session_state.plotter = PlantMultiFluidDynamicPlot(combined_results, default_baseline=baseline_name)

        if st.session_state.plot_data_ready and st.session_state.results_dict:
            if st.session_state.plotter is None:
                st.session_state.plotter = PlantMultiFluidDynamicPlot(st.session_state.results_dict, default_baseline=baseline_name)
            st.session_state.plotter._build_ui()

    # -------- Phase 2 --------
    elif phase == "Phase 2: Alignment":
        st.subheader("Alignment")
        if not st.session_state.results_dict:
            st.error("No data available. Run Phase 1 first.")
        else:
            available_fluids = list(st.session_state.results_dict.keys())
            baseline_name = st.selectbox("Select Baseline Fluid:", options=available_fluids)
            product_options = [f for f in available_fluids if f != baseline_name]
            if product_options:
                fluid_name = st.selectbox("Select Product Fluid to Align:", options=product_options)
                run_button = st.button("Run Alignment", type="primary")
                if run_button:
                    baseline_data = st.session_state.results_dict[baseline_name]
                    fluid_data = st.session_state.results_dict[fluid_name]
                    baseline_slope = baseline_data['bin_results']['slope']
                    aligned_name, aligned_result = align_to_match_baseline_plant(fluid_name, fluid_data, baseline_slope, baseline_name)
                    st.session_state.results_dict[fluid_name] = aligned_result
                    st.session_state.plotter = PlantMultiFluidDynamicPlot(st.session_state.results_dict, default_baseline=baseline_name)
            if st.session_state.plotter:
                st.session_state.plotter._build_ui()
