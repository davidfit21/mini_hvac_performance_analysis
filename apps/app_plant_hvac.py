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


def run():

    # ================== CONFIG (FIRST STREAMLIT CALL) ================== #
    st.set_page_config(
        page_title="Plant-HVAC",
        layout="wide",
        page_icon="htms_logo.jpg"
    )

    # ================== SESSION STATE ================== #
    defaults = {
        "analysis_run": False,
        "true_original_data": {},
        "aligned_data": {},
        "current_view": "original",
        "extend_lines": False,
        "extend_by": 5,
        "plotter": None,
        "results_dict": None,
        "plot_data_ready": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ================== HEADER ================== #
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("Plant-HVAC Performance Analysis")
    with col2:
        st.image("htms_logo.jpg", width=150)

    # ================== CONSTANTS ================== #
    TONS_CONSTANT = 3.51685
    WINDOW_LENGTH = 5
    POLYORDER = 2

    # ================== HELPERS ================== #
    def save_uploaded_files(uploaded_files):
        paths = []
        for f in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
            tmp.write(f.read())
            tmp.close()
            paths.append(Path(tmp.name))
        return paths

    def iqr_filter(series):
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        return (series >= Q1 - 1.5 * IQR) & (series <= Q3 + 1.5 * IQR)

    def _normalize(col):
        return re.sub(r'[^a-z0-9]', '', col.lower())

    def detect_fluid_in_out(df):
        norm = {c: _normalize(c) for c in df.columns}
        IN = ['fluidin', 'entering', 'inlet', 'tin', 'tempin']
        OUT = ['fluidout', 'leaving', 'outlet', 'tout', 'tempout']
        fin = fout = None
        for c, nc in norm.items():
            if any(k in nc for k in IN):
                fin = c
            if any(k in nc for k in OUT):
                fout = c
        if fin and fout:
            return fin, fout
        raise ValueError("Fluid in/out not detected")

    def detect_timestamp(df):
        for c in df.columns:
            if "DATE" in c.upper() or "TIME" in c.upper():
                return c
        return df.columns[0]

    def detect_state_on(df):
        for c in df.select_dtypes(include=np.number):
            if set(df[c].dropna().unique()).issubset({0, 1}):
                return c
        raise ValueError("No ON/OFF column")

    def detect_oat(df):
        for c in df.columns:
            if "OAT" in c.upper():
                return c
        raise ValueError("OAT column not found")

    def detect_flow_and_power(df, state_col, exclude):
        nums = [c for c in df.select_dtypes(include=np.number) if c not in exclude]
        corrs = {c: df[c].corr(df[state_col]) for c in nums}
        flow = max(corrs, key=lambda x: abs(corrs[x]))
        power = [c for c in nums if c != flow]
        if len(power) != 1:
            raise ValueError("Power column ambiguous")
        return flow, power[0]

    def oat_density_table(df):
        df = df.copy()
        df["oat_bin"] = np.floor(df["oat"]).astype(int)
        return df.groupby("oat_bin").size().sort_index()

    def build_horizontal_oat_table(density):
        bins = sorted(set().union(*[s.index for s in density.values()]))
        rows = []
        for name, series in density.items():
            row = {"Fluid": name}
            for b in bins:
                row[f"{b}°C"] = int(series.get(b, 0))
            rows.append(row)
        return pd.DataFrame(rows)

    def filter_by_trend(df, keep_fraction=0.9):
        X = df["oat"].values.reshape(-1, 1)
        y = df["cop"].values
        lr = LinearRegression().fit(X, y)
        res = np.abs(y - lr.predict(X))
        mask = res <= np.percentile(res, keep_fraction * 100)
        df = df[mask].copy()
        df["cop_trend"] = LinearRegression().fit(
            df["oat"].values.reshape(-1, 1),
            df["cop"].values
        ).predict(df["oat"].values.reshape(-1, 1))
        return df

    def process_fluid(paths, name, baseline_name):
        dfs = []
        for f in paths:
            df = pd.read_csv(f).dropna()
            ts = detect_timestamp(df)
            state = detect_state_on(df)
            fin, fout = detect_fluid_in_out(df)
            oat = detect_oat(df)
            flow, power = detect_flow_and_power(df, state, [fin, fout, oat, state])

            df = df.rename(columns={
                ts: "timestamp",
                state: "state_on",
                fin: "fluid_in",
                fout: "fluid_out",
                oat: "oat",
                flow: "flow_rate",
                power: "power_kw"
            })

            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
            df = df[df["state_on"] == 1]
            df["tons"] = (df["fluid_in"] - df["fluid_out"]) * (df["flow_rate"] / 24)
            df["cop"] = (df["tons"] * TONS_CONSTANT) / df["power_kw"]
            df = df[df["power_kw"] >= 20]
            df = df[iqr_filter(df["oat"]) & iqr_filter(df["cop"])]

            df = filter_by_trend(df)
            df["oat_binned"] = np.floor(df["oat"]).astype(int)

            g = df.groupby("oat_binned")["cop"].mean().reset_index()
            oat_vals = g["oat_binned"].values
            cop_vals = g["cop"].values

            mask = np.r_[np.diff(cop_vals) <= 0, True]
            oat_vals, cop_vals = oat_vals[mask], cop_vals[mask]

            w = min(WINDOW_LENGTH | 1, len(cop_vals))
            cop_s = savgol_filter(cop_vals, w, POLYORDER)

            slope, intercept = np.polyfit(oat_vals, cop_s, 1)
            dfs.append({
                "oat": oat_vals,
                "cop": cop_s,
                "slope": slope,
                "intercept": intercept,
                "r2": r2_score(cop_s, slope * oat_vals + intercept),
                "df": df
            })

        return dfs[0], df

    # ================== MAIN FLOW ================== #
    phase = st.radio(
        "Select Phase:",
        ["Phase 1: Analysis", "Phase 2: Alignment"],
        horizontal=True
    )

    # ================== PHASE 1 ================== #
    if phase == "Phase 1: Analysis":
        baseline = st.text_input("Baseline Name")
        base_files = st.file_uploader("Baseline Files", accept_multiple_files=True)
        product = st.text_input("Product Name")
        prod_files = st.file_uploader("Product Files", accept_multiple_files=True)

        if st.button("Analyse"):
            b_res, b_df = process_fluid(save_uploaded_files(base_files), baseline, baseline)
            p_res, p_df = process_fluid(save_uploaded_files(prod_files), product, baseline)

            st.session_state.results_dict = {
                baseline: b_res,
                product: p_res
            }
            st.session_state.plot_data_ready = True

            st.dataframe(
                build_horizontal_oat_table({
                    baseline: oat_density_table(b_df),
                    product: oat_density_table(p_df)
                }),
                use_container_width=True
            )

            st.session_state.plotter = PlantMultiFluidDynamicPlot(
                st.session_state.results_dict,
                default_baseline=baseline
            )

    # ================== PHASE 2 ================== #
    if phase == "Phase 2: Alignment" and st.session_state.results_dict:
        baseline = st.selectbox("Baseline", list(st.session_state.results_dict.keys()))
        product = st.selectbox(
            "Product",
            [k for k in st.session_state.results_dict if k != baseline]
        )

        if st.button("Run Alignment"):
            _, aligned = align_to_match_baseline_plant(
                product,
                st.session_state.results_dict[product],
                st.session_state.results_dict[baseline]["slope"],
                baseline
            )
            st.session_state.results_dict[product] = aligned
            st.session_state.plotter = PlantMultiFluidDynamicPlot(
                st.session_state.results_dict,
                default_baseline=baseline
            )

    # ================== PLOT ================== #
    if st.session_state.plotter:
        st.session_state.plotter._build_ui()
