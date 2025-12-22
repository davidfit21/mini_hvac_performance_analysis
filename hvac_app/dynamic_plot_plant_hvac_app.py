# dynamic_plot_plant_hvac_app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from hvac_app.cop_calculation_app import cop_percentage_change

class PlantMultiFluidDynamicPlot:

    def __init__(self, results_dict, default_baseline=None, window_length=5, polyorder=2):

        self.results_dict = results_dict
        self.default_baseline = default_baseline
        self.window_length = window_length
        self.polyorder = polyorder

    # ---------------- Core processing ---------------- #
    def _get_filtered_data(self):
        filtered = {}
        for name, d in self.results_dict.items():
            try:
                if "bin_results" in d and d["bin_results"] is not None:
                    X = np.array(d["bin_results"]["X"])
                    y = np.array(d["bin_results"]["y"])
                    slope = d["bin_results"]["slope"]
                    intercept = d["bin_results"]["y_pred"][0] if len(d["bin_results"]["y_pred"])>0 else 0
                    r2 = d["bin_results"]["r2"]
                    y_fit = slope*X + intercept
                elif "oat" in d and "cop" in d:
                    X = np.array(d["oat"])
                    y = np.array(d["cop"])
                    slope = d.get("slope", 0)
                    intercept = d.get("intercept", 0)
                    r2 = d.get("r2", None)
                    y_fit = slope*X + intercept
                else:
                    st.warning(f"Skipping fluid '{name}': No valid data found.")
                    continue
            except Exception as e:
                st.warning(f"Skipping fluid '{name}' due to error: {e}")
                continue
            filtered[name] = {"X": X, "y": y, "y_fit": y_fit, "slope": slope, "intercept": intercept, "r2": r2}
        return filtered
    
    def extend_regression_line(self, X, y_pred, slope, extend_by):
        if slope is None or len(X) == 0 or y_pred is None or len(y_pred) == 0:
            return None, None
        x_min, x_max = X.min(), X.max()
        X_ext_left = np.arange(x_min - extend_by, x_min, 1)
        X_ext_right = np.arange(x_max + 1, x_max + extend_by + 1, 1)
        X_ext = np.concatenate([X_ext_left, X, X_ext_right])
        Y_ext = slope * (X_ext - X[0]) + y_pred[0]
        return X_ext, Y_ext

    def _compute_metrics(self, baseline_name, filtered_data):
        processed = {}

        baseline_pdata = filtered_data[baseline_name]
        X_base = baseline_pdata["X"].reshape(-1, 1)
        y_base = baseline_pdata["y"]

        slope_base = baseline_pdata.get("slope", 0)
        intercept_base = baseline_pdata.get("intercept", 0)
        y_pred_base = slope_base * X_base.flatten() + intercept_base
        r2_base = baseline_pdata.get("r2", None)

        processed[baseline_name] = {
            "X": X_base.flatten(),
            "y": y_base,
            "y_pred": y_pred_base,
            "y_fit": baseline_pdata.get("y_fit", y_pred_base),
            "r2": r2_base,
            "slope": slope_base,
            "intercept": intercept_base,
            "cop_pct": None
        }

        for name, pdata in filtered_data.items():
            if name == baseline_name:
                continue
            X_arr = pdata["X"].reshape(-1, 1)
            y_arr = pdata["y"]

            from sklearn.linear_model import LinearRegression
            if len(X_arr) >= 2:
                lr = LinearRegression().fit(X_arr, y_arr)
                slope = lr.coef_[0]
                intercept = lr.intercept_
                y_pred = lr.predict(X_arr)
                r2_val = r2_score(y_arr, y_pred)
            else:
                slope = pdata.get("slope", 0)
                intercept = pdata.get("intercept", 0)
                y_pred = y_arr
                r2_val = pdata.get("r2", None)

            processed[name] = {"X": X_arr.flatten(),"y": y_arr,"y_pred": y_pred,"y_fit": y_pred, "r2": r2_val,"slope": slope,"intercept": intercept,"cop_pct": None}

        # COP % delta vs baseline
        if baseline_name in processed:
            df_base = pd.DataFrame({
                "oat_interval": processed[baseline_name]["X"],
                "cop": processed[baseline_name]["y"]
            })
            for name, p in processed.items():
                if name != baseline_name and len(p["X"]) > 0:
                    df_fluid = pd.DataFrame({
                        "oat_interval": p["X"],
                        "cop": p["y"]
                    })
                    p["cop_pct"] = cop_percentage_change(df_fluid, df_base)

        return processed

    # ---------------- UI / Plot ---------------- #
    def _build_ui(self):

        st.subheader("Interactive COP Plot (Plant HVAC)")

        extend_lines = st.checkbox("Extend regression lines", value=False)
        extend_by = st.slider("Extension range (°C)", 0, 15, 5, disabled=not extend_lines)
        st.session_state.extend_lines = extend_lines
        st.session_state.extend_by = extend_by

        fluid_names = list(self.results_dict.keys())
        if not fluid_names:
            st.warning("No plant HVAC data available.")
            return

        baseline_name = st.selectbox(
            "Select baseline fluid:",
            options=fluid_names,
            index=fluid_names.index(self.default_baseline) if self.default_baseline in fluid_names else 0
        )

        filtered_data = self._get_filtered_data()
        processed = self._compute_metrics(baseline_name, filtered_data)

        # ---------------- Plot ---------------- #
        all_X = np.concatenate([p["X"] for p in processed.values()])
        all_y = np.concatenate([p["y"] for p in processed.values()])
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        equations_text = []
        performance_text = []
        legend_added = set()

        for i, name in enumerate(fluid_names):
            pdata = processed[name]
            color = colors[i % len(colors)]
            r2_text = f" (R²={pdata['r2']:.3f})" if pdata['r2'] is not None else ""
            legend_name = f"{name}{r2_text}"

            # Original points
            fig.add_trace(go.Scatter(
                x=pdata["X"],
                y=pdata["y"],
                mode="markers",
                marker=dict(size=6, color=color),
                name=None if name in legend_added else legend_name,
                showlegend=name not in legend_added,
                hovertemplate="OAT: %{x:.1f}°C<br>COP: %{y:.3f}<extra></extra>"
            ))

            X_plot = pdata["X"]
            y_plot = pdata["y"]
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X_plot.reshape(-1,1), y_plot)
            y_fit = lr.predict(X_plot.reshape(-1,1))
            r2_plot = r2_score(y_plot, y_fit) 

            # Always plot the original regression line
            lr = LinearRegression().fit(X_plot.reshape(-1,1), y_plot)
            y_fit = lr.predict(X_plot.reshape(-1,1))
            fig.add_trace(go.Scatter(x=X_plot, y=y_fit, mode='lines', line=dict(color=color, width=3), showlegend=False))

            if extend_lines and pdata["slope"] is not None:
                X_ext, Y_ext = self.extend_regression_line(X=X_plot, y_pred=y_fit, slope=pdata["slope"], extend_by=extend_by)
                if X_ext is not None and Y_ext is not None:
                    fig.add_trace(go.Scatter(x=X_ext, y=Y_ext, mode='lines', line=dict(color=color, width=3), showlegend=False))

            legend_added.add(name)
            r2_text = f" (R²={r2_plot:.3f})"
            legend_name = f"{name}{r2_text}"

            if len(pdata["X"]) >= 2 and pdata["slope"] is not None:
                m = pdata["slope"]
                c = pdata["y_pred"][0] - m * pdata["X"][0]
                equations_text.append(f"{name}: y = {m:.3f}x + {c:.3f}")

            if name != baseline_name and pdata.get("cop_pct") is not None:
                symbol = "+" if pdata["cop_pct"] > 0 else ""
                performance_text.append(f"{name}: ΔCOP = {symbol}{pdata['cop_pct']:.1f}%")

        # Annotations
        annotations = []
        if equations_text:
            annotations.append(dict(
                x=1.2, y=0.3, xref="paper", yref="paper",
                text=f"<b>Equations:</b><br>{'<br>'.join(equations_text)}",
                showarrow=False, align="left",
                bordercolor="black", borderwidth=1, borderpad=4,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=12, color="black")
            ))
        if performance_text:
            annotations.append(dict(
                x=1.2, y=0.7, xref="paper", yref="paper",
                text=f"<b>Performance:</b><br>{'<br>'.join(performance_text)}",
                showarrow=False, align="left",
                bordercolor="green", borderwidth=1, borderpad=4,
                bgcolor="rgba(232,245,232,0.9)",
                font=dict(size=12, color="black")
            ))

        xaxis_range = [min(all_X) - 3, max(all_X) + 3] if len(all_X) > 0 else None
        yaxis_range = [min(all_y) - 1, max(all_y) + 1] if len(all_y) > 0 else None

        fig.update_layout(
            title=dict(text="COP vs OAT (Plant HVAC)", font=dict(color="black"), x=0.5, xanchor="center"),
            xaxis=dict(title=dict(text="OAT (°C)", font=dict(color="black")),
                       tickfont=dict(color="black"), showgrid=True, gridcolor="rgba(0,0,0,0.2)",
                       zeroline=False, showline=False, linecolor="black", range=xaxis_range,
                       tickmode='linear', dtick=1, tick0=min(all_X) if len(all_X) > 0 else 0),
            yaxis=dict(title=dict(text="COP", font=dict(color="black")),
                       tickfont=dict(color="black"), showgrid=True, gridcolor="rgba(0,0,0,0.2)",
                       zeroline=False, showline=False, linecolor="black", range=yaxis_range),
            plot_bgcolor="#f0f0f0", paper_bgcolor="#f0f0f0",
            font=dict(color="black"),
            legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=1.02, font=dict(color="black", size=12)),
            margin=dict(r=200), annotations=annotations
        )
        st.plotly_chart(fig, use_container_width=True, config={'doubleClick': 'reset+autosize', 'displayModeBar': True})