# dynamic_plot_app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from hvac_app import preprocessing_app, cop_calculation_app, regression_app
from hvac_app.cop_calculation_app import cop_percentage_change

class MultiFluidDynamicPlot:

    def __init__(self, results_dict, default_baseline=None, window_length=5, polyorder=2):
        self.results_dict = results_dict
        self.default_baseline = default_baseline
        self.window_length = window_length
        self.polyorder = polyorder
        self._build_ui()

    def _get_filtered_data(self):
        filtered = {}
        for name, data in self.results_dict.items():
            try:
                X = np.array(data["bin_results"]["X"])
                y = np.array(data["bin_results"]["y"])
            except KeyError:
                st.warning(f"Skipping fluid '{name}': Data structure is incomplete.")
                continue
            filtered[name] = {"X": X, "y": y}
        return filtered

    def _compute_metrics(self, baseline_name, filtered_data):
        processed = {}

        baseline_pdata = filtered_data[baseline_name]
        X_base = baseline_pdata["X"].reshape(-1, 1)
        y_base = baseline_pdata["y"]

        y_pred_base, _, r2_base, slope_base = regression_app.smoothed_linear_regression(
            X_base, y_base, window_length=self.window_length, polyorder=self.polyorder
        )

        processed[baseline_name] = {
            "X": X_base.flatten(),
            "y": y_base,
            "y_pred": y_pred_base,
            "r2": r2_base,
            "slope": slope_base,
            "cop_pct": None
        }

        for name, pdata in filtered_data.items():
            if name == baseline_name:
                continue
            X_arr = pdata["X"].reshape(-1, 1)
            y_arr = pdata["y"]

            if len(X_arr) >= 2:
                y_pred, _, r2, slope = regression_app.smoothed_linear_regression(
                    X_arr, y_arr, window_length=self.window_length, polyorder=self.polyorder
                )
            else:
                slope = slope_base
                y_pred = y_arr.flatten()
                r2 = None

            processed[name] = {
                "X": X_arr.flatten(),
                "y": y_arr,
                "y_pred": y_pred,
                "r2": r2,
                "slope": slope,
                "cop_pct": None
            }

        # Calculate COP percentage changes
        if baseline_name in processed and len(processed[baseline_name]["X"]) > 0:
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

    # --------------------- UI / Plot --------------------- #
    def _build_ui(self):
        st.subheader("Interactive COP Plot")

        extend_lines = st.checkbox("Extend regression lines", value=False)
        extend_by = st.slider("Extension range (°C)", 0, 15, 5, disabled=not extend_lines)

        fluid_names = list(self.results_dict.keys())
        if not fluid_names:
            st.warning("No fluid data available.")
            return

        baseline_name = st.selectbox(
            "Select baseline fluid:",
            options=fluid_names,
            index=fluid_names.index(self.default_baseline) if self.default_baseline in fluid_names else 0
        )

        filtered_data = self._get_filtered_data()
        processed = self._compute_metrics(baseline_name, filtered_data)

        # ---------------- Raw Data Table ---------------- #
        CHOSEN_BIN_SIZE = 1
        st.subheader("15-Minute COP Points per OAT Bin")
        all_oat_values = set()
        fluid_counts = {}

        for name in fluid_names:
            grouped = self.results_dict[name]["grouped"]
            grouped_binned = cop_calculation_app.oat_binning(grouped.copy(), bin_size=CHOSEN_BIN_SIZE, temp_col='avg_oat')
            counts_dict = grouped_binned.groupby('oat_interval').size().to_dict()
            for oat in counts_dict.keys():
                all_oat_values.add(int(float(oat)) if isinstance(oat, (float, int, str)) else oat)
            fluid_counts[name] = counts_dict

        sorted_oat_values = sorted(all_oat_values)
        table_data = []
        for name in fluid_names:
            row = {"Fluid": name}
            for oat_value in sorted_oat_values:
                count = fluid_counts[name].get(oat_value, 0)
                row[f"{oat_value}°C"] = f"{count:,}"
            table_data.append(row)

        if table_data:
        #    with st.container():
        #        st.dataframe(pd.DataFrame(table_data))

            st.dataframe(pd.DataFrame(table_data), width=2000)

        else:
            st.info("No 15-min COP points available.")

        # ---------------- Plot ---------------- #
        all_X = np.concatenate([filtered_data[name]["X"] for name in fluid_names]) if fluid_names else []
        all_y = np.concatenate([filtered_data[name]["y"] for name in fluid_names]) if fluid_names else []
        xaxis_range = [min(all_X) - 2, max(all_X) + 2] if len(all_X) > 0 else None
        yaxis_range = [min(all_y) - 0.2, max(all_y) + 0.2] if len(all_y) > 0 else None
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        equations_text = []
        performance_text = []
        legend_added = set()

        for i, name in enumerate(fluid_names):
            pdata = filtered_data[name]
            pproc = processed[name]
            color = colors[i % len(colors)]
            r2_text = f" (R²={pproc['r2']:.3f})" if pproc['r2'] is not None else ""
            legend_name = f"{name}{r2_text}"

            # ---------------- Original points ---------------- #
            fig.add_trace(go.Scatter(
                x=pproc["X"],
                y=pproc["y"],
                mode="markers",
                marker=dict(size=6, color=color),
                name=None if name in legend_added else legend_name,
                showlegend=name not in legend_added,
                hovertemplate="OAT: %{x:.1f}°C<br>COP: %{y:.3f}<extra></extra>"
            ))

            # ---------------- Plot Regression Line ---------------- #
            fig.add_trace(go.Scatter(
                x=pproc["X"],
                y=pproc["y_pred"],
                mode="lines",
                line=dict(color=color, width=3),
                name=None,
                showlegend=False
                ))
            
            # Determine which points to plot
            if extend_lines and len(pproc["X"]) > 0 and pproc["slope"] is not None:
                x_min, x_max = pproc["X"].min(), pproc["X"].max()
                x_ext_left = np.arange(x_min - extend_by, x_min, 1)
                x_ext_right = np.arange(x_max + 1, x_max + extend_by + 1, 1)
                x_plot = np.concatenate([x_ext_left, pproc["X"], x_ext_right])
                y_plot = pproc["slope"] * (x_plot - pproc["X"][0]) + pproc["y_pred"][0]
            else:
                x_plot = pproc["X"]
                y_plot = pproc["y_pred"]

            # Compute COP % difference for non-baseline fluids
            if name != baseline_name and len(x_plot) > 0:
                df_base = pd.DataFrame({
                    "oat_interval": processed[baseline_name]["X"],
                    "cop": processed[baseline_name]["y"]
                })
                df_fluid = pd.DataFrame({
                    "oat_interval": x_plot,
                    "cop": y_plot
                })
                pproc["cop_pct"] = cop_percentage_change(df_fluid, df_base)

            # Plot line + points
            fig.add_trace(go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines",#+markers",
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color),
                name=None if name in legend_added else legend_name,
                showlegend=False,#name not in legend_added,
                hovertemplate="OAT: %{x:.1f}°C<br>COP: %{y:.3f}<extra></extra>"
            ))

            legend_added.add(name)

            # ---------------- Equations and Performance ---------------- #
            if len(pproc["X"]) >= 2 and pproc["slope"] is not None:
                m = pproc["slope"]
                c = pproc["y_pred"][0] - m * pproc["X"][0]
                equations_text.append(f"{name}: y = {m:.3f}x + {c:.3f}")

            if name != baseline_name and pproc.get("cop_pct") is not None:
                symbol = "+" if pproc["cop_pct"] > 0 else ""
                performance_text.append(f"{name}: ΔCOP = {symbol}{pproc['cop_pct']:.1f}%")

        # ---------------- Annotations ---------------- #
        annotations = []
        if equations_text:
            annotations.append(dict(
                x=1.2, y=0.3, xref="paper", yref="paper",
                text=f"<b>Equations:</b><br>{'<br>'.join(equations_text)}",
                showarrow=False, align="left", bordercolor="black",
                borderwidth=1, borderpad=4, bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=12, color="black")
            ))
        if performance_text:
            annotations.append(dict(
                x=1.2, y=0.7, xref="paper", yref="paper",
                text=f"<b>Performance:</b><br>{'<br>'.join(performance_text)}",
                showarrow=False, align="left", bordercolor="green",
                borderwidth=1, borderpad=4, bgcolor="rgba(232,245,232,0.9)",
                font=dict(size=12, color="black")
            ))

        fig.update_layout(
            title=dict(text="COP vs OAT", font=dict(color="black"), x=0.5, xanchor="center"),
            xaxis=dict(
                title=dict(text="OAT (°C)", font=dict(color="black")),
                tickfont=dict(color="black"), showgrid=True, gridcolor="rgba(0,0,0,0.2)",
                zeroline=False, showline=False, linecolor="black", range=xaxis_range,
                tickmode='linear', dtick=1, tick0=min(all_X) if len(all_X) > 0 else 0
            ),
            yaxis=dict(
                title=dict(text="COP", font=dict(color="black")),
                tickfont=dict(color="black"), showgrid=True, gridcolor="rgba(0,0,0,0.2)",
                zeroline=False, showline=False, linecolor="black", range=yaxis_range
            ),
            plot_bgcolor="#f0f0f0", paper_bgcolor="#f0f0f0",
            font=dict(color="black"),
            legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=1.02,
                        font=dict(color="black", size=12)),
            margin=dict(r=200), annotations=annotations
        )

        st.plotly_chart(fig, width='stretch', config={'doubleClick': 'reset+autosize', 'displayModeBar': True})
