# dynamic_plot_app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from hvac_app import preprocessing_app, cop_calculation_app, regression_app
from hvac_app.cop_calculation_app import cop_percentage_change
st.set_page_config(layout="wide") 

class MultiFluidDynamicPlot:

    def __init__(self, results_dict, default_baseline=None, window_length=5, polyorder=2):
        self.results_dict = results_dict
        self.default_baseline = default_baseline
        self.window_length = window_length
        self.polyorder = polyorder

        fluid_names = list(results_dict.keys())

        if "removed_points" not in st.session_state:
            st.session_state.removed_points = {name: set() for name in fluid_names}
        if "removed_history" not in st.session_state:
            st.session_state.removed_history = {name: [] for name in fluid_names}

        self._build_ui()


    # --------------------- Data Methods --------------------- #
    def _get_filtered_data(self):
        filtered = {}
        for name, data in self.results_dict.items():
        # Get the ORIGINAL fluid name (strip "Enhanced" suffix if present)
            original_name = name.replace(" Enhanced", "")
        
            X = np.array(data["bin_results"]["X"])
            y = np.array(data["bin_results"]["y"])

        # Use original name for removal tracking
            removed_for_fluid = st.session_state.removed_points.get(original_name, set())
            mask = [i not in removed_for_fluid for i in range(len(X))]

            filtered[name] = {
            "X": X[mask],
            "y": y[mask]
            }
        return filtered

    def _compute_metrics(self, baseline_name, filtered_data):
        processed = {}
        for name, pdata in filtered_data.items():
            # Handle empty data case
            if len(pdata["X"]) == 0:
                processed[name] = {
                    "X": np.array([]), "y": np.array([]), "y_pred": [],
                    "r2": None, "slope": None, "cop_pct": None
                }
                continue

            X_arr = pdata["X"].reshape(-1, 1)
            y_arr = pdata["y"]

            if len(X_arr) >= 2:
                y_pred, _, r2, slope = regression_app.smoothed_linear_regression(
                    X_arr, y_arr, window_length=self.window_length, polyorder=self.polyorder
                )
                processed[name] = {
                    "X": X_arr.flatten(),
                    "y": y_arr,
                    "y_pred": y_pred.tolist(),
                    "r2": r2,
                    "slope": slope,
                    "cop_pct": None
                }
            else:
                processed[name] = {
                    "X": X_arr.flatten(),
                    "y": y_arr,
                    "y_pred": y_arr.tolist(),
                    "r2": None,
                    "slope": None,
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
        processed_data = self._compute_metrics(baseline_name, filtered_data)

        CHOSEN_BIN_SIZE = 1
        st.subheader("Raw 1-Second Data Points per OAT Bin")
        
        all_oat_values = set()
        fluid_counts = {}
        
        for name in fluid_names:
            # Get the raw data and bin it
            df_raw = self.results_dict[name]["df_raw"]
            df_raw_binned = cop_calculation_app.oat_binning(df_raw.copy(), bin_size=CHOSEN_BIN_SIZE, temp_col='T3')
            raw_counts = df_raw_binned.groupby('oat_interval').size().reset_index(name=name)
            
            # Convert interval strings to integer OAT values
            counts_dict = {}
            for _, row in raw_counts.iterrows():
                oat_interval = row['oat_interval']
                if isinstance(oat_interval, str) and '-' in oat_interval:
                    oat_value = int(float(oat_interval.split('-')[0]))
                else:
                    oat_value = int(float(oat_interval))
                counts_dict[oat_value] = row[name]
                all_oat_values.add(oat_value)
            
            fluid_counts[name] = counts_dict
        
        sorted_oat_values = sorted(all_oat_values)
        
        table_data = []
        for name in fluid_names:
            row = {"Fluid": name}
            for oat_value in sorted_oat_values:
                count = fluid_counts[name].get(oat_value, 0)
                row[f"{oat_value}°C"] = f"{count:,}"
            table_data.append(row)
        
        # Display as table
        if table_data:
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True)
        else:
            st.info("No raw data counts available.")

        # --------------------- Plot --------------------- #
        all_X = np.concatenate([filtered_data[name]["X"] for name in fluid_names]) if fluid_names else []
        all_y = np.concatenate([filtered_data[name]["y"] for name in fluid_names]) if fluid_names else []

        xaxis_range = [min(all_X) - 2, max(all_X) + 2] if len(all_X) > 0 else None
        yaxis_range = [min(all_y) - 0.2, max(all_y) + 0.2] if len(all_y) > 0 else None

        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Calculate metrics for annotations
        processed = self._compute_metrics(baseline_name, filtered_data)
        equations_text = []
        performance_text = []

        for i, name in enumerate(fluid_names):
            pdata = filtered_data[name]
            color = colors[i % len(colors)]

            # Handle single point vs multiple points
            if len(pdata["X"]) == 1:
                # Single point - show scatter in legend
                fig.add_trace(go.Scatter(
                    x=pdata["X"],
                    y=pdata["y"],
                    mode="markers",
                    name=name,
                    marker=dict(size=9, color=color, symbol='circle'),
                    hovertemplate=f"OAT: %{{x}}°C<br>COP: %{{y:.3f}}<extra></extra>",
                    showlegend=True  # Force show in legend for single points
                ))
                # Add single point to equations display
                equations_text.append(f"{name}: R², slope = N/A")
                
            elif len(pdata["X"]) >= 2:
                # Multiple points - normal behavior, Scatter points - no legend
                fig.add_trace(go.Scatter(
                    x=pdata["X"],
                    y=pdata["y"],
                    mode="markers",
                    name=name,
                    marker=dict(size=9, color=color, symbol='circle'),
                    hovertemplate=f"OAT: %{{x}}°C<br>COP: %{{y:.3f}}<extra></extra>",
                    showlegend=False  # Don't show scatter in legend (regression line will show instead)
                ))

                # Regression line for 2+ points
                pproc = processed[name]
                hover_text = [f"OAT: {x:.1f}°C<br>Predicted COP: {y:.3f}<extra></extra>"
                            for x, y in zip(pproc["X"], pproc["y_pred"])]
                
                # Add R² to legend name
                r2_str = f"{pproc['r2']:.4f}" if pproc["r2"] is not None else "N/A"
                legend_name = f"{name} (R² = {r2_str})"
                
                fig.add_trace(go.Scatter(
                    x=pproc["X"],
                    y=pproc["y_pred"],
                    mode="lines",
                    name=legend_name,
                    line=dict(color=color, width=3, dash='solid'),
                    showlegend=True,
                    hovertemplate="%{text}",
                    text=hover_text
                ))
                
                # Prepare equations text
                if pproc["slope"] is not None and len(pproc["X"]) > 0 and len(pproc["y_pred"]) > 0:
                    m = pproc["slope"]
                    x_sample = pproc["X"][0]
                    y_sample = pproc["y_pred"][0]
                    c = y_sample - m * x_sample
                    equation_str = f"{name}: y = {m:.3f}x + {c:.3f}"
                    equations_text.append(equation_str)

            # Prepare performance text (for both single and multiple points)
            if name != baseline_name and processed[name]["cop_pct"] is not None:
                symbol = "+" if processed[name]["cop_pct"] > 0 else ""
                perf_str = f"{name}: ΔCOP = {symbol}{processed[name]['cop_pct']:.1f}%"
                performance_text.append(perf_str)

        equations_box = "<br>".join(equations_text)
        performance_box = "<br>".join(performance_text)
        annotations = []

        # Equations annotation (directly below legend)
        if equations_text:
            annotations.append(dict(
                x=1.2,
                y=0.3,
                xref="paper",
                yref="paper",
                text=f"<b>Equations:</b><br>{equations_box}",
                showarrow=False,
                align="left",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=12, color="black")
            ))

        # Performance annotation (below equations)
        if performance_text:
            annotations.append(dict(
                x=1.2,
                y=0.7,
                xref="paper",
                yref="paper", 
                text=f"<b>Performance:</b><br>{performance_box}",
                showarrow=False,
                align="left",
                bordercolor="green",
                borderwidth=1,
                borderpad=4,
                bgcolor="rgba(232,245,232,0.9)",
                font=dict(size=12, color="black")
            ))

        fig.update_layout(
            title=dict(text="COP vs OAT", font=dict(color="black"), x=0.5, xanchor="center"),
            xaxis=dict(
                title=dict(text="OAT (°C)", font=dict(color="black")),
                tickfont=dict(color="black"),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.2)",
                zeroline=False,
                showline=False,
                linecolor="black",
                range=xaxis_range,
                tickmode='linear',
                dtick=1,
                tick0=min(all_X) if len(all_X) > 0 else 0 
            ),
            yaxis=dict(
                title=dict(text="COP", font=dict(color="black")),
                tickfont=dict(color="black"),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.2)",
                zeroline=False,
                showline=False,
                linecolor="black",
                range=yaxis_range
            ),
            plot_bgcolor="#f0f0f0",
            paper_bgcolor="#f0f0f0",
            font=dict(color="black"),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=1.02,
                font=dict(color="black", size=12)
            ),
            margin=dict(r=350),
            annotations=annotations
        )

        st.plotly_chart(fig, use_container_width=True, config={'doubleClick': 'reset+autosize', 'displayModeBar': True})

        # --------------------- Remove Points / Undo --------------------- #
        st.write("### Remove Points")

        for name in fluid_names:
        # Get original name for removal tracking
            original_name = name.replace(" Enhanced", "")
    
            pdata = self.results_dict[name]["bin_results"]
            n_points = len(pdata["X"])

            if n_points == 0:
                continue

            point_labels = [f"OAT={pdata['X'][i]:.1f}°C, COP={pdata['y'][i]:.3f}" for i in range(n_points)]

            selected_to_remove = st.multiselect(
        f"Select points to remove from {name}:",
        options=point_labels,
        key=f"temp_select_{name}"  # Keep unique key per displayed name
    )

    # Remove points button - use original_name for tracking
            if st.button(f"Remove selected points from {name}", key=f"remove_{name}"):
                if selected_to_remove:
                    to_remove_idx = {point_labels.index(lbl) for lbl in selected_to_remove}
            
            # Use original_name for removal tracking
                    if original_name not in st.session_state.removed_points:
                        st.session_state.removed_points[original_name] = set()
                    if original_name not in st.session_state.removed_history:
                        st.session_state.removed_history[original_name] = []
            
                    st.session_state.removed_points[original_name].update(to_remove_idx)
                    st.session_state.removed_history[original_name].append(to_remove_idx.copy())
                    st.rerun()

        # Undo button - use original_name for tracking
            if st.button(f"Undo last removal for {name}", key=f"undo_{name}"):
                if original_name in st.session_state.removed_history and st.session_state.removed_history[original_name]:
                    last_removed = st.session_state.removed_history[original_name].pop()
                    st.session_state.removed_points[original_name].difference_update(last_removed)
                    st.rerun()
