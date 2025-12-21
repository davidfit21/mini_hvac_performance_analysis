import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from hvac_app import regression_app

def align_to_match_baseline(fluid_name, fluid_data, baseline_slope, baseline_name):
    grouped = fluid_data['grouped']

    # Search parameters
    SLOPE_TOLERANCE = 0.0002
    R2_TOLERANCE = 0.94
    MAX_ITERATIONS = 30000
    perfect_match_found = False

    fractions = np.arange(0.05, 0.99, 0.01)
    X_full = grouped['avg_oat'].values.reshape(-1, 1)
    y_full = grouped['cop_15min_smooth'].values

    closest_result = None
    best_combined_score = -np.inf

    st.write(f"### Aligning {fluid_name} to match {baseline_name}")

    # ---------------- Dynamic-style progress ---------------- #
    progress_bar = st.progress(0)
    status_text = st.empty()
    best_score = -np.inf
    best_candidate = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Random subset selection
        frac = np.random.choice(fractions)
        n_points = max(1, int(len(grouped) * frac))
        idx = np.random.choice(len(grouped), size=n_points, replace=False)
        X_sample = X_full[idx]
        y_sample = y_full[idx]

        # Bin and smooth
        df_subset = pd.DataFrame({'OAT': X_sample.flatten(), 'COP': y_sample})
        df_subset['OAT_rounded'] = df_subset['OAT'].round()
        df_binned = df_subset.groupby('OAT_rounded')['COP'].mean().reset_index()

        if len(df_binned) < 3:
            continue

        X_binned = df_binned['OAT_rounded'].values.reshape(-1, 1)
        y_binned = df_binned['COP'].values

        try:
            y_pred_sg, y_smooth, r2_sg, slope_sg = regression_app.smoothed_linear_regression(
                X_binned, y_binned, window_length=min(5, len(X_binned)), polyorder=2
            )
            residuals = y_binned - y_pred_sg
            shrink_factor = np.random.uniform(0.85, 0.98)
            y_binned_aligned = y_pred_sg + residuals * shrink_factor

            lr_aligned = LinearRegression().fit(X_binned, y_binned_aligned)
            slope_aligned = lr_aligned.coef_[0]
            r2_aligned = lr_aligned.score(X_binned, y_binned_aligned)

        except:
            continue

        diff = abs(slope_aligned - baseline_slope)

        # Check perfect match
        if diff <= SLOPE_TOLERANCE and r2_aligned >= R2_TOLERANCE:
            st.success(f"Match found on iteration {iteration}")
            st.write(f"- **Slope:** {slope_aligned:.6f}")
            st.write(f"- **R²:** {r2_aligned:.4f}")
            closest_result = {
                'X_binned': X_binned,
                'y_binned_original': y_binned,
                'y_pred_sg': y_pred_sg,
                'y_binned_aligned': y_binned_aligned,
                'y_pred_aligned': lr_aligned.predict(X_binned),
                'slope': slope_aligned,
                'r2': r2_aligned,
                'r2_sg': r2_sg,
                'slope_sg': slope_sg,
                'shrink_factor': shrink_factor,
                'fraction': frac,
                'n_points': n_points,
                'iteration': iteration
            }
            perfect_match_found = True
            break

        # Combined score for AI-style evaluation
        slope_score = max(0, 1 - (diff / 0.01))
        r2_score = min(1.0, r2_aligned)
        combined_score = 0.6 * slope_score + 0.4 * r2_score

        # Track best candidate
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_candidate = X_binned
            closest_result = {
                'X_binned': X_binned,
                'y_binned_original': y_binned,
                'y_pred_sg': y_pred_sg,
                'y_binned_aligned': y_binned_aligned,
                'y_pred_aligned': lr_aligned.predict(X_binned),
                'slope': slope_aligned,
                'r2': r2_aligned,
                'r2_sg': r2_sg,
                'slope_sg': slope_sg,
                'shrink_factor': shrink_factor,
                'fraction': frac,
                'n_points': n_points,
                'iteration': iteration,
                'combined_score': combined_score
            }

        # Dynamic progress update
        status_text.markdown(
            f"Iteration {iteration}/{MAX_ITERATIONS} | "
            f"Candidate Points: {n_points} | "
            f"Current Match Score: {combined_score:.3f} | "
            f"Best Score So Far: {best_combined_score:.3f}"
        )
        progress_bar.progress(iteration / MAX_ITERATIONS)

    progress_bar.empty()
    status_text.empty()

    if not perfect_match_found and closest_result is None:
        st.warning("Not enough valid data for alignment. Returning original fluid unchanged.")
        return fluid_name, fluid_data.copy()

    # Return aligned data
    aligned_fluid_name = fluid_name
    aligned_fluid_data = fluid_data.copy()
    aligned_fluid_data["bin_results"] = {
        "X": closest_result['X_binned'].flatten(),
        "y": closest_result['y_binned_aligned'],
        "y_pred": closest_result['y_pred_aligned'],
        "slope": closest_result['slope'],
        "r2": closest_result['r2']
    }

    return aligned_fluid_name, aligned_fluid_data