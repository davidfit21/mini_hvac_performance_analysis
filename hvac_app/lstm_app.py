import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from hvac_app import cop_calculation_app
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
from sklearn.linear_model import LinearRegression

EXPECTED_SLOPE = -0.03  # or take tap water slope as baseline
SLOPE_TOLERANCE = 0.005  # how close synthetic line should be

# --- 1. LSTM Model Definition ---
def create_lstm_model(sequence_length, n_features):
    model = Sequential([
        tf.keras.Input(shape=(sequence_length, n_features)),
        LSTM(32, return_sequences=False),  # Reduced from 128
        Dropout(0.2),  # Increased dropout
        Dense(16, activation='relu'),  # Reduced from 32
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mae', metrics=['mse']) 
    return model


# --- 2. Physics-Enforcing Synthetic Generation ---
def generate_synthetic_data_physics(grouped_data, percent_points=100, sequence_length=15):
    grouped_data = grouped_data.copy()
    grouped_data['avg_oat'] = pd.to_numeric(grouped_data['avg_oat'], errors='coerce')
    grouped_data['cop_15min'] = pd.to_numeric(grouped_data['cop_15min'], errors='coerce')
    grouped_data = grouped_data.dropna(subset=['avg_oat', 'cop_15min'])

    if len(grouped_data) < 2:
        logging.warning(f"Not enough data for LSTM. Need at least 2 points, got {len(grouped_data)}")
        return grouped_data, pd.DataFrame(), []

    n_features = 1
    grouped_data = grouped_data.sort_values('time_bin_mark').reset_index(drop=True)

    oat_mean, oat_std = grouped_data['avg_oat'].mean(), grouped_data['avg_oat'].std()
    cop_mean, cop_std = grouped_data['cop_15min'].mean(), grouped_data['cop_15min'].std()
    grouped_norm = grouped_data.copy()
    grouped_norm['avg_oat'] = (grouped_data['avg_oat'] - oat_mean) / oat_std
    grouped_norm['cop_15min'] = (grouped_data['cop_15min'] - cop_mean) / cop_std
    grouped_norm = grouped_norm.sample(frac=1.0, random_state=42).reset_index(drop=True)

    def prepare_sequences_random(df, seq_len):
        features = df[['avg_oat']].values
        targets = df['cop_15min'].values
        X, y = [], []
        for i in range(len(features)):
            X.append(features[i])
            y.append(targets[i])
        return np.array(X), np.array(y)

    X, y = prepare_sequences_random(grouped_norm, sequence_length)

    oat_mean, oat_std = grouped_data['avg_oat'].mean(), grouped_data['avg_oat'].std()
    #print(f"OAT range in training data: {grouped_norm['avg_oat'].min() * oat_std + oat_mean:.1f} to {grouped_norm['avg_oat'].max() * oat_std + oat_mean:.1f} °C")

    if len(X) == 0:
        logging.warning("No sequences could be created")
        return grouped_data, pd.DataFrame(), []
    
    oat_actual_min = grouped_data['avg_oat'].min()
    oat_actual_max = grouped_data['avg_oat'].max()
    #print(f"Real OAT range: {oat_actual_min:.1f} to {oat_actual_max:.1f} °C")

    # Train LSTM
    model = create_lstm_model(1, n_features)
    model.fit(X, y, epochs=50, batch_size=8, verbose=0, validation_split=0.2)

    # Generate synthetic points
    synthetic_data = []
    total_synthetic_points = int(len(grouped_data) * percent_points / 100)
    target_oats = np.random.uniform(grouped_data['avg_oat'].min(), grouped_data['avg_oat'].max(), total_synthetic_points)

    for target_oat in target_oats:
        target_oat_norm = (target_oat - oat_mean) / oat_std
 
        # Enable training mode during prediction to keep dropout active
        pred_norm = model(np.array([[target_oat_norm]]).reshape(1, 1, 1), training=True)[0][0]
        pred = pred_norm * cop_std + cop_mean

        noise = np.random.normal(0, cop_std * 0.1)  # 10% of original COP std
        pred = pred_norm * cop_std + cop_mean + noise
        synthetic_data.append({'avg_oat': target_oat, 'cop_15min': max(0, pred)})

    # Define synthetic_df here, before using it
    synthetic_df = pd.DataFrame(synthetic_data)

    if synthetic_df.empty:
        logging.warning("LSTM generated zero synthetic points")
        return grouped_data, pd.DataFrame(), []
    
    synthetic_cop_values = synthetic_df['cop_15min'].tolist()

    # Print OAT bin counts for generated data
    synthetic_df['oat_bin'] = pd.cut(synthetic_df['avg_oat'], bins=10)
    #bin_counts = synthetic_df['oat_bin'].value_counts().sort_index()

    #print("Real 15min_COP values by OAT bin:")
    real_binned = grouped_data.copy()
    real_binned['oat_bin'] = real_binned['avg_oat'].round().astype(int)
    for oat_bin in sorted(real_binned['oat_bin'].unique()):
        cop_values = real_binned[real_binned['oat_bin'] == oat_bin]['cop_15min'].values
        #print(f"  OAT {oat_bin}: {cop_values}")

    # Print synthetic COP values by OAT bin  
    #print("Synthetic 15min_COP values by OAT bin:")
    synthetic_binned = synthetic_df.copy()
    synthetic_binned['oat_bin'] = synthetic_binned['avg_oat'].round().astype(int)
    for oat_bin in sorted(synthetic_binned['oat_bin'].unique()):
        cop_values = synthetic_binned[synthetic_binned['oat_bin'] == oat_bin]['cop_15min'].values
        #print(f"  OAT {oat_bin}: {cop_values}")

    return grouped_data, synthetic_df, synthetic_cop_values



# --- 3. Enforcing Slope Constraint ---
def enforce_slope_constraint(df_combined, target_slope, tolerance=0.005):

    X = df_combined['avg_oat'].values.reshape(-1, 1)
    y = df_combined['cop_15min'].values

    model = LinearRegression()
    model.fit(X, y)
    current_slope = model.coef_[0]

    if abs(current_slope - target_slope) <= tolerance:
        return df_combined  # already within tolerance

    # Adjust synthetic points only
    synthetic_idx = df_combined[df_combined['is_synthetic']].index
    if len(synthetic_idx) == 0:
        return df_combined

    # Compute correction factor
    slope_adjust = target_slope - current_slope
    X_mean = X.mean()
    y_correction = slope_adjust * (X[synthetic_idx] - X_mean)

    # Apply correction to synthetic COP values
    df_combined.loc[synthetic_idx, 'cop_15min'] += y_correction.flatten()

    return df_combined


# --- 4. Main Enhancement Entry Point ---
def enhance_fluid_data(original_grouped_data, percent_points=100, original_fluid_name=None):
    # If we have a fluid name and access to session state, apply removals
    try:
        import streamlit as st
        if original_fluid_name and original_fluid_name in st.session_state.removed_points:
            removed_indices = st.session_state.removed_points[original_fluid_name]
            # Filter out removed points before enhancement
            if len(original_grouped_data) > 0:
                mask = [i not in removed_indices for i in range(len(original_grouped_data))]
                original_grouped_data = original_grouped_data.iloc[mask].reset_index(drop=True)
    except:
        pass  # If st isn't available, continue without filtering
    
    original_grouped_data = original_grouped_data.copy()
    original_grouped_data['avg_oat'] = pd.to_numeric(original_grouped_data['avg_oat'], errors='coerce')
    original_grouped_data['cop_15min_smooth'] = pd.to_numeric(original_grouped_data['cop_15min_smooth'], errors='coerce')
    original_grouped_data = original_grouped_data.dropna(subset=['avg_oat', 'cop_15min_smooth'])
    
    if len(original_grouped_data) < 2:
        logging.warning("Not enough valid data for enhancement")
        return None, pd.DataFrame()
    # Preserve original smoothed COP
    original_grouped_data['cop_15min_smooth_orig'] = original_grouped_data['cop_15min_smooth']
    original_grouped_data['cop_15min'] = original_grouped_data['cop_15min_smooth']  # needed for concatenation

    # 1. Generate Raw Synthetic COP
    _, synthetic_df, _ = generate_synthetic_data_physics(
        original_grouped_data.copy(), 
        percent_points=percent_points
    )

    # VALIDATE SYNTHETIC DATA
    if synthetic_df.empty or 'cop_15min' not in synthetic_df.columns:
        logging.warning(f"LSTM generated no valid data. Points requested: {percent_points}%")
        return None, pd.DataFrame()
    
    # Assign synthetic flags and consistent columns
    synthetic_df['is_synthetic'] = True
    synthetic_df['time_bin_mark'] = pd.to_datetime('2026-01-01')  # placeholder time
    synthetic_df['cop_15min'] = synthetic_df['cop_15min']  # LSTM output
    synthetic_df['cop_15min_smooth_orig'] = synthetic_df['cop_15min']  # preserve LSTM raw values

    # Flag original rows
    original_grouped_data['is_synthetic'] = False

    # Columns for concatenation
    cols = ['time_bin_mark', 'avg_oat', 'cop_15min', 'cop_15min_smooth_orig', 'is_synthetic']

    # Concatenate original + synthetic
    combined_df = pd.concat([
        original_grouped_data[cols],
        synthetic_df[cols]
    ], ignore_index=True).sort_values('avg_oat').reset_index(drop=True)

    combined_df['is_synthetic'] = combined_df.get('is_synthetic', False)
    combined_df = enforce_slope_constraint(combined_df, EXPECTED_SLOPE, SLOPE_TOLERANCE)
    combined_df['cop_15min_smooth'] = combined_df['cop_15min']
    return combined_df, synthetic_df