# hvac_analyzer/cop_calculation.py
import pandas as pd
import numpy as np
from . import preprocessing_app # Use relative import

import numpy as np
import pandas as pd

#def oat_binning(df, bin_size=1, temp_col='avg_oat'):

#    if df.empty:
#        return df
    
    # Use numpy.floor instead of .floor() method
#    min_temp = np.floor(df[temp_col].min())
#    max_temp = np.ceil(df[temp_col].max())
    
    # Create bins
#    bins = np.arange(min_temp, max_temp + bin_size, bin_size)
#    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    # Bin the data
#    df['oat_interval'] = pd.cut(
#        df[temp_col], 
#        bins=bins, 
#        labels=labels, 
#        include_lowest=True
#    )
    
#    return df

def oat_binning(df, bin_size, temp_col='avg_oat'):
    df['oat_interval'] = np.round(df[temp_col] / bin_size) * bin_size
    df['count_per_bin'] = df.groupby('oat_interval')[temp_col].transform('count')
    return df


# Keep your other functions the same...
def compute_delta_energy(df):
    """
    Compute delta energy for both energy columns.
    """
    df = df.copy()
    df['delta_energy1'] = df['max_energy1'] - df['min_energy1']
    df['delta_energy2'] = df['max_energy2'] - df['min_energy2']
    return df

def remove_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Placeholder for Z-score removal, leveraging the function from preprocessing.
    """
    return preprocessing_app.remove_outliers_zscore(df, column, threshold)


def cop_percentage_change(df_fluid: pd.DataFrame, df_base: pd.DataFrame) -> float | None:
    """
    Calculates the average percentage change in COP between the fluid and the baseline
    for common temperature bins.
    
    NOTE: Assumes both DataFrames have 'oat_interval' (numeric) and 'cop' columns.
    """
    if df_fluid.empty or df_base.empty:
        return None

    # Merge the two dataframes on the OAT interval
    merged = pd.merge(
        df_fluid.rename(columns={'cop': 'cop_fluid'}),
        df_base.rename(columns={'cop': 'cop_base'}),
        on='oat_interval',
        how='inner'
    )
    
    if merged.empty:
        return None

    # Calculate the percentage change: (Fluid - Base) / Base * 100
    # Filter out any zero or near-zero baseline COP values to prevent division by zero
    merged = merged[merged['cop_base'].abs() > 1e-6] 
    
    merged['pct_change'] = (merged['cop_fluid'] - merged['cop_base']) / merged['cop_base'] * 100
    
    # Return the mean percentage change across all common OAT intervals
    return merged['pct_change'].mean()