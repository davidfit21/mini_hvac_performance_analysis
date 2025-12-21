import numpy as np
import pandas as pd
from scipy.stats import zscore

def add_datetime(df):
    """Combines Date and Time columns, converts to datetime objects, and sets the index."""
    
    # Check if Date is already a datetime
    if pd.api.types.is_datetime64_any_dtype(df['Date']):
        # Already datetime - just set as index and drop original columns
        df = df.set_index('Date')
        df = df.drop(columns=['Time'])
        df.index.name = 'Datetime'
    else:
        # Original logic for string dates
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                       format='%d/%m/%Y %H:%M:%S')
        df = df.set_index('Datetime')
        df = df.drop(columns=['Date', 'Time'])
    
    #df = df.sort_index()
    return df


def resample_to_minute(df):
    """Resamples the DataFrame to 1-minute intervals (mean)."""
    df_1min = df.resample('1min').mean()
    #df_1min['1_min_helper'] = df_1min.index.floor('1min')
    df_1min['1_min_helper'] = df_1min.index.floor('1min')
    
    return df_1min

def iqr_filter(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

def remove_outliers_zscore(df, column, threshold=2):
    z_scores = np.abs(zscore(df[column], nan_policy='omit'))
    filtered = df[z_scores < threshold]
    return filtered