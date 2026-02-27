import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_profile_monthly():
    input_file = r'd:\Projects\AWE\dataset\ods031.csv'
    output_base_dir = r'd:\Projects\AWE\output\power\wind'
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, sep=';')
    
    # Parse Datetime
    # Format: 2025-12-31T23:45:00+01:00
    # Try inferred format first, then specific if needed
    # Convert to UTC to handle timezones uniformly
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True, format='ISO8601')
    
    # Check if we have valid datetimes
    if df['Datetime'].isnull().all():
        print("Warning: All datetimes are NaT. Trying mixed format.")
        df = pd.read_csv(input_file, sep=';')
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)

    # Convert to a fixed timezone (e.g., Europe/Brussels as in original data) or keep UTC
    # The issue "2024-12" appearing is likely due to 2025-01-01 00:00:00+01:00 being 2024-12-31 23:00:00 UTC
    # We should convert to the local time of the dataset (Europe/Brussels) to respect the original monthly boundaries
    df['Datetime'] = df['Datetime'].dt.tz_convert('Europe/Brussels')

    # Extract Data
    col_name = 'Measured & Upscaled'
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df = df.dropna(subset=[col_name, 'Datetime'])
    
    # Sort chronologically
    df = df.sort_values('Datetime')
    
    # Scale to 0 - 40 MW
    data = df[col_name].values
    d_min = np.min(data)
    d_max = np.max(data)
    
    print(f"Original Data Range: {d_min} to {d_max}")
    
    if d_max == d_min:
        norm_data = np.zeros_like(data)
    else:
        norm_data = (data - d_min) / (d_max - d_min)
        
    scaled_data = norm_data * 40.0e6
    df['P_ref'] = scaled_data
    
    # Group by Month
    # Sort first
    df = df.sort_values('Datetime')
    
    # Check data range
    print(f"Data Time Range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"Total Rows: {len(df)}")
    
    grouped = df.groupby(df['Datetime'].dt.to_period('M'))
    
    for period, group in grouped:
        month_str = period.strftime('%Y-%m')
        print(f"Processing {month_str}...")
        
        # Interpolate to 1min
        # Original: 15min
        # Get seconds relative to month start
        start_time = group['Datetime'].iloc[0]
        t_original = (group['Datetime'] - start_time).dt.total_seconds().values
        
        P_original = group['P_ref'].values
        
        # New time grid (1min = 60s)
        # Duration based on actual data span for this month
        duration = t_original[-1]
        
        # Create full minute grid for the month
        # Start at 0, go up to duration
        t_new = np.arange(0, duration + 60.0, 60.0)
        
        # Filter t_new to not exceed max original time if we don't want extrapolation
        # t_new = t_new[t_new <= t_original[-1]]
        
        P_interp = np.interp(t_new, t_original, P_original)
        
        # Save
        month_dir = os.path.join(output_base_dir)
        if not os.path.exists(month_dir):
            os.makedirs(month_dir)
            
        filename = f"wind_power_{month_str}_1min.csv"
        output_path = os.path.join(month_dir, filename)
        
        out_df = pd.DataFrame({
            't': t_new,
            'P_ref': P_interp
        })
        
        out_df.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(out_df)} points)")

if __name__ == "__main__":
    process_profile_monthly()
