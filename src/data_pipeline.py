"""
data_pipeline.py — Kelmarsh Wind Farm SCADA Data Pipeline
Progress Report 3 | Ibrahim Alghrabi | MATH 619

Loads, cleans, and prepares the Kelmarsh dataset for PINN training.
Expects CSV files in data/ folder (one per turbine).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ── Kelmarsh turbine specs ──────────────────────────────────────────
RATED_POWER_KW = 2050.0
ROTOR_DIAMETER_M = 92.0
CUT_IN_SPEED = 3.0      # m/s
RATED_SPEED = 12.5       # m/s
CUT_OUT_SPEED = 25.0     # m/s
AIR_DENSITY = 1.225      # kg/m³


# Resolve project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_kelmarsh_data(data_dir=None, turbines=None):
    """
    Load Kelmarsh SCADA CSVs for specified turbines.
    
    The Kelmarsh CSVs have a hash-prefixed timestamp column and 
    use specific column naming conventions. This function handles
    the various column name formats found in the dataset.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if turbines is None:
        turbines = list(range(1, 7))  # Turbines 1-6
    
    all_data = []
    
    for t in turbines:
        # Try common file naming patterns
        possible_names = [
            f"Turbine{t}.csv",
            f"turbine{t}.csv",
            f"Kelmarsh_SCADA_Turbine_{t}.csv",
            f"Kelmarsh_SCADA_Turbine{t}_2021-01-01_-_2021-07-01_228.csv",
        ]
        
        filepath = None
        for name in possible_names:
            candidate = os.path.join(data_dir, name)
            if os.path.exists(candidate):
                filepath = candidate
                break
        
        if filepath is None:
            # Try to find any CSV with the turbine number
            for f in os.listdir(data_dir):
                if f.endswith('.csv') and str(t) in f:
                    filepath = os.path.join(data_dir, f)
                    break
        
        if filepath is None:
            print(f"  [WARN] No CSV found for Turbine {t}, skipping.")
            continue
        
        print(f"  Loading Turbine {t}: {os.path.basename(filepath)}")
        
        # Try reading the CSV — handle both multi-header Kelmarsh and simple formats
        try:
            # First try: simple read (works for synthetic and simple CSVs)
            df = pd.read_csv(filepath)
            
            # Check if we got usable columns
            test_map = _build_column_map(df.columns)
            if len(test_map) < 3:
                # Second try: skip multi-header rows (real Kelmarsh format)
                df = pd.read_csv(filepath, skiprows=[0, 2, 3])
                test_map = _build_column_map(df.columns)
                if len(test_map) < 3:
                    # Third try: just skip first row
                    df = pd.read_csv(filepath, skiprows=[0])
        except Exception as e:
            print(f"    [WARN] Error reading {filepath}: {e}")
            continue
        
        # Fix hash-prefixed timestamp column
        ts_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or c.startswith('#')]
        if ts_col:
            col = ts_col[0]
            df.rename(columns={col: 'timestamp'}, inplace=True)
            if df['timestamp'].dtype == object:
                df['timestamp'] = df['timestamp'].str.lstrip('# ')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Standardize column names — map from Kelmarsh naming to our convention
        col_map = _build_column_map(df.columns)
        df = df.rename(columns=col_map)
        
        # Keep only the columns we need
        required = ['wind_speed', 'wind_direction', 'power', 'yaw', 'pitch', 'rotor_speed']
        available = [c for c in required if c in df.columns]
        
        if len(available) < 4:
            print(f"    [WARN] Only {len(available)} required columns found, skipping.")
            continue
        
        df = df[['timestamp'] + available] if 'timestamp' in df.columns else df[available]
        df['turbine_id'] = t
        all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError(
            f"No valid Kelmarsh CSVs found in '{data_dir}/'. "
            "Please place the Kelmarsh SCADA CSV files there."
        )
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total raw records loaded: {len(combined):,}")
    return combined


def _has_multi_header(filepath):
    """Check if file has the multi-row header typical of Kelmarsh CSVs."""
    with open(filepath, 'r') as f:
        first_line = f.readline()
        return first_line.startswith('#') or 'Timestamp' not in first_line


def _build_column_map(columns):
    """Map Kelmarsh column names to standardized names."""
    col_map = {}
    for c in columns:
        cl = c.lower().strip()
        if ('wind' in cl and 'speed' in cl) or 'wind_speed' in cl or 'windspeed' in cl:
            if 'direction' not in cl and 'dir' not in cl:
                col_map[c] = 'wind_speed'
        elif ('wind' in cl and ('dir' in cl or 'direction' in cl)) or 'wind_dir' in cl:
            col_map[c] = 'wind_direction'
        elif 'power' in cl and 'reactive' not in cl:
            col_map[c] = 'power'
        elif 'yaw' in cl or 'nacelle' in cl:
            col_map[c] = 'yaw'
        elif 'pitch' in cl:
            col_map[c] = 'pitch'
        elif 'rotor' in cl and ('speed' in cl or 'rpm' in cl):
            col_map[c] = 'rotor_speed'
    return col_map


def clean_data(df):
    """
    Clean SCADA data: remove NaNs, filter operating ranges, 
    remove curtailed periods, compute derived features.
    """
    n_before = len(df)
    
    # Drop NaN in critical columns
    critical = ['wind_speed', 'power']
    critical = [c for c in critical if c in df.columns]
    df = df.dropna(subset=critical)
    
    # Filter to valid operating ranges
    df = df[df['wind_speed'].between(0.1, 30.0)]
    df = df[df['power'].between(-50, RATED_POWER_KW + 200)]
    
    # Remove anomalous pitch values (sensor errors)
    if 'pitch' in df.columns:
        df = df[df['pitch'].between(-10, 95)]
    
    # Remove rotor speed outliers
    if 'rotor_speed' in df.columns:
        df = df[df['rotor_speed'].between(0, 20)]
    
    print(f"  Cleaned: {n_before:,} → {len(df):,} records "
          f"({n_before - len(df):,} removed)")
    
    return df.reset_index(drop=True)


def compute_features(df):
    """Compute derived features for PINN training."""
    
    # Yaw misalignment (handle 360° wraparound)
    if 'yaw' in df.columns and 'wind_direction' in df.columns:
        diff = df['wind_direction'] - df['yaw']
        df['yaw_misalignment'] = (diff + 180) % 360 - 180  # range: [-180, 180]
        df['abs_yaw_misalignment'] = df['yaw_misalignment'].abs()
    
    # Operating region classification
    df['region'] = 'below_cutin'
    df.loc[df['wind_speed'] >= CUT_IN_SPEED, 'region'] = 'region2'
    df.loc[df['wind_speed'] >= RATED_SPEED, 'region'] = 'region3'
    
    # Normalize power to [0, 1] for training
    df['power_normalized'] = df['power'] / RATED_POWER_KW
    df['power_normalized'] = df['power_normalized'].clip(0, 1)
    
    # Wind speed normalized
    df['wind_speed_normalized'] = df['wind_speed'] / CUT_OUT_SPEED
    
    # Cosine/sine encoding for angular variables (handles wraparound)
    if 'yaw' in df.columns:
        df['yaw_cos'] = np.cos(np.radians(df['yaw']))
        df['yaw_sin'] = np.sin(np.radians(df['yaw']))
    if 'wind_direction' in df.columns:
        df['wd_cos'] = np.cos(np.radians(df['wind_direction']))
        df['wd_sin'] = np.sin(np.radians(df['wind_direction']))
    if 'pitch' in df.columns:
        df['pitch_normalized'] = df['pitch'] / 90.0  # pitch range ~0-90°
    
    # Thrust coefficient estimate (simplified actuator disk)
    # Ct ≈ 4a(1-a) where a is induction factor
    # For Region 2: Ct ≈ 0.8 (typical); Region 3: Ct decreases
    df['Ct_est'] = 0.8
    df.loc[df['region'] == 'region3', 'Ct_est'] = 0.4
    df.loc[df['region'] == 'below_cutin', 'Ct_est'] = 0.0
    
    return df


def prepare_pinn_inputs(df, feature_cols=None, target_col='power_normalized',
                        test_size=0.2, random_state=42):
    """
    Prepare train/val/test splits with normalization for PINN training.
    
    Returns dict with tensors-ready numpy arrays and the scaler.
    """
    if feature_cols is None:
        feature_cols = ['wind_speed_normalized', 'wd_cos', 'wd_sin',
                        'yaw_cos', 'yaw_sin', 'pitch_normalized']
    
    # Only keep rows with all features available
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        print(f"  [INFO] Using {len(available_cols)}/{len(feature_cols)} features: {available_cols}")
        feature_cols = available_cols
    
    subset = df[feature_cols + [target_col]].dropna()
    
    X = subset[feature_cols].values.astype(np.float32)
    y = subset[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Train / Val / Test split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state)
    
    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"  Features: {feature_cols}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
    }


def run_pipeline(data_dir=None):
    """Run the complete pipeline and return processed data."""
    print("=" * 60)
    print("KELMARSH WIND FARM — DATA PIPELINE")
    print("=" * 60)
    
    print("\n[1/4] Loading data...")
    df = load_kelmarsh_data(data_dir)
    
    print("\n[2/4] Cleaning data...")
    df = clean_data(df)
    
    print("\n[3/4] Computing features...")
    df = compute_features(df)
    
    print("\n[4/4] Preparing PINN inputs...")
    data = prepare_pinn_inputs(df)
    data['df'] = df  # Keep full dataframe for analysis
    
    print("\n✓ Pipeline complete.")
    return data


if __name__ == "__main__":
    data = run_pipeline()
    print(f"\nDataset summary:")
    print(data['df'][['wind_speed', 'power', 'yaw_misalignment']].describe().round(2))
