"""
Wind Turbine SCADA Data Exploration - Kelmarsh Wind Farm
=========================================================
This script explores the Kelmarsh wind farm SCADA data and creates
visualizations for Progress Report 2.

Author: Ibrahim Alghrabi
Project: Physics-Informed Neural Networks for Wind Turbine Control
Data Source: Kelmarsh Wind Farm (Zenodo)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ============================================================
# CONFIGURATION - Update these paths for your system
# ============================================================
DATA_DIR = Path("data/Kelmarsh_SCADA_2021_3087")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column mapping from Kelmarsh to our standard names
COLUMN_MAPPING = {
    '# Date and time': 'timestamp',
    'Wind speed (m/s)': 'wind_speed',
    'Wind direction (°)': 'wind_direction',
    'Nacelle position (°)': 'yaw_angle',
    'Blade angle (pitch position) A (°)': 'pitch_angle',
    'Power (kW)': 'power',
    'Rotor speed (RPM)': 'rotor_speed',
    'Nacelle ambient temperature (°C)': 'ambient_temp',
}


def load_kelmarsh_data(data_dir=DATA_DIR):
    """
    Load all Kelmarsh turbine SCADA data files.

    Parameters:
    -----------
    data_dir : Path
        Directory containing the Kelmarsh CSV files

    Returns:
    --------
    pd.DataFrame
        Combined data from all turbines
    """
    data_dir = Path(data_dir)
    all_data = []

    print("=" * 60)
    print("LOADING KELMARSH WIND FARM DATA")
    print("=" * 60)

    # Find all turbine data files
    turbine_files = sorted(data_dir.glob("Turbine_Data_Kelmarsh_*.csv"))

    if not turbine_files:
        raise FileNotFoundError(f"No turbine data files found in {data_dir}")

    for filepath in turbine_files:
        # Extract turbine number from filename
        turbine_num = int(filepath.stem.split("_")[3])

        print(f"Loading Turbine {turbine_num}: {filepath.name}")

        # Read CSV, skipping the 9 header comment lines
        df = pd.read_csv(filepath, skiprows=9)

        # Select and rename columns
        cols_to_keep = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
        df = df[cols_to_keep].rename(columns=COLUMN_MAPPING)

        # Add turbine ID
        df['turbine_id'] = turbine_num

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        all_data.append(df)
        print(f"  - Loaded {len(df)} records")

    # Combine all turbines
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"\nTotal records: {len(combined_df)}")
    print(f"Turbines: {combined_df['turbine_id'].nunique()}")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

    return combined_df


def clean_data(df):
    """
    Clean the SCADA data.

    - Remove NaN values in critical columns
    - Filter to valid operating range
    - Remove obvious outliers
    """
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    initial_count = len(df)

    # Critical columns that must have values
    critical_cols = ['wind_speed', 'power', 'yaw_angle', 'pitch_angle']

    # Remove rows with NaN in critical columns
    df_clean = df.dropna(subset=critical_cols)
    print(f"After removing NaN: {len(df_clean)} / {initial_count}")

    # Filter to valid operating range
    df_clean = df_clean[
        (df_clean['wind_speed'] >= 0) &
        (df_clean['wind_speed'] <= 30) &
        (df_clean['power'] >= -50) &  # Small negative for parasitic loads
        (df_clean['power'] <= 2200)   # Slightly above rated (2050 kW)
    ]
    print(f"After range filter: {len(df_clean)}")

    # Remove curtailed/shutdown periods (power near zero when wind is good)
    # Keep data where either wind < 3 (below cut-in) or power > 10 kW
    df_clean = df_clean[
        (df_clean['wind_speed'] < 3) |
        (df_clean['power'] > 10)
    ]
    print(f"After removing curtailment: {len(df_clean)}")

    print(f"\nFinal dataset: {len(df_clean)} records ({len(df_clean)/initial_count*100:.1f}%)")

    return df_clean


def explore_data_statistics(df):
    """Generate comprehensive data statistics."""

    print("\n" + "=" * 60)
    print("DATA STATISTICS SUMMARY")
    print("=" * 60)

    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of Turbines: {df['turbine_id'].nunique()}")

    # Numerical statistics
    numeric_cols = ['wind_speed', 'wind_direction', 'power',
                    'yaw_angle', 'pitch_angle', 'rotor_speed']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    print("\nNumerical Statistics:")
    print("-" * 60)
    stats = df[numeric_cols].describe()
    print(stats.round(2))

    # Missing values
    print("\nMissing Values:")
    print("-" * 60)
    missing = df[numeric_cols].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({'Count': missing, 'Percentage': missing_pct}))

    # Save statistics
    stats.to_csv(OUTPUT_DIR / "data_statistics.csv")
    print(f"\n✓ Statistics saved to {OUTPUT_DIR / 'data_statistics.csv'}")

    return stats


def plot_power_curve(df, save=True):
    """Plot the power curve (power vs wind speed)."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, df['turbine_id'].nunique()))
    for i, tid in enumerate(sorted(df['turbine_id'].unique())):
        mask = df['turbine_id'] == tid
        ax1.scatter(df.loc[mask, 'wind_speed'], df.loc[mask, 'power'],
                   alpha=0.3, s=5, label=f'Turbine {tid}', color=colors[i])

    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Power Curve - Kelmarsh Wind Farm')
    ax1.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='Cut-in (3 m/s)')
    ax1.axvline(x=12, color='g', linestyle='--', alpha=0.5, label='Rated (12 m/s)')
    ax1.axhline(y=2050, color='orange', linestyle='--', alpha=0.5, label='Rated Power')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, 25)
    ax1.set_ylim(-100, 2300)

    # Binned average
    ax2 = axes[1]
    df['ws_bin'] = pd.cut(df['wind_speed'], bins=np.arange(0, 26, 1))
    binned = df.groupby('ws_bin', observed=True)['power'].agg(['mean', 'std']).reset_index()
    binned['ws_mid'] = binned['ws_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

    ax2.errorbar(binned['ws_mid'], binned['mean'], yerr=binned['std'],
                 fmt='o-', capsize=3, color='navy', label='Mean ± Std')
    ax2.fill_between(binned['ws_mid'],
                     binned['mean'] - binned['std'],
                     binned['mean'] + binned['std'],
                     alpha=0.3)
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Binned Power Curve with Uncertainty')
    ax2.set_xlim(0, 25)
    ax2.axhline(y=2050, color='orange', linestyle='--', alpha=0.5, label='Rated Power')
    ax2.legend()

    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / 'power_curve.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'power_curve.png'}")

    plt.close()
    return fig


def plot_yaw_analysis(df, save=True):
    """Analyze yaw misalignment and its effect on power."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate yaw error (handle wraparound at 360°)
    df['yaw_error'] = df['wind_direction'] - df['yaw_angle']
    df['yaw_error'] = (df['yaw_error'] + 180) % 360 - 180

    # Yaw error distribution
    ax1 = axes[0, 0]
    ax1.hist(df['yaw_error'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Yaw Misalignment (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Yaw Misalignment')
    ax1.set_xlim(-60, 60)

    # Yaw error vs power (binned)
    ax2 = axes[0, 1]
    # Filter to region 2 wind speeds for fair comparison
    region2 = df[(df['wind_speed'] >= 6) & (df['wind_speed'] <= 10)].copy()

    if len(region2) > 0:
        yaw_bins = pd.cut(region2['yaw_error'].abs(), bins=np.arange(0, 35, 5))
        power_by_yaw = region2.groupby(yaw_bins, observed=True)['power'].mean()

        if len(power_by_yaw) > 0:
            ax2.bar(range(len(power_by_yaw)), power_by_yaw.values,
                    tick_label=[f'{int(i.left)}-{int(i.right)}°' for i in power_by_yaw.index],
                    color='coral', edgecolor='black')
            ax2.set_xlabel('Absolute Yaw Error (degrees)')
            ax2.set_ylabel('Mean Power (kW)')
            ax2.set_title('Power vs Yaw Misalignment (6-10 m/s wind)')

    # Time series of yaw tracking
    ax3 = axes[1, 0]
    sample = df[df['turbine_id'] == 1].head(500)
    if len(sample) > 0:
        ax3.plot(range(len(sample)), sample['wind_direction'], label='Wind Direction', alpha=0.7)
        ax3.plot(range(len(sample)), sample['yaw_angle'], label='Nacelle Position', alpha=0.7)
        ax3.set_xlabel('Time Index (10-min intervals)')
        ax3.set_ylabel('Angle (degrees)')
        ax3.set_title('Yaw Tracking Behavior (Turbine 1 Sample)')
        ax3.legend()

    # Polar plot of wind direction and power
    ax4 = axes[1, 1]
    ax4.remove()
    ax4 = fig.add_subplot(224, projection='polar')

    # Bin by wind direction
    df['wd_bin'] = pd.cut(df['wind_direction'], bins=np.linspace(0, 360, 37))
    wd_power = df.groupby('wd_bin', observed=True)['power'].mean().dropna()

    if len(wd_power) > 0:
        theta = np.deg2rad([x.mid for x in wd_power.index])
        ax4.bar(theta, wd_power.values, width=np.deg2rad(10), alpha=0.7, color='teal')
    ax4.set_title('Power by Wind Direction', pad=20)

    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / 'yaw_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'yaw_analysis.png'}")

    plt.close()
    return fig


def plot_pitch_analysis(df, save=True):
    """Analyze blade pitch behavior."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Pitch vs wind speed
    ax1 = axes[0]
    ax1.scatter(df['wind_speed'], df['pitch_angle'], alpha=0.1, s=5, c='purple')
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Pitch Angle (degrees)')
    ax1.set_title('Pitch Angle vs Wind Speed')
    ax1.axvline(x=12, color='red', linestyle='--', label='Rated wind speed')
    ax1.legend()
    ax1.set_xlim(0, 25)

    # Pitch vs power
    ax2 = axes[1]
    ax2.scatter(df['pitch_angle'], df['power'], alpha=0.1, s=5, c='green')
    ax2.set_xlabel('Pitch Angle (degrees)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Power vs Pitch Angle')

    # Pitch distribution by operating region
    ax3 = axes[2]
    df['region'] = pd.cut(df['wind_speed'],
                          bins=[0, 3, 12, 25, 100],
                          labels=['Below Cut-in', 'Region 2', 'Region 3', 'Above Cut-out'])

    colors = {'Below Cut-in': 'gray', 'Region 2': 'blue', 'Region 3': 'green', 'Above Cut-out': 'red'}
    for region in ['Below Cut-in', 'Region 2', 'Region 3']:
        data = df[df['region'] == region]['pitch_angle'].dropna()
        if len(data) > 0:
            ax3.hist(data, bins=30, alpha=0.5, label=region, color=colors.get(region, 'gray'))

    ax3.set_xlabel('Pitch Angle (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Pitch Distribution by Operating Region')
    ax3.legend()

    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / 'pitch_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'pitch_analysis.png'}")

    plt.close()
    return fig


def plot_correlation_matrix(df, save=True):
    """Plot correlation matrix of key variables."""

    numeric_cols = ['wind_speed', 'wind_direction', 'power',
                    'yaw_angle', 'pitch_angle', 'rotor_speed']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, ax=ax)

    ax.set_title('Correlation Matrix - Kelmarsh SCADA Variables')

    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'correlation_matrix.png'}")

    plt.close()
    return fig


def plot_wake_effects(df, save=True):
    """Visualize wake effects between turbines."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Power comparison by turbine
    ax1 = axes[0]
    turbine_power = df.groupby('turbine_id')['power'].mean().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(turbine_power)))

    bars = ax1.bar(turbine_power.index, turbine_power.values, color=colors, edgecolor='black')
    ax1.set_xlabel('Turbine ID')
    ax1.set_ylabel('Mean Power (kW)')
    ax1.set_title('Average Power by Turbine')

    # Add percentage labels
    max_power = turbine_power.max()
    for bar, (tid, power) in zip(bars, turbine_power.items()):
        pct = power / max_power * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{pct:.1f}%', ha='center', fontsize=10)

    # Power curves by turbine
    ax2 = axes[1]
    for tid in sorted(df['turbine_id'].unique()):
        mask = df['turbine_id'] == tid
        ws_bins = pd.cut(df.loc[mask, 'wind_speed'], bins=np.arange(0, 26, 1))
        binned = df.loc[mask].groupby(ws_bins, observed=True)['power'].mean()
        ws_mid = [x.mid for x in binned.index]
        ax2.plot(ws_mid, binned.values, 'o-', label=f'Turbine {tid}', markersize=4, alpha=0.7)

    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Power Curves by Turbine')
    ax2.legend()
    ax2.set_xlim(0, 25)

    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / 'wake_effects.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'wake_effects.png'}")

    plt.close()
    return fig


def prepare_pinn_dataset(df, output_path=None):
    """
    Prepare data for PINN training.
    """
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR PINN TRAINING")
    print("=" * 60)

    # Select relevant features
    input_features = ['wind_speed', 'wind_direction', 'yaw_angle', 'pitch_angle']
    output_features = ['power']

    # Filter valid data
    df_clean = df.dropna(subset=input_features + output_features).copy()
    print(f"Clean samples: {len(df_clean)}")

    # Normalize to [0, 1]
    normalization_params = {}

    for col in input_features + output_features:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        df_clean[f'{col}_norm'] = (df_clean[col] - min_val) / (max_val - min_val + 1e-8)
        normalization_params[col] = {'min': float(min_val), 'max': float(max_val)}

    # Create arrays
    X = df_clean[[f'{c}_norm' for c in input_features]].values
    y = df_clean[[f'{c}_norm' for c in output_features]].values

    # Train/validation/test split
    n = len(X)
    idx = np.random.permutation(n)

    train_idx = idx[:int(0.7*n)]
    val_idx = idx[int(0.7*n):int(0.85*n)]
    test_idx = idx[int(0.85*n):]

    data_splits = {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'input_features': input_features,
        'output_features': output_features
    }

    print(f"\nData splits:")
    print(f"  Training:   {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples")
    print(f"  Test:       {len(test_idx)} samples")

    if output_path:
        np.savez(output_path, **data_splits)
        # Save normalization params separately
        import json
        with open(str(output_path).replace('.npz', '_norm_params.json'), 'w') as f:
            json.dump(normalization_params, f, indent=2)
        print(f"\n✓ Saved to {output_path}")

    return data_splits, normalization_params


def main():
    """Main exploration pipeline."""

    print("\n" + "=" * 60)
    print("KELMARSH WIND FARM - DATA EXPLORATION")
    print("Progress Report 2")
    print("=" * 60)

    # Load data
    df = load_kelmarsh_data(DATA_DIR)

    # Clean data
    df = clean_data(df)

    # Save cleaned data
    df.to_csv(OUTPUT_DIR.parent / 'kelmarsh_cleaned.csv', index=False)
    print(f"✓ Saved cleaned data to {OUTPUT_DIR.parent / 'kelmarsh_cleaned.csv'}")

    # Statistics
    stats = explore_data_statistics(df)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_power_curve(df)
    plot_yaw_analysis(df)
    plot_pitch_analysis(df)
    plot_correlation_matrix(df)
    plot_wake_effects(df)

    # Prepare PINN data
    prepare_pinn_dataset(df, OUTPUT_DIR.parent / 'pinn_training_data.npz')

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nFigures saved to: {OUTPUT_DIR}")
    print("\nNext: Run PINN training with pinnstorch")


if __name__ == "__main__":
    main()