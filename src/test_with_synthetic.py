"""
test_with_synthetic.py — Test the full pipeline using synthetic data.

Run this to verify everything works before using real Kelmarsh data.
This generates fake SCADA CSVs that mimic the Kelmarsh format.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_kelmarsh(data_dir=None, n_records=5000, n_turbines=6):
    """Generate synthetic SCADA data mimicking Kelmarsh format."""
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    np.random.seed(42)
    
    for t in range(1, n_turbines + 1):
        n = n_records
        
        # Wind speed: Weibull distribution (shape=2, scale=7)
        wind_speed = np.random.weibull(2, n) * 7
        wind_speed = np.clip(wind_speed, 0.5, 25)
        
        # Wind direction: uniform with some clustering
        wind_direction = (np.random.normal(200, 80, n)) % 360
        
        # Yaw: follows wind direction with some lag/error
        yaw = wind_direction + np.random.normal(0, 8, n)
        yaw = yaw % 360
        
        # Pitch: near 0 for Region 2, increases for Region 3
        pitch = np.where(wind_speed < 12, 
                         np.random.normal(0.5, 1, n),
                         (wind_speed - 12) * 3 + np.random.normal(0, 1, n))
        pitch = np.clip(pitch, -2, 90)
        
        # Power: cubic law + yaw cosine loss + noise
        yaw_misalign = np.radians((wind_direction - yaw + 180) % 360 - 180)
        cp = 0.4 * np.cos(yaw_misalign) ** 3
        area = np.pi * (92/2)**2
        power = 0.5 * 1.225 * area * cp * wind_speed**3 / 1000  # kW
        power = np.clip(power, -10, 2050) + np.random.normal(0, 20, n)
        power = np.clip(power, -20, 2080)
        
        # Wake effect for turbine 6 (reduced power)
        if t == 6:
            power *= 0.85
        
        # Rotor speed
        rotor_speed = np.where(wind_speed < 3, 0,
                               np.clip(wind_speed * 1.2 + np.random.normal(0, 0.5, n), 0, 15.5))
        
        # Timestamps
        timestamps = pd.date_range('2021-01-01', periods=n, freq='10min')
        
        df = pd.DataFrame({
            '# Date and time': timestamps.strftime('%Y-%m-%d %H:%M:%S'),
            'Wind speed (m/s)': wind_speed,
            'Wind direction (°)': wind_direction,
            'Power (kW)': power,
            'Nacelle position (°)': yaw,
            'Pitch angle (°)': pitch,
            'Rotor speed (rpm)': rotor_speed,
        })
        
        filepath = os.path.join(data_dir, f"Turbine{t}.csv")
        df.to_csv(filepath, index=False)
    
    print(f"Generated {n_turbines} synthetic turbine files in {data_dir}/")
    print(f"  {n_records} records per turbine")


if __name__ == "__main__":
    print("Generating synthetic Kelmarsh data for testing...\n")
    generate_synthetic_kelmarsh()
    
    print("\nRunning full pipeline on synthetic data...\n")
    from train_pinn import main
    main()
