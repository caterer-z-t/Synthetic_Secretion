import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d

# === Step 1: Define file paths ===
channel1_csv = '/Users/griffinrashoff/Team Rotation Data/1_28_islet1/CaWaveForm_PeakValues_Only_Ch1.csv'
channel2_csv = '/Users/griffinrashoff/Team Rotation Data/1_28_islet1/delta/CaWaveForm_PeakValues_Only_Ch1.csv'
output_dir = '/Users/griffinrashoff/Team Rotation Data/1_28_islet1'
output_csv = os.path.join(output_dir, '1_28_1_Merged_CorrelationMatrix.csv')

# === Step 2: Load and merge the CSVs ===
channel1_df = pd.read_csv(channel1_csv)
channel2_df = pd.read_csv(channel2_csv)

# Add labels to columns to identify cell type
channel1_df.columns = [f"Beta_{col}" for col in channel1_df.columns]
channel2_df.columns = [f"Delta_{col}" for col in channel2_df.columns]

# Combine them horizontally (columns = cells)
timecourses = pd.concat([channel1_df, channel2_df], axis=1)

# Save combined timecourses (optional)
timecourses.to_csv(output_csv, index=False)
print(f"Merged file saved at {output_csv}")

# === Step 3: Detrending and Smoothing ===
detrended_ca = pd.DataFrame(index=timecourses.index)
smoothed_ca = pd.DataFrame(index=timecourses.index)

for cell in timecourses.columns:
    current = timecourses[cell]

    # Detrend
    norm_current = pd.Series(detrend(current, type='linear'), index=current.index)
    trend = current - norm_current
    avg_detrended = norm_current.mean()

    # Store results
    detrended_ca[cell] = norm_current
    smoothed_ca[cell] = uniform_filter1d(norm_current, size=5)

    # Plot (optional)
    plt.figure(figsize=(10, 4))
    plt.plot(current, label='Original')
    plt.plot(norm_current, label='Detrended')
    plt.plot(trend, label='Trend')
    plt.axhline(avg_detrended, color='gray', linestyle='--', label='Mean of Detrended')
    plt.title(f'Cell {cell}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# === Step 4: Save outputs (optional) ===
# detrended_ca.to_csv(os.path.join(output_dir, "Detrended_Traces.csv"), index=False)
# smoothed_ca.to_csv(os.path.join(output_dir, "Smoothed_Traces.csv"), index=False)

# === Step 5: Calculate Duty Cycle ===
# You must have `detrended_ca` and `smoothed_ca` already defined at this point

duty_cycles = []

for cell in detrended_ca.columns:
    current = detrended_ca[cell]
    smoothed = smoothed_ca[cell]

    # Use mean as threshold (customizable if needed)
    thresh = current.mean()

    # Visualize (optional)
    plt.figure(figsize=(10, 4))
    plt.plot(current, label='Detrended')
    plt.plot(smoothed, label='Smoothed')
    plt.axhline(thresh, linestyle='--', color='gray', label='Threshold')
    plt.title(f'Duty Cycle - Cell {cell}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Determine where signal is "on"
    is_on = smoothed > thresh
    percent_on = is_on.sum() / len(smoothed)
    duty_cycles.append(percent_on)

# Convert to DataFrame and save
duty_df = pd.DataFrame({
    'Cell': detrended_ca.columns,
    'DutyCycle': duty_cycles
})

# Save to CSV or just display
duty_df.to_csv(os.path.join(output_dir, 'DutyCycle_PerCell.csv'), index=False)
print("Duty cycle saved to:", os.path.join(output_dir, 'DutyCycle_PerCell.csv'))
