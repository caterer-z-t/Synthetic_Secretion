import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.fft import fft

channel1_csv = '/Users/griffinrashoff/Team Rotation Data/2_15_islet2KO/BetaCaWaveForm_PeakValues_Only_Ch1.csv'
channel2_csv = '/Users/griffinrashoff/Team Rotation Data/2_15_islet2KO/deltas/CaWaveForm_PeakValues_Only_Ch1.csv'
output_dir = '/Users/griffinrashoff/Team Rotation Data/2_15_islet2KO'
output_csv = os.path.join(output_dir, '2_15_I2KOMerged_CorrelationMatrix.csv')

# === Step 2: Load and merge the CSVs ===
channel1_df = pd.read_csv(channel1_csv)
channel2_df = pd.read_csv(channel2_csv)

# Add labels to columns to identify cell type
channel1_df.columns = [f"Beta_{col}" for col in channel1_df.columns]
channel2_df.columns = [f"Delta_{col}" for col in channel2_df.columns]

# Combine them horizontally (columns = cells)
df = pd.concat([channel1_df, channel2_df], axis=1)

# Save combined timecourses (optional)
df.to_csv(output_csv, index=False)
print(f"Merged file saved at {output_csv}")

# == Duty Cycle ==
# Initialize an empty list to store the duty cycle for each cell
duty_cycle = []

# Loop over each cell column and calculate baseline and duty cycle
for cell in df.columns:
    current = df[cell].dropna().to_numpy()  # Remove any NaNs (time points not collected for that cell)

    # Skip if the cell has no valid data points
    if current.size == 0:
        print(f"Cell {cell}: No valid data points. Skipping duty cycle calculation.")
        duty_cycle.append(0.0)
        continue

    # Step 1: Set baseline as the average value over the trace (non-detrended)
    baseline = np.nanmean(current)

    # Step 2: Check if there is a spike that is at least 1.5x the baseline
    spike_check_threshold = 1.3 * baseline
    if np.max(current) < spike_check_threshold:
        print(f"Cell {cell}: No spikes exceed 2x the baseline. Skipping duty cycle calculation.")
        duty_cycle.append(0.0)  # No duty cycle calculated for this cell
        continue  # Skip further calculations for this cell

    # Step 3: Subtract baseline from the current trace to normalize it
    norm_current = current - baseline

    # Step 4: Smooth the normalized data
    smooth_current = np.convolve(norm_current, np.ones((5,))/5, mode='same')

    # Step 5: Set a dynamic threshold for 'on' state (for example, 50% above the baseline-adjusted mean)
    dynamic_threshold = 0.5 * np.mean(smooth_current)

    # Step 6: Calculate duty cycle by checking the percentage of time the cell is "on"
    on_state = smooth_current > dynamic_threshold
    duty_cycle.append(np.sum(on_state) / len(smooth_current))

    # Optional plotting for visual verification (can comment out if not needed)
    plt.figure()
    plt.plot(current, label='Original data')
    plt.plot(smooth_current, label='Smoothed data')
    plt.axhline(y=dynamic_threshold, linestyle='--', color='r', label='Threshold')
    plt.title(f'Cell {cell}')
    plt.legend()
    plt.show()

# Output the duty cycle for each cell
print("Duty cycle for each cell:")
for cell_idx, dc in enumerate(duty_cycle):
    print(f"Cell {cell_idx + 1}: {dc:.4f}")

# == Amplitude ==

# Convert DataFrame to NumPy array
CellTC = df.to_numpy()

# Initialize arrays for normalized data, average amplitude, and peak amplitude
normalized_ca = np.full_like(CellTC, np.nan)  # Use NaN for missing values
fold_change_ca = np.full_like(CellTC, np.nan)

average_amplitude = np.zeros(CellTC.shape[1])
peak_amplitude = np.zeros(CellTC.shape[1])

# Arrays to store fold-change for peak and average amplitudes
fold_change_peak = np.zeros(CellTC.shape[1])
fold_change_avg = np.zeros(CellTC.shape[1])

# Normalize each cell's data and calculate the average and peak amplitude
for cell in range(CellTC.shape[1]):  # Loop through each cell
    current = CellTC[:, cell]
    
    # Exclude NaN values in the current cell trace
    current = current[~np.isnan(current)]
    
    if current.size > 0:  # Check if there is valid data in the current cell trace
        # Step 1: Find the minimum value of the current cell trace
        min_val = np.nanmin(current)
        
        # Step 2: Subtract the minimum value from all points (baseline set to 0)
        norm_current = current - min_val
        
        # Step 3: Calculate fold-change relative to the baseline (min_val)
        fold_change_current = current / min_val if min_val != 0 else np.full_like(current, np.nan)  # Avoid division by 0
        
        # Store normalized data (only for available time points)
        normalized_ca[:len(norm_current), cell] = norm_current
        fold_change_ca[:len(fold_change_current), cell] = fold_change_current
        
        # Step 4: Calculate the average and peak amplitude for each cell (raw values)
        average_amplitude[cell] = np.nanmean(norm_current)  # Raw normalized average
        peak_amplitude[cell] = np.nanmax(norm_current)  # Raw normalized peak
        
        # Step 5: Calculate the fold change for average and peak amplitude relative to baseline
        if min_val != 0:
            fold_change_avg[cell] = np.nanmean(current) / min_val  # Fold change of average amplitude
            fold_change_peak[cell] = np.nanmax(current) / min_val  # Fold change of peak amplitude
        else:
            fold_change_avg[cell] = np.nan  # Handle case where baseline is 0
            fold_change_peak[cell] = np.nan

        # Optional: Plot original, normalized, and fold-change data
        plt.figure()
        plt.plot(current, label="Original signal")
        plt.plot(norm_current, label="Normalized data (min set to 0)")
        plt.plot(fold_change_current, label="Fold change relative to baseline")
        plt.axhline(y=0, color='r', linestyle='--', label="Baseline (0)")
        plt.legend(loc="upper right")
        plt.title(f'Cell {cell + 1}')
        plt.show()

# Output the results
print("Average and Peak Amplitude for each cell (normalized data and fold change):")
for cell in range(CellTC.shape[1]):
    print(f"Cell {cell + 1}:")
    print(f"  - Raw Normalized Average Amplitude = {average_amplitude[cell]:.4f}")
    print(f"  - Fold Change Average Amplitude = {fold_change_avg[cell]:.4f}x baseline")
    print(f"  - Raw Normalized Peak Amplitude = {peak_amplitude[cell]:.4f}")
    print(f"  - Fold Change Peak Amplitude = {fold_change_peak[cell]:.4f}x baseline")

# === Frequency of Oscillations ===
from scipy.signal import find_peaks

# === Frequency (Number of Peaks) ===
num_oscillations = []

for cell in range(CellTC.shape[1]):
    signal = CellTC[:, cell]
    signal = signal[~np.isnan(signal)]  # Remove NaNs

    if len(signal) == 0:
        num_oscillations.append(np.nan)
        continue

    # Find peaks using prominence threshold (adjust as needed)
    peaks, _ = find_peaks(signal, prominence=np.std(signal) * 0.5)
    num_oscillations.append(len(peaks))

    # Optional plot for verification
    plt.figure()
    plt.plot(signal, label='Signal')
    plt.plot(peaks, signal[peaks], "x", label='Peaks')
    plt.title(f"Cell {cell + 1} Peaks")
    plt.legend()
    plt.show()


# === Save Results to CSV ===

# Extract cleaned cell names (already in df.columns)
cell_names = df.columns

# Build a DataFrame with the calculated metrics
results_df = pd.DataFrame({
    'Cell': cell_names,
    'DutyCycle': duty_cycle,
    'RawNorm_AvgAmplitude': average_amplitude,
    'FoldChange_AvgAmplitude': fold_change_avg,
    'RawNorm_PeakAmplitude': peak_amplitude,
    'FoldChange_PeakAmplitude': fold_change_peak,
    'NumOscillations': num_oscillations  # Add this!
})

# Save the results
results_csv_path = os.path.join(output_dir, '2CalciumActivityMetrics.csv')
results_df.to_csv(results_csv_path, index=False)

print(f"Activity metrics saved to: {results_csv_path}")
