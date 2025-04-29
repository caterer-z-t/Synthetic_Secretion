import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === USER SETTINGS ===
# Replace these with your actual file paths
channel1_csv = '/Users/griffinrashoff/Team Rotation Data/CaWaveForm_PeakValues_Only_Ch1.csv'
channel2_csv = '/Users/griffinrashoff/Team Rotation Data/delta/CaWaveForm_PeakValues_Only_Ch1.csv'
output_dir = '/Users/griffinrashoff/Team Rotation Data'
output_csv = os.path.join(output_dir, 'Merged_CorrelationMatrix.csv')
output_heatmap = os.path.join(output_dir, 'Merged_CorrelationMatrix_Heatmap.png')

# === LOAD DATA ===
df_ch1 = pd.read_csv(channel1_csv)
df_ch2 = pd.read_csv(channel2_csv)

# Remove any unnamed index columns if present
df_ch1 = df_ch1.loc[:, ~df_ch1.columns.str.contains('^Unnamed')]
df_ch2 = df_ch2.loc[:, ~df_ch2.columns.str.contains('^Unnamed')]

# === MERGE CHANNELS ===
# Assume data is time x cell and we want to merge cells from both channels
merged_df = pd.concat([df_ch1, df_ch2], axis=1)  # Horizontal merge: [timepoints x all_cells]

# === COMPUTE CORRELATION MATRIX ===
correlation_matrix = merged_df.corr(method='pearson')  # Correlate across timepoints

# === SAVE CORRELATION MATRIX ===
correlation_matrix.to_csv(output_csv)
print(f"Correlation matrix saved to: {output_csv}")

# === PLOT HEATMAP ===
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.8}, xticklabels=True, yticklabels=True)
plt.title('Correlation Matrix Heatmap (Merged Channels)')
plt.tight_layout()
plt.savefig(output_heatmap, dpi=300)
plt.show()

print(f"Heatmap saved to: {output_heatmap}")
