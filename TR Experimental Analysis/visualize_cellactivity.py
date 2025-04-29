import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load CSV ===
file_path = '/Users/griffinrashoff/Team Rotation Data/Analysis.csv'  # Update path as needed
df = pd.read_csv(file_path)

# === Ensure column names ===
# Expected columns: ['Cell', 'Condition', 'CellType', 'NumOscillations']
# If not already, ensure 'Condition' and 'CellType' are categorical
df['Condition'] = df['Condition'].astype(str)
df['CellType'] = df['Cell'].astype(str)

# === Set plot style and color palette ===
sns.set(style="whitegrid")
palette = {'wt': 'royalblue', 'KO': 'darkorange'}

# === Create the bar plot with individual dots ===
plt.figure(figsize=(8, 6))

# Bar plot: show mean ± standard deviation
sns.barplot(
    data=df,
    x='CellType',
    y='NumOscillations',
    hue='Condition',
    ci='sd',
    palette=palette,
    capsize=0.1,
    errwidth=1.5
)

# Overlay stripplot: show individual cell points
sns.stripplot(
    data=df,
    x='CellType',
    y='NumOscillations',
    hue='Condition',
    palette=palette,
    dodge=True,
    marker='o',
    size=6,
    alpha=0.7,
    edgecolor='gray',
    linewidth=0.5
)

# === Fix the legend (remove duplicates) ===
handles, labels = plt.gca().get_legend_handles_labels()
n = len(df['Condition'].unique())
plt.legend(handles[:n], labels[:n], title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

# === Labeling ===
plt.title("Number of Oscillations by Cell Type and Condition")
plt.xlabel("")
plt.ylabel("Num Oscillations")
plt.tight_layout()

# === Show or save ===
plt.show()
# plt.savefig("oscillation_barplot.png", dpi=300)

# === Load the Analysis Data ===
file_path = '/Users/griffinrashoff/Team Rotation Data/Analysis.csv'  # Update if needed
df = pd.read_csv(file_path)

# === Prepare the Data ===
# Melt the dataframe to have one column for amplitude values and one for type
df_melted = pd.melt(df, 
                    id_vars=['Cell', 'Condition'], 
                    value_vars=['FoldChange_AvgAmplitude', 'FoldChange_PeakAmplitude'],
                    var_name='AmplitudeType',
                    value_name='Amplitude')

# Clean up names
df_melted['AmplitudeType'] = df_melted['AmplitudeType'].replace({
    'FoldChange_AvgAmplitude': 'AvgAmplitude',
    'FoldChange_PeakAmplitude': 'PeakAmplitude'
})

# === Plot ===
plt.figure(figsize=(10, 6))

# Define consistent color palette
palette = {'wt': 'blue', 'KO': 'orange'}

# Barplot (means ± SE)
sns.barplot(
    data=df_melted,
    x='Cell',
    y='Amplitude',
    hue='Condition',
    capsize=0.1,
    palette=palette,
    dodge=True
)


# Overlay individual points
sns.stripplot(
    data=df_melted,
    x='Cell',
    y='Amplitude',
    hue='Condition',
    dodge=True,
    alpha=0.7,
    size=5,
    marker='o',
    edgecolor='gray',
    linewidth=0.5,
    palette=palette
)

# === Tidy up ===
# Remove duplicated legends
handles, labels = plt.gca().get_legend_handles_labels()
n = len(palette)
plt.legend(handles[:n], labels[:n], title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Fold Change Amplitude (Avg and Peak)')
plt.xlabel('Cell Type (Beta / Delta)')
plt.ylabel('Fold Change Amplitude')
plt.tight_layout()
plt.show()
