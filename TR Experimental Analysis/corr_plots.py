import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Load Excel with multi-level headers ===
file_path = '/Users/griffinrashoff/Team Rotation Data/Correlations.xlsx'

# Read with two header rows
df = pd.read_excel(file_path, header=[0, 1])

# Flatten multi-level column headers
df.columns = ['_'.join(col).strip().replace(' ', '') for col in df.columns]

# === Step 2: Reshape to long format ===
df_long = df.melt(var_name='Group', value_name='Correlation')
df_long[['Pair', 'Condition']] = df_long['Group'].str.split('_', expand=True)
df_long.drop(columns='Group', inplace=True)

# === Set desired group order (wt first, then ko)
df_long['Condition'] = pd.Categorical(df_long['Condition'], categories=['wt', 'ko'], ordered=True)

# === Step 3: Compute Summary Statistics ===
group_stats = df_long.groupby(['Pair', 'Condition'])['Correlation'].agg(['mean', 'sem']).reset_index()
group_stats['Condition'] = pd.Categorical(group_stats['Condition'], categories=['wt', 'ko'], ordered=True)


# === Step 4: Bar Plot with Overlaid Dots ===
plt.figure(figsize=(8, 5))

# Bar plot (means with SE)
sns.barplot(
    data=group_stats,
    x='Pair',
    y='mean',
    hue='Condition',
    capsize=0.1,
    errwidth=1.2,
    palette='pastel',
    dodge=True
)

# Overlay individual points
sns.stripplot(
    data=df_long,
    x='Pair',
    y='Correlation',
    hue='Condition',
    dodge=True,
    alpha=0.7,
    size=5,
    marker='o',
    linewidth=0.5,
    palette='dark',
    edgecolor='gray'
)

# Fix legend (remove duplicates)
handles, labels = plt.gca().get_legend_handles_labels()
n = len(set(df_long['Condition']))
plt.legend(handles[:n], labels[:n], title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.ylabel("Correlation Coefficient (r)")
plt.title("Bar Plot of Correlation Coefficients with Individual Points")
plt.xlabel("")
plt.tight_layout()
plt.show()

# === Step 5: 
plt.figure(figsize=(8, 5))
sns.stripplot(data=df_long, x='Pair', y='Correlation', hue='Condition', dodge=True, jitter=True)
plt.title("Individual Correlation Coefficients")
plt.ylabel("Pearson r")
plt.xlabel("")
plt.legend(title="Condition")
plt.tight_layout()
plt.show()

# === Step 6: Violin Plot ===
plt.figure(figsize=(8, 5))
sns.violinplot(data=df_long, x='Pair', y='Correlation', hue='Condition', split=True, inner='quartile')
plt.title("Violin Plot of Correlation Coefficients")
plt.ylabel("Pearson r")
plt.xlabel("")
plt.tight_layout()
plt.show()

# === Step 7: Histogram Faceted by Pair Type ===
g = sns.FacetGrid(df_long, col="Pair", hue="Condition", sharex=True, sharey=False)
g.map(sns.histplot, "Correlation", bins=10, alpha=0.6, edgecolor="black")
g.add_legend()
g.set_axis_labels("Pearson r", "Count")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Histogram of Correlation Coefficients")
plt.show()
