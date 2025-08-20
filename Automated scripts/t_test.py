import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import scikit_posthocs as sp

# === LOAD AND CLEAN DATA ===
df = pd.read_csv('cell_counts_summary.csv')

# Ensure consistent formatting & no whitespace
df['condition'] = df['condition'].str.lower().str.strip()

# ratio from run name
df['ratio'] = df['run'].str.extract(r'ratio_(\d+)_')[0].astype(int)

# Calculate derived metrics
df['blastemal_to_regressive'] = df['blastemal'] / df['regressive']
df['total_live'] = df['blastemal'] + df['regressive']
df['percent_dead'] = df['dead'] / df['total_live'] * 100

# Ratio label mapping
ratio_labels = {
    1: '99/1',
    2: '97/3',
    3: '95/5',
    4: '93/7',
    5: '90/10'
}
df['ratio_label'] = df['ratio'].map(ratio_labels)

order = ['99/1', '97/3', '95/5', '93/7', '90/10']

df['blast_to_reg'] = df['blastemal'] / df['regressive']
df['reg_to_blast'] = df['regressive'] / df['blastemal']

# === STATS ANALYSIS ===

def violin_plot_ratio(metric, ylabel):
    for cond in ['dox', 'no_dox']:
        subset = df[df['condition'] == cond]
        if subset.empty or metric not in subset.columns:
            print(f"⚠️ Skipping {cond.upper()} – no data for {metric}")
            continue

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=subset, x='ratio_label', y=metric, inner='box', cut=0)
        plt.title(f"{ylabel} by Initial Ratio ({cond.upper()})")
        plt.xlabel("Initial Ratio (Regressive/Blastemal)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

# T-test between treatment groups (blastemal count)
blast_dox = df[df['condition'] == 'dox']['blastemal']
blast_no_dox = df[df['condition'] == 'no_dox']['blastemal']
stat, pval = stats.ttest_ind(blast_dox, blast_no_dox)
print(f"T-test p-value for blastemal counts: {pval:.4f}")
print("→ Significant difference!" if pval < 0.05 else "→ Not significant.")

# Kruskal-Wallis for each metric
metrics = {
    'blastemal': 'Blastemal Count',
    'blastemal_to_regressive': 'Blastemal : Regressive',
    'percent_dead': '% Dead'
}

for metric, label in metrics.items():
    for cond in ['dox', 'no_dox']:
        subset = df[df['condition'] == cond]
        groups = [g[metric].dropna().values for _, g in subset.groupby('ratio')]
        if all(len(g) > 0 for g in groups):
            stat, pval = stats.kruskal(*groups)
            print(f"{cond.upper()} Kruskal-Wallis p-value ({label}): {pval:.4f}")

import numpy as np

print(df['ratio'].unique())
print(df['ratio_label'].unique())

print(df[['run', 'ratio', 'ratio_label']].drop_duplicates())

# Label mapping and order
label_map = {
    1: '99/1',
    2: '97/3',
    3: '95/5',
    4: '93/7',
    5: '90/10'
}
df['ratio'] = df['ratio'].astype(int)
df['ratio_label'] = df['ratio'].map(label_map)

def dunn_test_with_heatmap(subset, condition_name, value_col='blastemal'):
    from matplotlib.colors import Normalize

    # Label mapping and order
    label_map = {
        1: '99/1',
        2: '97/3',
        3: '95/5',
        4: '93/7',
        5: '90/10'
    }
    subset = subset.copy()
    subset['ratio_label'] = subset['ratio'].map(label_map)

    # Ensure ratio_label is categorical and ordered
    subset['ratio_label'] = pd.Categorical(subset['ratio_label'], categories=list(label_map.values()), ordered=True)

    # Run Dunn's test
    p_matrix = sp.posthoc_dunn(subset, val_col=value_col, group_col='ratio_label', p_adjust='bonferroni')

    # Drop duplicate or invalid index entries
    p_matrix = p_matrix.loc[label_map.values(), label_map.values()]  # reorders and ensures match
    p_matrix = p_matrix[~p_matrix.index.duplicated(keep='first')]
    p_matrix = p_matrix.loc[:, ~p_matrix.columns.duplicated(keep='first')]

    # Print result
    print(f"\n🔬 DUNN'S TEST — {condition_name.upper()} ({value_col}):")
    print(p_matrix.round(4))

    # Plot heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        p_matrix,
        annot=True,
        fmt=".4f",
        cmap='coolwarm_r',
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 9},
        cbar_kws={"label": "p-value"}
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f"Dunn’s Test Heatmap ({condition_name.upper()} – {value_col})")
    plt.tight_layout()
    plt.show()

dunn_test_with_heatmap(df[df['condition'] == 'dox'], 'dox', value_col='blastemal')
dunn_test_with_heatmap(df[df['condition'] == 'no_dox'], 'no_dox', value_col='blastemal')

# === PLOTTING ===

violin_plot_ratio('blast_to_reg', 'Blastemal : Regressive Ratio')
violin_plot_ratio('reg_to_blast', 'Regressive : Blastemal Ratio')

def plot_metric(metric, ylabel):
    for cond in ['dox', 'no_dox']:
        subset = df[df['condition'] == cond]
        if subset.empty:
            print(f"⚠️ Skipping {cond} — no data")
            continue
        if metric not in subset.columns:
            print(f"⚠️ '{metric}' not found in columns")
            continue
        if subset[metric].isna().all():
            print(f"⚠️ All values are NaN for '{metric}' in {cond}")
            continue

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=subset, x='ratio_label', y=metric)
        plt.title(f"{ylabel} by Initial Ratio ({cond.upper()})")
        plt.xlabel("Initial Ratio (Regressive/Blastemal)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

# Plot each metric
plot_metric('blastemal', 'Blastemal Cell Count')
plot_metric('blastemal_to_regressive', 'Blastemal : Regressive Ratio')
plot_metric('percent_dead', '% Dead Cells')

