import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your summary CSV
df = pd.read_csv("cell_counts_summary.csv")

# Extract numeric ratio from run name
df['ratio'] = df['run'].str.extract(r'ratio_(\d+)_')[0].astype(int)

# Map ratio numbers to custom labels
ratio_labels = {
    1: '99/1',
    2: '97/3',
    3: '95/5',
    4: '93/7',
    5: '90/10'
}
df['ratio_label'] = df['ratio'].map(ratio_labels)

# Melt to long format
long_df = pd.melt(df,
                  id_vars=['condition', 'run', 'ratio', 'ratio_label'],
                  value_vars=['regressive', 'blastemal'],
                  var_name='agent_type',
                  value_name='count')

# Loop over conditions and agent types separately
for cond in ['dox', 'no_dox']:
    for agent in ['regressive', 'blastemal']:
        plt.figure(figsize=(8, 6))
        plot_data = long_df[
            (long_df['condition'] == cond) &
            (long_df['agent_type'] == agent)
        ]

        sns.violinplot(data=plot_data,
                       x='ratio_label',
                       y='count',
                       inner='quartile',
                       scale='width')

        plt.title(f'{agent.capitalize()} Cell Count by Initial Ratio ({cond.upper()})')
        plt.xlabel('Initial Cell Ratio (Regressive/Blastemal)')
        plt.ylabel('Cell Count')
        plt.tight_layout()
        plt.show()
