import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('data/merged_shark_data.csv')

# Drop missing values
df = df.dropna()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Add red warning at the top
fig.suptitle('⚠️ SIMULATED DATA — NOT REAL MEASUREMENTS', fontsize=16, color='red', fontweight='bold')

# Plot 1: TL_cm vs avg_pec_cm2, colored by sex
colors = {'M': 'blue', 'F': 'red'}
for sex in df['sex'].unique():
    subset = df[df['sex'] == sex]
    axes[0].scatter(subset['TL_cm'], subset['avg_pec_cm2'], color=colors[sex], label=f'{sex}ales', alpha=0.7)
axes[0].set_xlabel('Total Length (cm)')
axes[0].set_ylabel('Avg Pectoral Fin Area (cm²)')
axes[0].set_title('Total Length vs Avg Pectoral Fin Area')
axes[0].legend()
axes[0].grid(True)

# Plot 2: left_pec_cm2 vs right_pec_cm2
axes[1].scatter(df['left_pec_cm2'], df['right_pec_cm2'], alpha=0.7)
# 1:1 line
min_val = min(df['left_pec_cm2'].min(), df['right_pec_cm2'].min())
max_val = max(df['left_pec_cm2'].max(), df['right_pec_cm2'].max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
axes[1].set_xlabel('Left Pectoral Fin Area (cm²)')
axes[1].set_ylabel('Right Pectoral Fin Area (cm²)')
axes[1].set_title('Left vs Right Fin Symmetry')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Log TL vs Log avg_pec_cm2
log_TL = np.log(df['TL_cm'])
log_pec = np.log(df['avg_pec_cm2'])
for sex in df['sex'].unique():
    subset = df[df['sex'] == sex]
    axes[2].scatter(np.log(subset['TL_cm']), np.log(subset['avg_pec_cm2']), color=colors[sex], label=f'{sex}ales', alpha=0.7)
axes[2].set_xlabel('Log Total Length (cm)')
axes[2].set_ylabel('Log Avg Pectoral Fin Area (cm²)')
axes[2].set_title('Log TL vs Log Pectoral Fin Area')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('outputs/shark_raw_plots.png', dpi=150)
plt.show()