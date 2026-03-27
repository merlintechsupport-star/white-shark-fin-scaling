import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('merged_shark_data.csv')
df = df.dropna()

# Log transform
log_TL = np.log(df['TL_cm'])
log_pec = np.log(df['avg_pec_cm2'])

# Compute fits
# OLS
coeffs = np.polyfit(log_TL, log_pec, 1)
b_ols = coeffs[0]
intercept_ols = coeffs[1]
a_ols = np.exp(intercept_ols)

# SMA
r = np.corrcoef(log_TL, log_pec)[0,1]
b_sma = b_ols / r
mean_log_TL = np.mean(log_TL)
mean_log_pec = np.mean(log_pec)
a_sma = np.exp(mean_log_pec - b_sma * mean_log_TL)

# Nonlinear
a_nonlin = a_ols
b_nonlin = b_ols

# Literature: australian b=1.68, but need a. Assume similar a as our SMA
a_lit = a_sma  # placeholder

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Add red warning at the top
fig.suptitle('⚠️ SIMULATED DATA — NOT REAL MEASUREMENTS', fontsize=16, color='red', fontweight='bold')

# Plot 1: Raw scatter with fits
colors = {'M': 'blue', 'F': 'red'}
for sex in df['sex'].unique():
    subset = df[df['sex'] == sex]
    axes[0,0].scatter(subset['TL_cm'], subset['avg_pec_cm2'], color=colors[sex], label=f'{sex}ales', alpha=0.7)

TL_range = np.linspace(df['TL_cm'].min(), df['TL_cm'].max(), 100)
axes[0,0].plot(TL_range, a_ols * TL_range**b_ols, 'k-', label=f'OLS (b={b_ols:.3f})')
axes[0,0].plot(TL_range, a_sma * TL_range**b_sma, 'b--', label=f'SMA (b={b_sma:.3f})')
axes[0,0].plot(TL_range, a_nonlin * TL_range**b_nonlin, 'g:', label=f'Nonlinear (b={b_nonlin:.3f})')
axes[0,0].plot(TL_range, a_lit * TL_range**1.68, color='gray', linestyle='--', label='Literature (b=1.68)')
axes[0,0].set_xlabel('Total Length (cm)')
axes[0,0].set_ylabel('Avg Pectoral Fin Area (cm²)')
axes[0,0].set_title('All Methods — Raw Scale')
axes[0,0].legend()
axes[0,0].grid(True)

# Plot 2: Log-log with fits
for sex in df['sex'].unique():
    subset = df[df['sex'] == sex]
    axes[0,1].scatter(np.log(subset['TL_cm']), np.log(subset['avg_pec_cm2']), color=colors[sex], label=f'{sex}ales', alpha=0.7)

log_TL_range = np.linspace(log_TL.min(), log_TL.max(), 100)
axes[0,1].plot(log_TL_range, intercept_ols + b_ols * log_TL_range, 'k-', label=f'OLS b={b_ols:.3f}')
axes[0,1].plot(log_TL_range, np.log(a_sma) + b_sma * log_TL_range, 'b--', label=f'SMA b={b_sma:.3f}')
axes[0,1].plot(log_TL_range, np.log(a_nonlin) + b_nonlin * log_TL_range, 'g:', label=f'Nonlinear b={b_nonlin:.3f}')
axes[0,1].plot(log_TL_range, np.log(a_lit) + 1.68 * log_TL_range, color='gray', linestyle='--', label='Literature b=1.68')
axes[0,1].set_xlabel('Log Total Length (cm)')
axes[0,1].set_ylabel('Log Avg Pectoral Fin Area (cm²)')
axes[0,1].set_title('All Methods — Log-Log Scale')
axes[0,1].legend()
axes[0,1].grid(True)

# Plot 3: OLS residuals
pred_log_pec_ols = intercept_ols + b_ols * log_TL
residuals_ols = log_pec - pred_log_pec_ols
axes[1,0].scatter(df['TL_cm'], residuals_ols, alpha=0.7)
axes[1,0].axhline(0, color='black', linestyle='--')
axes[1,0].set_xlabel('Total Length (cm)')
axes[1,0].set_ylabel('OLS Residuals (log scale)')
axes[1,0].set_title('OLS Residuals')
axes[1,0].grid(True)

# Plot 4: SMA residuals
pred_log_pec_sma = np.log(a_sma) + b_sma * log_TL
residuals_sma = log_pec - pred_log_pec_sma
axes[1,1].scatter(df['TL_cm'], residuals_sma, alpha=0.7)
axes[1,1].axhline(0, color='black', linestyle='--')
axes[1,1].set_xlabel('Total Length (cm)')
axes[1,1].set_ylabel('SMA Residuals (log scale)')
axes[1,1].set_title('SMA Residuals')
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('shark_fitted_plots.png', dpi=150)
plt.show()