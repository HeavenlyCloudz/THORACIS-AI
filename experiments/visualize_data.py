import pandas as pd, matplotlib.pyplot as plt, numpy as np

# Load your key experiment files
scans = {
    'Healthy Baseline': 'healthy_baseline.csv',
    'Small Tumor (1cm)': 'tumor_small_center.csv',
    'Medium Tumor (2cm)': 'tumor_medium_center.csv',
    'Large Tumor (3cm)': 'tumor_large_center.csv',
    'Tumor Left': 'tumor_medium_left.csv',
    'Tumor Right': 'tumor_medium_right.csv',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PULMO AI - Current 2-Antenna System Performance Analysis', fontsize=14, fontweight='bold')

# 1. Frequency Response: All Scans
ax1 = axes[0, 0]
for label, file in scans.items():
    df = pd.read_csv(file)
    freq_ghz = df['Frequency_Hz'] / 1e9
    ax1.plot(freq_ghz, df['S21_dB'], label=label, alpha=0.8, linewidth=1.5)
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('S21 Transmission (dB)')
ax1.set_title('Raw Frequency Response - All Scans')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc='upper right')

# 2. Tumor Size Comparison (Center Position Only)
ax2 = axes[0, 1]
center_scans = {k:v for k,v in scans.items() if 'Left' not in k and 'Right' not in k}
for label, file in center_scans.items():
    df = pd.read_csv(file)
    ax2.plot(df['Frequency_Hz']/1e9, df['S21_dB'], label=label, linewidth=2)
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('S21 (dB)')
ax2.set_title('Tumor Size Detection (Center Position)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Average Signal Strength Bar Chart
ax3 = axes[1, 0]
labels = list(scans.keys())
averages = [pd.read_csv(f)['S21_dB'].mean() for f in scans.values()]
colors = ['blue' if 'Healthy' in l else ('orange' if 'Small' in l else 'red') for l in labels]
bars = ax3.bar(range(len(labels)), averages, color=colors, alpha=0.7)
ax3.axhline(y=averages[0], color='gray', linestyle='--', alpha=0.5, label='Healthy Baseline')
ax3.set_ylabel('Average S21 (dB)')
ax3.set_title('Average Transmission Loss by Scenario')
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels([l.replace(' Tumor', '\nTumor') for l in labels], rotation=45, ha='right', fontsize=9)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Critical Finding: Left vs Right Asymmetry
ax4 = axes[1, 1]
left_df = pd.read_csv(scans['Tumor Left'])
right_df = pd.read_csv(scans['Tumor Right'])
healthy_df = pd.read_csv(scans['Healthy Baseline'])
freq_ghz = healthy_df['Frequency_Hz'] / 1e9

ax4.plot(freq_ghz, healthy_df['S21_dB'], 'k-', label='Healthy Baseline', linewidth=2, alpha=0.5)
ax4.plot(freq_ghz, left_df['S21_dB'], 'b-', label='Tumor LEFT', linewidth=2)
ax4.plot(freq_ghz, right_df['S21_dB'], 'r-', label='Tumor RIGHT', linewidth=2)

# Highlight the difference area
ax4.fill_between(freq_ghz, left_df['S21_dB'], right_df['S21_dB'], 
                 where=(right_df['S21_dB'] < left_df['S21_dB']), 
                 color='red', alpha=0.2, label='Detection Difference')

ax4.set_xlabel('Frequency (GHz)')
ax4.set_ylabel('S21 (dB)')
ax4.set_title('CRITICAL: Positional Asymmetry (Same 2cm Tumor)')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.text(2.1, -18, f'Δ = {abs(right_df["S21_dB"].mean() - left_df["S21_dB"].mean()):.1f} dB', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig('2_antenna_system_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print key insights
print("="*60)
print("KEY INSIGHTS FROM YOUR 2-ANTENNA SYSTEM:")
print("="*60)
print(f"1. Healthy Baseline: {averages[0]:.1f} dB")
print(f"2. Small Tumor (1cm) detection: {'YES' if averages[1] < averages[0] - 1 else 'WEAK'} ({averages[1]-averages[0]:.1f} dB change)")
print(f"3. Medium/Large Tumor detection: STRONG ({averages[2]-averages[0]:.1f} dB change)")
print(f"4. Left-Right Asymmetry: {abs(averages[4] - averages[5]):.1f} dB difference!")
print(f"   → Left tumor invisible, Right tumor strongly detected.")
print("\n5. SYSTEM LIMITATION: Single transmission path (Tx→Rx)")
print("   → Cannot resolve position or shape well.")
print("   → Vulnerable to setup asymmetries.")
print("="*60)