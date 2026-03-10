# analyze_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=== PULMO AI RESULTS ANALYSIS ===\n")

# Load data
baseline_files = list(Path('baseline_data').glob('path*.csv'))
healthy_files = list(Path('healthy_data').glob('path*.csv'))
tumor_files = list(Path('.').glob('tumor_path*.csv'))

# Load baseline
baseline_data = []
for f in sorted(baseline_files):
    df = pd.read_csv(f)
    baseline_data.append(df['S21_dB'].values)
baseline_avg = np.mean(baseline_data, axis=0)

# Load healthy
healthy_data = []
for f in sorted(healthy_files):
    df = pd.read_csv(f)
    healthy_data.append(df['S21_dB'].values)
healthy_avg = np.mean(healthy_data, axis=0)

# Load tumor
tumor_data = []
for f in sorted(tumor_files):
    df = pd.read_csv(f)
    tumor_data.append(df['S21_dB'].values)
tumor_avg = np.mean(tumor_data, axis=0)

freqs = np.linspace(2, 3, len(baseline_avg))

# Calculate key metrics
print("KEY RESULTS:")
print("-" * 40)

# Find best frequency (2.76 GHz peak)
peak_idx = np.argmax(baseline_avg)
peak_freq = freqs[peak_idx]
print(f"\nBest frequency: {peak_freq:.2f} GHz")
print(f"  Baseline: {baseline_avg[peak_idx]:.2f} dB")
print(f"  Healthy:  {healthy_avg[peak_idx]:.2f} dB")
print(f"  Tumor:    {tumor_avg[peak_idx]:.2f} dB")

healthy_drop = healthy_avg[peak_idx] - baseline_avg[peak_idx]
tumor_drop = tumor_avg[peak_idx] - healthy_avg[peak_idx]

print(f"\nHealthy phantom attenuation: {abs(healthy_drop):.2f} dB")
print(f"Tumor-induced drop: {abs(tumor_drop):.2f} dB")

if abs(tumor_drop) >= 4.9:
    print("\n✅ TUMOR DETECTED! (>4.9 dB threshold)")
else:
    print("\n⚠️  Tumor signal below threshold")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(freqs, baseline_avg, 'b-', label='Air Baseline', linewidth=2)
plt.plot(freqs, healthy_avg, 'g-', label='Healthy Phantom', linewidth=2)
plt.plot(freqs, tumor_avg, 'r-', label='Tumor Phantom', linewidth=2)

plt.axhline(y=baseline_avg[peak_idx]-4.9, color='r', linestyle='--', 
            alpha=0.5, label='Tumor Threshold (4.9 dB)')

plt.xlabel('Frequency (GHz)')
plt.ylabel('S21 (dB)')
plt.title('PULMO AI: Tumor Detection Validation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-50, 0)
plt.savefig('tumor_detection_result.png', dpi=300)
plt.show()

print("\n✅ Analysis complete! Check 'tumor_detection_result.png'")
