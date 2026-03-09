import pandas as pd
import matplotlib.pyplot as plt
import glob

# Find all CSV files
csv_files = sorted(glob.glob("path*.csv"))

if not csv_files:
    print("No CSV files found!")
    exit()

print(f"Found {len(csv_files)} files")

# Create figure
plt.figure(figsize=(12, 8))

# Plot each file
for csv_file in csv_files:
    try:
        data = pd.read_csv(csv_file)
        if len(data) > 0:
            # Extract path number from filename
            path_num = csv_file.split('_')[0]
            
            plt.plot(data['Frequency_Hz']/1e9, data['S21_dB'], 
                    label=path_num, linewidth=2)
            print(f"✓ {csv_file}: {len(data)} points, "
                  f"avg={data['S21_dB'].mean():.1f} dB")
    except Exception as e:
        print(f"✗ Error loading {csv_file}: {e}")

plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('S21 (dB)', fontsize=12)
plt.title('PULMO AI - 4-Antenna Array Measurements', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('all_paths_plot.png', dpi=150)
plt.show()

print("\nPlot saved as 'all_paths_plot.png'")
