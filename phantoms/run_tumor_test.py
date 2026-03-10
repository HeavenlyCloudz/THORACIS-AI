# run_tumor_test.py
import serial
import time
import csv
import numpy as np
from datetime import datetime

print("=== PULMO AI TUMOR DETECTION TEST ===")
print("\n1. First, run BASELINE (air) - already done")
print("2. Now run HEALTHY phantom - done")
print("3. Embed tumor simulant in phantom")
input("\nPress ENTER when tumor is embedded...")

# Connect to VNA
vna = serial.Serial('COM4', 115200, timeout=2)
time.sleep(2)

# Scan all 4 paths
paths = ['Path 1 (1→3)', 'Path 2 (1→4)', 'Path 3 (2→3)', 'Path 4 (2→4)']

for i, path_name in enumerate(paths, 1):
    print(f"\nScanning {path_name}...")
    
    vna.write(b':sweep:data? s21\r\n')
    time.sleep(1)
    
    data = vna.readline()
    if data:
        values = data.decode().strip().split(',')
        
        # Save with tumor label
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tumor_path{i}_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_Hz', 'S21_dB'])
            freqs = np.linspace(2e9, 3e9, len(values))
            for fq, val in zip(freqs, values):
                try:
                    writer.writerow([fq, float(val)])
                except:
                    pass
        print(f"Saved {filename}")

vna.close()
print("\n✅ Tumor test complete!")
