import serial
import time
import csv
import math
from datetime import datetime

# CONFIGURATION
PORT = 'COM4'  # or COM3 - use whichever works
BAUDRATE = 115200
START_FREQ = 2000000000  # 2.0 GHz
STOP_FREQ = 3000000000   # 3.0 GHz
POINTS = 201

def capture_path(path_num, path_name):
    """Capture S21 data for a single antenna path"""
    print(f"\n=== Capturing {path_name} ===")
    
    filename = f"path{path_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with serial.Serial(PORT, BAUDRATE, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        
        # Send scan command (5 = freq + S21 data)
        cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5"
        print(f"Sending: {cmd}")
        ser.write((cmd + '\n').encode())
        time.sleep(1)  # Wait for sweep to complete
        
        # Collect data
        data_points = []
        lines_collected = 0
        
        while lines_collected < POINTS:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            # Skip command echo and prompts
            if line and not line.startswith('ch>') and not line.startswith('scan'):
                parts = line.split()
                if len(parts) >= 3:  # freq, real, imag
                    try:
                        freq_hz = float(parts[0])
                        s21_real = float(parts[1])
                        s21_imag = float(parts[2])
                        
                        # Convert to magnitude in dB
                        magnitude = (s21_real**2 + s21_imag**2)**0.5
                        magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                        
                        data_points.append([freq_hz, magnitude_db])
                        lines_collected += 1
                        
                        if lines_collected % 50 == 0:
                            print(f"  Collected {lines_collected}/{POINTS} points...")
                    except:
                        continue
        
        # Save to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_Hz', 'S21_dB'])
            writer.writerows(data_points)
        
        avg_db = sum(d[1] for d in data_points) / len(data_points)
        print(f"✓ Saved {filename}")
        print(f"  Average S21: {avg_db:.2f} dB")
        print(f"  Range: {min(d[1] for d in data_points):.1f} to {max(d[1] for d in data_points):.1f} dB")
        
        return data_points

# MAIN CAPTURE LOOP
print("="*50)
print("PULMO AI - 4-ANTENNA ARRAY CAPTURE")
print("="*50)
print(f"Port: {PORT}, {START_FREQ/1e9:.1f}-{STOP_FREQ/1e9:.1f} GHz, {POINTS} points")
print("\nMake sure VNA is ON and connected.")
print("On your Pi, you'll set each path when prompted.\n")

paths = [
    (1, "Path 1 (1→3)"),
    (2, "Path 2 (1→4)"),
    (3, "Path 3 (2→3)"),
    (4, "Path 4 (2→4)")
]

for path_num, path_name in paths:
    input(f"\n{path_name} ready? Set path on Pi then press ENTER to capture...")
    capture_path(path_num, path_name)

print("\n" + "="*50)
print("✓ All 4 paths captured successfully!")
print("="*50)
input("\nPress Enter to exit...")
