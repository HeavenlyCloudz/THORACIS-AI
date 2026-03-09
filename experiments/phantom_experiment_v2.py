import serial
import time
import csv
import math
import os

# CONFIGURATION
PORT = 'COM3'
BAUDRATE = 115200
START_FREQ = 2000000000  # 2.0 GHz
STOP_FREQ = 3000000000   # 3.0 GHz
POINTS = 201

def send_command(ser, cmd):
    ser.write((cmd + '\n').encode())
    time.sleep(0.5)  # Important delay

def collect_scan(scan_name):
    print(f"\n=== Starting Scan: {scan_name} ===")
    
    with serial.Serial(PORT, BAUDRATE, timeout=2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()
        
        # Send scan command (5 = freq + S21 data)
        cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5"
        print(f"Command: {cmd}")
        send_command(ser, cmd)
        
        # Collect data
        data_points = []
        lines_collected = 0
        
        while lines_collected < POINTS:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            # Skip command echo and prompts
            if line and not line.startswith('ch>') and not line.startswith('scan'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        freq_hz = float(parts[0])
                        s21_real = float(parts[1])  # Real is in column 2
                        s21_imag = float(parts[2])  # Imag is in column 3
                        
                        magnitude = (s21_real**2 + s21_imag**2)**0.5
                        magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                        
                        data_points.append([freq_hz, magnitude_db])
                        lines_collected += 1
                        
                        if lines_collected % 50 == 0:
                            print(f"  Collected {lines_collected}/{POINTS} points...")
                    except:
                        continue
        
        # Save to CSV
        filename = f"{scan_name}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_Hz', 'S21_dB'])
            writer.writerows(data_points)
        
        avg_db = sum(d[1] for d in data_points) / len(data_points)
        print(f"✓ Scan complete: {filename}")
        print(f"  Average S21: {avg_db:.2f} dB")
        print(f"  Range: {min(d[1] for d in data_points):.1f} to {max(d[1] for d in data_points):.1f} dB")
        
        return avg_db

def load_air_baseline():
    """Load the previously saved air baseline data."""
    if os.path.exists('scan_air.csv'):
        print("\nLoading existing air baseline data from scan_air.csv...")
        with open('scan_air.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data = list(reader)
        
        # Calculate average from saved data
        s21_values = [float(row[1]) for row in data]
        avg_db = sum(s21_values) / len(s21_values)
        print(f"✓ Air baseline loaded: {avg_db:.2f} dB")
        return avg_db
    else:
        print("\n⚠️  No existing air baseline found. Running fresh scan...")
        return collect_scan("scan_air")

# MAIN EXPERIMENT
print("="*50)
print("PULMO AI - HIGH CONTRAST TUMOR EXPERIMENT")
print("="*50)

# Load or collect air baseline
air_avg = load_air_baseline()

print("\n1. Place your SALT WATER phantom between antennas.")
input("   Press Enter to scan HEALTHY phantom...")
healthy_avg = collect_scan("scan_phantom_healthy")

print("\n2. Insert HIGH-CONTRAST 'tumor' (air cavity/metal) into phantom.")
input("   Press Enter to scan PHANTOM WITH HIGH-CONTRAST TUMOR...")
tumor_avg = collect_scan("scan_phantom_tumor_highcontrast")

print("\n" + "="*50)
print("EXPERIMENT SUMMARY")
print("="*50)
print(f"Air baseline average S21:       {air_avg:.2f} dB")
print(f"Healthy phantom average S21:    {healthy_avg:.2f} dB")
print(f"Phantom + tumor average S21:    {tumor_avg:.2f} dB")
print(f"Signal drop (Air → Healthy):    {air_avg - healthy_avg:.2f} dB")
print(f"Signal drop (Healthy → Tumor):  {healthy_avg - tumor_avg:.2f} dB")

# Enhanced analysis
drop = healthy_avg - tumor_avg
if drop > 2.0:
    print("\n✅ EXCELLENT! Strong tumor signal detected (>2 dB)!")
    print("   Your system has high sensitivity for anomalies.")
elif drop > 0.5:
    print("\n✅ GOOD! Tumor signal detected.")
    print("   Your system can distinguish subtle differences.")
else:
    print("\n⚠️  Weak tumor signal. Try a metal object to verify detection capability.")
    print("   Then adjust phantom composition or tumor size.")

# Quick comparison to previous carrot experiment
if os.path.exists('scan_phantom_tumor.csv'):
    with open('scan_phantom_tumor.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        old_data = list(reader)
        old_avg = sum(float(row[1]) for row in old_data) / len(old_data)
    
    improvement = (healthy_avg - tumor_avg) - (healthy_avg - old_avg)
    print(f"\n📈 Improvement over carrot: {improvement:.2f} dB better contrast")

print(f"\nData saved to: scan_phantom_healthy.csv, scan_phantom_tumor_highcontrast.csv")
input("\nPress Enter to exit...")