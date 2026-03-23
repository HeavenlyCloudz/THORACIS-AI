# scan_tumor_only.py
"""
QUICK TUMOR PHANTOM SCAN - Just the good stuff!
Run this to scan ONLY your enhanced tumor phantom
"""
import serial
import time
import csv
import math
import os
from datetime import datetime
import numpy as np
from pathlib import Path

# CONFIGURATION
VNA_PORT = 'COM4'
BAUDRATE = 115200
START_FREQ = 2000000000  # 2.0 GHz
STOP_FREQ = 3000000000   # 3.0 GHz
POINTS = 201

# SCAN CONFIGURATION
ROTATIONS = [0, 120, 240]  # Do all 3 rotations for spatial data

# Path configurations
PATHS = [
    {'num': 1, 'name': '1→3', 'description': 'Antenna 1 → Antenna 3'},
    {'num': 2, 'name': '1→4', 'description': 'Antenna 1 → Antenna 4'},
    {'num': 3, 'name': '2→3', 'description': 'Antenna 2 → Antenna 3'},
    {'num': 4, 'name': '2→4', 'description': 'Antenna 2 → Antenna 4'},
]

def capture_path(path_num, path_name, rotation=0, condition_name="tumor_phantom_enhanced"):
    """Capture S21 data for a single antenna path"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"path{path_num}_rot{rotation}_{timestamp}.csv"
    
    os.makedirs(condition_name, exist_ok=True)
    full_path = os.path.join(condition_name, filename)
    
    print(f"  📡 Capturing {path_name}...", end='', flush=True)
    
    try:
        with serial.Serial(VNA_PORT, BAUDRATE, timeout=2) as ser:
            time.sleep(1)
            ser.reset_input_buffer()
            
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5"
            ser.write((cmd + '\n').encode())
            time.sleep(1.5)
            
            data_points = []
            lines_collected = 0
            timeout_start = time.time()
            
            while lines_collected < POINTS and (time.time() - timeout_start) < 15:
                line = ser.readline().decode('ascii', errors='ignore').strip()
                if line and not line.startswith('ch>') and not line.startswith('scan'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            freq_hz = float(parts[0])
                            s21_real = float(parts[1])
                            s21_imag = float(parts[2])
                            
                            magnitude = (s21_real**2 + s21_imag**2)**0.5
                            magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                            
                            data_points.append([freq_hz, magnitude_db])
                            lines_collected += 1
                        except:
                            continue
            
            if len(data_points) == POINTS:
                with open(full_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frequency_Hz', 'S21_dB'])
                    writer.writerows(data_points)
                
                s21_values = [d[1] for d in data_points]
                avg_db = np.mean(s21_values)
                print(f" ✅ avg={avg_db:.1f}dB")
                return True, avg_db
            else:
                print(f" ⚠️ Only {len(data_points)}/{POINTS} points")
                return False, None
                
    except Exception as e:
        print(f" ❌ Error: {e}")
        return False, None

def main():
    print("="*70)
    print("🔥 TUMOR PHANTOM SCAN - ENHANCED RECIPE (More Salt + Vanilla)")
    print("="*70)
    
    print("\n📋 YOUR ENHANCED TUMOR PHANTOM:")
    print("   ✓ More salt added (higher conductivity)")
    print("   ✓ Vanilla extract (increases water content)")
    print("   ✓ Should create STRONGER contrast with healthy tissue")
    
    # Test VNA
    print("\n🔌 Checking VNA connection...")
    try:
        with serial.Serial(VNA_PORT, BAUDRATE, timeout=2) as test_ser:
            test_ser.write(b'info\n')
            time.sleep(0.5)
            response = test_ser.read_all().decode()
            print(f"✅ VNA connected on {VNA_PORT}")
    except Exception as e:
        print(f"❌ Cannot connect to VNA: {e}")
        return
    
    # Create folder for enhanced tumor
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"tumor_enhanced_{timestamp}"
    print(f"\n📁 Saving to: {folder_name}")
    
    # Scan each rotation
    all_readings = []
    
    for rotation in ROTATIONS:
        print(f"\n{'='*50}")
        print(f"📸 ROTATION: {rotation}°")
        print(f"{'='*50}")
        
        if rotation == 0:
            print("   Position: Tumor facing antennas normally")
        elif rotation == 120:
            print("   Position: Rotate phantom 120°")
        else:
            print("   Position: Rotate phantom 240°")
        
        input("\nPress ENTER after positioning phantom...")
        
        rotation_readings = []
        
        for path in PATHS:
            print(f"\n  🔧 Set Pi to Path {path['num']}: {path['description']}")
            print(f"     On Pi terminal, type: {path['num']}")
            input("     Press ENTER after setting path...")
            
            success, avg_db = capture_path(
                path['num'], 
                path['name'],
                rotation=rotation,
                condition_name=folder_name
            )
            
            if success:
                rotation_readings.append(avg_db)
                all_readings.append(avg_db)
        
        print(f"\n  ✅ Rotation {rotation}° complete!")
        print(f"     Avg S21 for this rotation: {np.mean(rotation_readings):.1f} dB")
    
    # Quick analysis
    print("\n" + "="*70)
    print("📊 QUICK RESULTS")
    print("="*70)
    
    # Load baseline from previous scan if available
    baseline_folders = sorted(Path('.').glob('phantom_data_*'))
    if baseline_folders:
        latest_baseline = baseline_folders[-1]
        baseline_path = latest_baseline / '01_baseline_air'
        
        if baseline_path.exists():
            import pandas as pd
            baseline_files = list(baseline_path.glob('*.csv'))
            baseline_avgs = []
            
            for f in baseline_files[:4]:  # First 4 paths
                df = pd.read_csv(f)
                baseline_avgs.append(df['S21_dB'].mean())
            
            baseline_avg = np.mean(baseline_avgs)
            tumor_avg = np.mean(all_readings)
            drop = baseline_avg - tumor_avg
            
            print(f"\n📈 COMPARISON:")
            print(f"   Air Baseline (from previous): {baseline_avg:.2f} dB")
            print(f"   Enhanced Tumor Phantom:       {tumor_avg:.2f} dB")
            print(f"   Signal Drop:                  {drop:.2f} dB")
            
            if drop >= 4.9:
                print(f"\n🎉 SUCCESS! Tumor detected! ({drop:.1f} dB drop > 4.9 dB threshold)")
                print("   Your enhanced recipe worked!")
            elif drop >= 2.0:
                print(f"\n👍 Good improvement! ({drop:.1f} dB drop)")
                print("   Getting closer to the 4.9 dB threshold")
            else:
                print(f"\n⚠️  Still below threshold ({drop:.1f} dB drop)")
                print("   Try adding more salt or increasing tumor size")
    
    print(f"\n📁 All data saved in: {folder_name}/")
    print(f"\n🚀 NEXT: Compare with baseline using:")
    print(f"   python compare_tumor_to_baseline.py {folder_name}")
    
    # Save metadata
    import json
    with open(f"{folder_name}/scan_metadata.json", 'w') as f:
        json.dump({
            'type': 'enhanced_tumor',
            'recipe': 'added more salt + vanilla extract',
            'rotations': ROTATIONS,
            'num_scans': len(all_readings),
            'avg_s21': float(np.mean(all_readings)),
            'timestamp': timestamp
        }, f, indent=2)

if __name__ == "__main__":
    main()
