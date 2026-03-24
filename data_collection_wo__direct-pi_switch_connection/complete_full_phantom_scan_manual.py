# complete_full_phantom_scan_manual.py - UPDATED
"""
COMPLETE PHANTOM SCANNING PROTOCOL - MANUAL MODE (UPDATED)
Only scans Path 1 (1→3) and Path 2 (2→4) - the two opposite-facing antenna pairs
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
VNA_PORT = 'COM4'  # Change to your VNA port
BAUDRATE = 115200
START_FREQ = 2000000000  # 2.0 GHz
STOP_FREQ = 3000000000   # 3.0 GHz
POINTS = 201

# SCAN CONFIGURATION - Only 2 paths now
PATHS = [
    {'num': 1, 'name': '1→3', 'description': 'Antenna 1 → Antenna 3 (opposite, aligned)'},
    {'num': 2, 'name': '2→4', 'description': 'Antenna 2 → Antenna 4 (opposite, aligned)'},
]

# Rotation - you can still use rotations to test different tumor positions
ROTATIONS = [0, 120, 240]
ROTATION_ENABLED = True

def capture_path(path_num, path_name, rotation=0, condition_name=""):
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
                min_db = np.min(s21_values)
                max_db = np.max(s21_values)
                
                print(f" ✅ {len(data_points)} pts, avg={avg_db:.1f}dB")
                return True, {'avg': avg_db, 'min': min_db, 'max': max_db}
            else:
                print(f" ⚠️ Only {len(data_points)}/{POINTS} points")
                return False, None
                
    except Exception as e:
        print(f" ❌ Error: {e}")
        return False, None

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def main():
    print_header("PULMO AI - PHANTOM SCANNING (OPTIMIZED: PATHS 1 & 2 ONLY)")
    
    print("\n📋 CONFIGURATION:")
    print(f"   VNA Port: {VNA_PORT}")
    print(f"   Frequency: {START_FREQ/1e9:.1f} - {STOP_FREQ/1e9:.1f} GHz")
    print(f"   Points: {POINTS}")
    print(f"\n📡 ANTENNA PATHS (OPTIMIZED):")
    print(f"   Path 1: Antenna 1 → Antenna 3 (opposite, aligned)")
    print(f"   Path 2: Antenna 2 → Antenna 4 (opposite, aligned)")
    print(f"\n📊 SCAN SUMMARY:")
    print(f"   • 3 Conditions (Air Baseline, Healthy, Tumor)")
    print(f"   • 3 Rotations per condition (0°, 120°, 240°)")
    print(f"   • 2 Paths per rotation")
    print(f"   • Total: 3 × 3 × 2 = 18 CSV files (was 36)")
    print(f"\n⏱️  Estimated time: ~8-10 minutes")
    
    # Test VNA connection
    print("\n🔌 Checking VNA connection...")
    try:
        with serial.Serial(VNA_PORT, BAUDRATE, timeout=2) as test_ser:
            test_ser.write(b'info\n')
            time.sleep(0.5)
            response = test_ser.read_all().decode()
            if 'ch>' in response or 'NanoVNA' in response:
                print(f"✅ VNA connected on {VNA_PORT}")
            else:
                print(f"⚠️ VNA connected but unexpected response")
    except Exception as e:
        print(f"❌ Cannot connect to VNA on {VNA_PORT}: {e}")
        return
    
    CONDITIONS = [
        {
            'name': '01_baseline_air',
            'description': 'EMPTY - No phantom, just air',
            'prompt': 'Remove ALL phantoms, ensure antennas face each other'
        },
        {
            'name': '02_healthy_phantom',
            'description': 'HEALTHY PHANTOM - Agar phantom (no tumor)',
            'prompt': 'Place HEALTHY phantom between antennas'
        },
        {
            'name': '03_tumor_phantom',
            'description': 'TUMOR PHANTOM - With embedded tumor simulant',
            'prompt': 'Place TUMOR phantom between antennas'
        }
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"phantom_data_{timestamp}"
    os.makedirs(base_folder, exist_ok=True)
    print(f"\n📁 Main folder: {base_folder}")
    
    metadata = {
        'timestamp': timestamp,
        'vna_port': VNA_PORT,
        'freq_start_ghz': START_FREQ/1e9,
        'freq_stop_ghz': STOP_FREQ/1e9,
        'num_points': POINTS,
        'paths_used': ['1→3', '2→4'],
        'rotations': ROTATIONS if ROTATION_ENABLED else [0],
        'conditions': []
    }
    
    for cond_idx, condition in enumerate(CONDITIONS, 1):
        print_header(f"CONDITION {cond_idx}/{len(CONDITIONS)}: {condition['name']}")
        print(f"📝 {condition['description']}")
        print(f"\n👉 {condition['prompt']}")
        
        cond_folder = f"{base_folder}/{condition['name']}"
        os.makedirs(cond_folder, exist_ok=True)
        
        input("\nPress ENTER when ready...")
        
        rotations_to_use = ROTATIONS if ROTATION_ENABLED else [0]
        total_scans = 0
        successful_scans = 0
        
        for rot_idx, rotation in enumerate(rotations_to_use, 1):
            print(f"\n{'='*50}")
            print(f"  🔄 ROTATION {rot_idx}/{len(rotations_to_use)}: {rotation}°")
            print(f"{'='*50}")
            
            for path in PATHS:
                total_scans += 1
                
                print(f"\n    🔧 Set Pi to Path {path['num']}: {path['description']}")
                print(f"       On Pi terminal, type: {path['num']}")
                input("       Press ENTER after setting path on Pi...")
                
                success, stats = capture_path(
                    path['num'], 
                    path['name'],
                    rotation=rotation,
                    condition_name=cond_folder
                )
                
                if success:
                    successful_scans += 1
                else:
                    print(f"    ⚠️ Failed, retrying...")
                    time.sleep(2)
                    success, stats = capture_path(
                        path['num'], 
                        path['name'],
                        rotation=rotation,
                        condition_name=cond_folder
                    )
                    if success:
                        successful_scans += 1
            
            print(f"\n  ✅ Rotation {rotation}° complete!")
            time.sleep(0.5)
        
        print(f"\n✅ Condition '{condition['name']}' complete!")
        print(f"   Successful scans: {successful_scans}/{total_scans}")
        
        metadata['conditions'].append({
            'name': condition['name'],
            'description': condition['description'],
            'planned_scans': total_scans,
            'successful_scans': successful_scans,
            'rotations': rotations_to_use,
            'files': [f.name for f in Path(cond_folder).glob('*.csv')]
        })
        
        if cond_idx < len(CONDITIONS):
            response = input("\nContinue to next condition? (y/n): ")
            if response.lower() != 'y':
                print("\n⚠️ Stopping early")
                break
    
    import json
    with open(f"{base_folder}/scan_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_header("SCANNING COMPLETE!")
    print(f"📁 All data saved in: {base_folder}")
    
    total_files = len([f for f in Path(base_folder).rglob('*.csv')])
    print(f"📊 Total CSV files: {total_files}")
    
    print("\n🔍 NEXT STEP: Run analysis script")
    print(f"   python analyze_phantom_data.py {base_folder}")

if __name__ == "__main__":
    main()
