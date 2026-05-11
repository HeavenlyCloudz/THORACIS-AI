# complete_full_phantom_scan_all_paths.py
"""
COMPLETE PHANTOM SCANNING PROTOCOL - ALL 4 PATHS
UPDATED: 4 phantom types (Dynamic Healthy, Dynamic Tumor, Ex Vivo Healthy, Ex Vivo Tumor)
Run this on your COMPUTER
You'll manually set paths on Pi when prompted
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

# SCAN CONFIGURATION
ROTATIONS = [0, 120, 240]  # Rotation angles (degrees)
ROTATION_ENABLED = True

# Path configurations - ALL 4 PATHS
PATHS = [
    {'num': 1, 'name': '1→3', 'description': 'Antenna 1 → Antenna 3 (opposite)'},
    {'num': 2, 'name': '1→4', 'description': 'Antenna 1 → Antenna 4 (diagonal)'},
    {'num': 3, 'name': '2→3', 'description': 'Antenna 2 → Antenna 3 (diagonal)'},
    {'num': 4, 'name': '2→4', 'description': 'Antenna 2 → Antenna 4 (opposite)'},
]

def capture_path(path_num, path_name, rotation=0, condition_name="", phantom_id=""):
    """Capture S21 data for a single antenna path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{phantom_id}_path{path_num}_rot{rotation}_{timestamp}.csv"
    
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
                            
                            data_points.append([freq_hz, magnitude_db, s21_real, s21_imag])
                            lines_collected += 1
                        except:
                            continue
            
            if len(data_points) == POINTS:
                with open(full_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frequency_Hz', 'S21_dB', 'S21_Real', 'S21_Imag'])
                    writer.writerows(data_points)
                
                s21_values = [d[1] for d in data_points]
                avg_db = np.mean(s21_values)
                
                print(f" ✅ {len(data_points)} pts, avg={avg_db:.1f}dB")
                return True, {'avg': avg_db}
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
    print_header("PULMO AI - PHANTOM SCANNING (ALL 4 PATHS)")
    
    print("\n📋 CONFIGURATION:")
    print(f"   VNA Port: {VNA_PORT}")
    print(f"   Frequency: {START_FREQ/1e9:.1f} - {STOP_FREQ/1e9:.1f} GHz")
    print(f"   Points: {POINTS}")
    print(f"   Rotations: {ROTATIONS}")
    print(f"\n📡 ANTENNA PATHS (ALL 4):")
    for path in PATHS:
        print(f"   Path {path['num']}: {path['description']}")
    
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
    
    # =========================================================
    # NEW PHANTOM CONFIGURATION
    # =========================================================
    
    # Air baseline (always first)
    AIR_BASELINE = {
        'name': '00_air_baseline',
        'description': 'AIR BASELINE - No phantom, just air between antennas',
        'prompt': 'Remove ALL phantoms from between antennas',
        'phantom_type': 'baseline',
        'class_label': 0
    }
    
    # Dynamic Agar Phantoms (12 total: 6 healthy, 6 tumor)
    DYNAMIC_PHANTOMS = []
    for i in range(1, 7):  # 6 healthy dynamic
        DYNAMIC_PHANTOMS.append({
            'name': f'01_dynamic_healthy_{i}',
            'description': f'DYNAMIC AGAR HEALTHY {i} - Inflatable balloon phantom, NO tumor',
            'prompt': f'Place Dynamic Healthy Phantom {i} between antennas',
            'phantom_type': 'dynamic_healthy',
            'class_label': 1  # Healthy
        })
    for i in range(1, 7):  # 6 tumor dynamic
        DYNAMIC_PHANTOMS.append({
            'name': f'02_dynamic_tumor_{i}',
            'description': f'DYNAMIC AGAR TUMOR {i} - Inflatable balloon WITH graphite tumor',
            'prompt': f'Place Dynamic Tumor Phantom {i} between antennas',
            'phantom_type': 'dynamic_tumor',
            'class_label': 2  # Tumor
        })
    
    # Ex Vivo Chicken Phantoms (4 total: 2 healthy, 2 tumor)
    EX_VIVO_PHANTOMS = []
    for i in range(1, 3):  # 2 healthy ex vivo
        EX_VIVO_PHANTOMS.append({
            'name': f'03_exvivo_healthy_{i}',
            'description': f'EX VIVO CHICKEN HEALTHY {i} - Real chicken breast, NO tumor',
            'prompt': f'Place Ex Vivo Healthy Phantom {i} between antennas',
            'phantom_type': 'exvivo_healthy',
            'class_label': 1  # Healthy
        })
    for i in range(1, 3):  # 2 tumor ex vivo
        EX_VIVO_PHANTOMS.append({
            'name': f'04_exvivo_tumor_{i}',
            'description': f'EX VIVO CHICKEN TUMOR {i} - Real chicken breast WITH graphite tumor',
            'prompt': f'Place Ex Vivo Tumor Phantom {i} between antennas',
            'phantom_type': 'exvivo_tumor',
            'class_label': 2  # Tumor
        })
    
    # Combine all phantoms in order
    ALL_PHANTOMS = [AIR_BASELINE] + DYNAMIC_PHANTOMS + EX_VIVO_PHANTOMS
    
    print(f"\n📊 SCAN SUMMARY:")
    print(f"   • Air Baseline: 1 scan set")
    print(f"   • Dynamic Agar Healthy: 6 phantoms")
    print(f"   • Dynamic Agar Tumor: 6 phantoms")
    print(f"   • Ex Vivo Chicken Healthy: 2 phantoms")
    print(f"   • Ex Vivo Chicken Tumor: 2 phantoms")
    print(f"   • TOTAL PHANTOMS: {len(ALL_PHANTOMS)}")
    print(f"\n   • Per phantom: 3 rotations × 4 paths = 12 CSV files")
    print(f"   • TOTAL CSV FILES: {len(ALL_PHANTOMS) * 12} = {len(ALL_PHANTOMS) * 12}")
    
    # Create main folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"phantom_data_{timestamp}"
    os.makedirs(base_folder, exist_ok=True)
    print(f"\n📁 Main folder: {base_folder}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'vna_port': VNA_PORT,
        'freq_start_ghz': START_FREQ/1e9,
        'freq_stop_ghz': STOP_FREQ/1e9,
        'num_points': POINTS,
        'paths': [1, 2, 3, 4],
        'rotations': ROTATIONS if ROTATION_ENABLED else [0],
        'phantoms': []
    }
    
    # Scan each phantom
    for phantom_idx, phantom in enumerate(ALL_PHANTOMS, 1):
        print_header(f"PHANTOM {phantom_idx}/{len(ALL_PHANTOMS)}: {phantom['name']}")
        print(f"📝 {phantom['description']}")
        print(f"\n👉 {phantom['prompt']}")
        
        # Create condition folder
        cond_folder = f"{base_folder}/{phantom['name']}"
        os.makedirs(cond_folder, exist_ok=True)
        
        input("\nPress ENTER when phantom is in place...")
        
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
                    condition_name=cond_folder,
                    phantom_id=phantom['name']
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
                        condition_name=cond_folder,
                        phantom_id=phantom['name']
                    )
                    if success:
                        successful_scans += 1
            
            print(f"\n  ✅ Rotation {rotation}° complete!")
            time.sleep(0.5)
        
        print(f"\n✅ Phantom '{phantom['name']}' complete!")
        print(f"   Successful scans: {successful_scans}/{total_scans}")
        
        metadata['phantoms'].append({
            'name': phantom['name'],
            'description': phantom['description'],
            'phantom_type': phantom['phantom_type'],
            'class_label': phantom['class_label'],
            'planned_scans': total_scans,
            'successful_scans': successful_scans,
            'rotations': rotations_to_use,
            'files': [f.name for f in Path(cond_folder).glob('*.csv')]
        })
        
        # Ask to continue
        if phantom_idx < len(ALL_PHANTOMS):
            response = input("\nContinue to next phantom? (y/n): ")
            if response.lower() != 'y':
                print("\n⚠️ Stopping early")
                break
    
    # Save metadata
    import json
    with open(f"{base_folder}/scan_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_header("SCANNING COMPLETE!")
    print(f"📁 All data saved in: {base_folder}")
    
    # Count total files
    total_files = len([f for f in Path(base_folder).rglob('*.csv')])
    print(f"📊 Total CSV files: {total_files}")
    
    print("\n📁 Folder structure:")
    for phantom in metadata['phantoms']:
        phantom_path = Path(base_folder) / phantom['name']
        if phantom_path.exists():
            file_count = len(list(phantom_path.glob('*.csv')))
            print(f"   📂 {phantom['name']}/ : {file_count} files")
    
    print("\n🔍 NEXT STEP: Run analysis script")
    print(f"   python analyze_phantom_data_all_paths_v2.py {base_folder}")

if __name__ == "__main__":
    main()
