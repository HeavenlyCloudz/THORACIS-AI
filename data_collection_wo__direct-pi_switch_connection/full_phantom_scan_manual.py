# full_phantom_scan_manual.py
"""
COMPLETE PHANTOM SCANNING PROTOCOL - MANUAL MODE (FAST VERSION)
Run this on your COMPUTER
You'll manually set paths on Pi when prompted

Total scans: 3 conditions × 3 rotations × 4 paths = 36 files
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
ROTATION_ENABLED = True      # Set to True to use rotations

# Path configurations (for display only)
PATHS = [
    {'num': 1, 'name': '1→3', 'description': 'Antenna 1 → Antenna 3'},
    {'num': 2, 'name': '1→4', 'description': 'Antenna 1 → Antenna 4'},
    {'num': 3, 'name': '2→3', 'description': 'Antenna 2 → Antenna 3'},
    {'num': 4, 'name': '2→4', 'description': 'Antenna 2 → Antenna 4'},
]

def capture_path(path_num, path_name, rotation=0, condition_name=""):
    """Capture S21 data for a single antenna path with metadata"""
    
    # Create filename with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"path{path_num}_rot{rotation}_{timestamp}.csv"
    
    # Ensure folder exists
    os.makedirs(condition_name, exist_ok=True)
    full_path = os.path.join(condition_name, filename)
    
    print(f"  📡 Capturing {path_name}...", end='', flush=True)
    
    try:
        with serial.Serial(VNA_PORT, BAUDRATE, timeout=2) as ser:
            time.sleep(1)
            ser.reset_input_buffer()
            
            # Send scan command (5 = freq + S21 data in dB format)
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5"
            ser.write((cmd + '\n').encode())
            time.sleep(1.5)  # Wait for sweep
            
            # Collect data
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
                            
                            # Convert to magnitude in dB
                            magnitude = (s21_real**2 + s21_imag**2)**0.5
                            magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                            
                            data_points.append([freq_hz, magnitude_db])
                            lines_collected += 1
                        except:
                            continue
            
            if len(data_points) == POINTS:
                # Save to CSV
                with open(full_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frequency_Hz', 'S21_dB'])
                    writer.writerows(data_points)
                
                # Calculate statistics
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
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def main():
    print_header("PULMO AI - FAST PHANTOM SCANNING PROTOCOL (MANUAL MODE)")
    
    print("\n📋 CONFIGURATION:")
    print(f"   VNA Port: {VNA_PORT}")
    print(f"   Frequency: {START_FREQ/1e9:.1f} - {STOP_FREQ/1e9:.1f} GHz")
    print(f"   Points: {POINTS}")
    print(f"   Rotations: {ROTATIONS}")
    print(f"\n📊 SCAN SUMMARY:")
    print(f"   • 3 Conditions (Baseline, Healthy, Tumor)")
    print(f"   • 3 Rotations per condition (0°, 120°, 240°)")
    print(f"   • 4 Paths per rotation")
    print(f"   • Total: 3 × 3 × 4 = 36 CSV files")
    print(f"\n🤖 CONTROL MODE: Manual")
    print("   You'll set paths on Pi when prompted")
    print("   Make sure pi_switch_controller.py is running on Pi")
    print("\n⏱️  Estimated time: ~15-20 minutes")
    
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
                print(f"⚠️  VNA connected but unexpected response")
    except Exception as e:
        print(f"❌ Cannot connect to VNA on {VNA_PORT}: {e}")
        print("   Check: VNA powered on, USB connected, correct port")
        return
    
    # Test conditions (ONLY 3 conditions now!)
    CONDITIONS = [
        {
            'name': '01_baseline_air',
            'description': 'EMPTY - No phantom, just air between antennas',
            'prompt': 'Remove ALL phantoms'
        },
        {
            'name': '02_healthy_phantom',
            'description': 'HEALTHY PHANTOM - Agar phantom (no tumor)',
            'prompt': 'Place HEALTHY phantom between antennas'
        },
        {
            'name': '03_tumor_phantom',
            'description': 'TUMOR PHANTOM - Healthy phantom with embedded tumor simulant',
            'prompt': 'Place TUMOR phantom between antennas'
        }
    ]
    
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
        'rotations': ROTATIONS if ROTATION_ENABLED else [0],
        'mode': 'manual',
        'conditions': []
    }
    
    # Run through each condition
    for cond_idx, condition in enumerate(CONDITIONS, 1):
        print_header(f"CONDITION {cond_idx}/{len(CONDITIONS)}: {condition['name']}")
        print(f"📝 {condition['description']}")
        print(f"\n👉 {condition['prompt']}")
        
        # Create condition folder
        cond_folder = f"{base_folder}/{condition['name']}"
        os.makedirs(cond_folder, exist_ok=True)
        
        input("\nPress ENTER when ready to start this condition...")
        
        # Get rotations to use
        rotations_to_use = ROTATIONS if ROTATION_ENABLED else [0]
        
        # Track successful scans
        total_scans = 0
        successful_scans = 0
        
        # Scan with rotations (1 scan per rotation - no repeat runs!)
        for rot_idx, rotation in enumerate(rotations_to_use, 1):
            print(f"\n{'='*50}")
            print(f"  🔄 ROTATION {rot_idx}/{len(rotations_to_use)}: {rotation}°")
            print(f"{'='*50}")
            
            # Scan all 4 paths for this rotation
            for path in PATHS:
                total_scans += 1
                
                # Prompt user to set path on Pi
                print(f"\n    🔧 Set Pi to Path {path['num']}: {path['description']}")
                print(f"       On Pi terminal, type: {path['num']}")
                input("       Press ENTER after setting path on Pi...")
                
                # Capture data
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
        print(f"   Files saved in: {cond_folder}/")
        
        # Store metadata
        metadata['conditions'].append({
            'name': condition['name'],
            'description': condition['description'],
            'planned_scans': total_scans,
            'successful_scans': successful_scans,
            'rotations': rotations_to_use,
            'files': [f.name for f in Path(cond_folder).glob('*.csv')]
        })
        
        # Ask if user wants to continue
        if cond_idx < len(CONDITIONS):
            response = input("\nContinue to next condition? (y/n): ")
            if response.lower() != 'y':
                print("\n⚠️  Stopping early")
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
    for condition in CONDITIONS:
        cond_path = Path(base_folder) / condition['name']
        if cond_path.exists():
            file_count = len(list(cond_path.glob('*.csv')))
            print(f"   📂 {condition['name']}/ : {file_count} files")
    
    print("\n🔍 NEXT STEP: Run analysis script")
    print(f"python analyze_phantom_data.py {base_folder}")

if __name__ == "__main__":
    main()
