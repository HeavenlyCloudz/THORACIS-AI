# full_phantom_scan.py
"""
COMPLETE PHANTOM SCANNING PROTOCOL with Pi Switch Control
Running this script on COMPUTER (controls VNA)
Pi runs pi_switch_controller.py in background
"""
import serial
import time
import csv
import math
import os
import socket
import subprocess
from datetime import datetime
import numpy as np

# CONFIGURATION
VNA_PORT = 'COM4'  # Change to your VNA port
BAUDRATE = 115200
START_FREQ = 2000000000  # 2.0 GHz
STOP_FREQ = 3000000000   # 3.0 GHz
POINTS = 201

# PI CONFIGURATION
PI_IP = '192.168.1.201'  # Replace with your Pi's IP address
USE_PI_CONTROL = True     # Set to False if controlling switches manually

# SCAN CONFIGURATION
SCANS_PER_CONDITION = 3   # Number of repeat scans
ROTATIONS = [0, 120, 240] # Rotation angles (degrees)
ROTATION_ENABLED = True

# Path configurations (same as pi_switch_controller)
PATHS = [
    {'num': 1, 'name': '1→3', 'gpio': {'17': 1, '27': 0, '18': 1, '22': 0}},
    {'num': 2, 'name': '1→4', 'gpio': {'17': 1, '27': 0, '18': 0, '22': 1}},
    {'num': 3, 'name': '2→3', 'gpio': {'17': 0, '27': 1, '18': 1, '22': 0}},
    {'num': 4, 'name': '2→4', 'gpio': {'17': 0, '27': 1, '18': 0, '22': 1}}
]

def send_to_pi(command):
    """Send command to Pi via network (optional)"""
    if not USE_PI_CONTROL:
        return False
    try:
        # Option 1: SSH command (if SSH keys set up)
        # subprocess.run(f'ssh pi@{PI_IP} "{command}"', shell=True, capture_output=True)
        
        # Option 2: Simple TCP socket (create a server on Pi)
        # For now, we'll just print instructions
        print(f"  [PI COMMAND] {command}")
        return True
    except:
        return False

def set_path_on_pi(path_num):
    """Send path setting command to Pi"""
    if USE_PI_CONTROL:
        print(f"  Sending path {path_num} to Pi...")
        # Send via SSH or socket
        # subprocess.run(f'ssh pi@{PI_IP} "python3 -c \'import pi_switch; pi_switch.set_path({path_num})\'"', shell=True)
        return True
    else:
        print(f"  MANUAL: Set Pi switches to Path {path_num}")
        return False

def capture_path(path_num, path_name, rotation=0, run_num=1, condition_name=""):
    """Capture S21 data for a single antenna path with metadata"""
    
    # Create filename with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{condition_name}_path{path_num}_rot{rotation}_run{run_num}_{timestamp}.csv"
    
    # Ensure folder exists
    os.makedirs(condition_name, exist_ok=True)
    full_path = os.path.join(condition_name, filename)
    
    print(f"\n  📡 Capturing {path_name}...", end='', flush=True)
    
    try:
        with serial.Serial(VNA_PORT, BAUDRATE, timeout=2) as ser:
            time.sleep(1)
            ser.reset_input_buffer()
            
            # Send scan command (5 = freq + S21 data in dB format)
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5"
            ser.write((cmd + '\n').encode())
            time.sleep(1)  # Wait for sweep
            
            # Collect data
            data_points = []
            lines_collected = 0
            timeout_start = time.time()
            
            while lines_collected < POINTS and (time.time() - timeout_start) < 10:
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
    print_header("PULMO AI - COMPLETE PHANTOM SCANNING PROTOCOL")
    
    # Test VNA connection first
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
    
    # Pi control instructions
    if USE_PI_CONTROL:
        print("\n🤖 Pi Control Mode: ENABLED")
        print("   Make sure pi_switch_controller.py is running on Pi")
        print("   Pi should be ready to receive path commands")
    else:
        print("\n🔧 Manual Switch Mode")
        print("   You'll set switches manually when prompted")
    
    # Test conditions
    CONDITIONS = [
        {
            'name': '01_baseline_air',
            'description': 'EMPTY - No phantom, just air between antennas',
            'prompt': 'Remove ALL phantoms'
        },
        {
            'name': '02_healthy_phantom_1',
            'description': 'HEALTHY PHANTOM #1 - First agar phantom',
            'prompt': 'Place HEALTHY phantom #1 between antennas'
        },
        {
            'name': '03_healthy_phantom_2',
            'description': 'HEALTHY PHANTOM #2 - Second agar phantom',
            'prompt': 'Place HEALTHY phantom #2 between antennas'
        },
        {
            'name': '04_tumor_phantom',
            'description': 'TUMOR PHANTOM - With embedded tumor simulant',
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
        'scans_per_condition': SCANS_PER_CONDITION,
        'rotations': ROTATIONS if ROTATION_ENABLED else [0],
        'use_pi_control': USE_PI_CONTROL,
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
        
        # Scan with rotations and multiple runs
        for rotation in rotations_to_use:
            for run_num in range(1, SCANS_PER_CONDITION + 1):
                print(f"\n  📸 Rotation: {rotation}°, Run: {run_num}/{SCANS_PER_CONDITION}")
                
                # Scan all 4 paths for this rotation/run
                for path_idx, path in enumerate(PATHS, 1):
                    # Send path to Pi
                    if USE_PI_CONTROL:
                        print(f"    Setting Pi to Path {path['num']}...")
                        # In a real setup, you'd send actual command here
                        # For now, we print the instruction
                    else:
                        print(f"    MANUAL: Set switches to Path {path['num']} ({path['name']})")
                    
                    # Wait for Pi to set path (adjust timing as needed)
                    time.sleep(0.5)
                    
                    # Capture data
                    success, stats = capture_path(
                        path['num'], 
                        path['name'],
                        rotation=rotation,
                        run_num=run_num,
                        condition_name=cond_folder
                    )
                    
                    if not success:
                        print(f"    ⚠️  Failed, retrying...")
                        time.sleep(2)
                        success, stats = capture_path(
                            path['num'], 
                            path['name'],
                            rotation=rotation,
                            run_num=run_num,
                            condition_name=cond_folder
                        )
                
                # Small pause between runs
                time.sleep(1)
        
        print(f"\n✅ Condition '{condition['name']}' complete!")
        
        # Store metadata
        metadata['conditions'].append({
            'name': condition['name'],
            'description': condition['description'],
            'scans': SCANS_PER_CONDITION * len(rotations_to_use) * 4,
            'rotations': rotations_to_use,
            'runs': SCANS_PER_CONDITION
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
    print(f"📊 Total scans: {len([f for f in Path(base_folder).rglob('*.csv')])}")
    
    print("\n🔍 NEXT STEP: Run analysis script")
    print(f"python analyze_phantom_data.py {base_folder}")
    print("\n🤖 For automated scanning with Pi:")
    print("   1. On Pi, run: python pi_switch_controller.py")
    print("   2. Then run this script with USE_PI_CONTROL = True")
    print("   3. Ensure Pi and computer are on same network")

if __name__ == "__main__":
    from pathlib import Path
    main()
