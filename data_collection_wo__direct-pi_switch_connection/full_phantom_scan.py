# full_phantom_scan.py
"""
COMPLETE PHANTOM SCANNING PROTOCOL
Script runs on COMPUTER (controls VNA)
Pi controls switches via SSH calling pi_switch_controller.py
"""
import serial
import time
import csv
import math
import os
import subprocess
from datetime import datetime
import numpy as np
from pathlib import Path

# CONFIGURATION
VNA_PORT = 'COM4'  # Change to your VNA port
BAUDRATE = 115200
START_FREQ = 2000000000  # 2.0 GHz
STOP_FREQ = 3000000000   # 3.0 GHz
POINTS = 201

# PI SSH CONFIGURATION
PI_IP = '192.168.1.201'  
PI_USER = 'anik'         
PI_PASSWORD = 'iloverollerchic'  
USE_SSH = True            

# SCAN CONFIGURATION
SCANS_PER_CONDITION = 3   # Number of repeat scans
ROTATIONS = [0, 120, 240] # Rotation angles (degrees)
ROTATION_ENABLED = False   # Set to True if you have rotation capability

def send_to_pi_ssh(command):
    """Send command to Pi via SSH with password authentication"""
    if not USE_SSH:
        return False
    
    try:
        # Using sshpass for password authentication
        result = subprocess.run(
            f'sshpass -p "{PI_PASSWORD}" ssh -o StrictHostKeyChecking=no {PI_USER}@{PI_IP} "{command}"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return True
        else:
            if result.stderr:
                print(f"  ⚠️ SSH error: {result.stderr[:100]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ SSH timeout - Pi not responding")
        return False
    except FileNotFoundError:
        print(f"  ⚠️ sshpass not installed!")
        print(f"     Install it: sudo apt install sshpass (Linux) or brew install sshpass (Mac)")
        print(f"     Or download for Windows from: https://sourceforge.net/projects/sshpass/")
        return False
    except Exception as e:
        print(f"  ⚠️ SSH error: {e}")
        return False

def set_path_on_pi(path_num):
    """Send path setting command to Pi via SSH"""
    if not USE_SSH:
        print(f"  🔧 MANUAL: Set Pi switches to Path {path_num}")
        return True
    
    # Call the pi_switch_controller.py script on the Pi
    command = f'cd /home/{PI_USER}/pulmo_ai_app && python3 pi_switch_controller.py {path_num}'
    
    print(f"  🤖 Setting Pi to Path {path_num}...")
    success = send_to_pi_ssh(command)
    
    if success:
        print(f"    ✅ Path {path_num} set")
    else:
        print(f"    ⚠️ Could not set path, check Pi connection")
    
    time.sleep(0.5)  # Allow switches to settle
    return success

def test_pi_connection():
    """Test if Pi is reachable via SSH"""
    if not USE_SSH:
        return True
    
    print("\n🔌 Testing Pi connection...")
    
    try:
        # Simple test command
        command = f'echo "Connected"'
        result = subprocess.run(
            f'sshpass -p "{PI_PASSWORD}" ssh -o StrictHostKeyChecking=no {PI_USER}@{PI_IP} "{command}"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and 'Connected' in result.stdout:
            print(f"✅ Pi connected at {PI_IP}")
            return True
        else:
            print(f"❌ Cannot connect to Pi at {PI_IP}")
            print("   Check: Pi is on, connected to network, SSH enabled")
            if result.stderr:
                print(f"   Error: {result.stderr[:100]}")
            return False
            
    except FileNotFoundError:
        print(f"❌ sshpass not installed!")
        print("   Install it:")
        print("     Linux: sudo apt install sshpass")
        print("     Mac: brew install hudochenkov/sshpass/sshpass")
        print("     Windows: Download from https://sourceforge.net/projects/sshpass/")
        return False
    except Exception as e:
        print(f"❌ Cannot connect to Pi: {e}")
        return False

def test_pi_switch_script():
    """Test the pi_switch_controller.py script on Pi"""
    if not USE_SSH:
        return True
    
    print("\n🔌 Testing Pi switch controller...")
    
    # Test the script exists and works
    command = f'cd /home/{PI_USER}/pulmo_ai_app && python3 pi_switch_controller.py --test'
    result = subprocess.run(
        f'sshpass -p "{PI_PASSWORD}" ssh -o StrictHostKeyChecking=no {PI_USER}@{PI_IP} "{command}"',
        shell=True,
        capture_output=True,
        text=True,
        timeout=15
    )
    
    if result.returncode == 0:
        print(f"✅ Pi switch controller working!")
        return True
    else:
        print(f"❌ Pi switch controller failed")
        print(f"   Make sure pi_switch_controller.py is in /home/{PI_USER}/pulmo_ai_app/")
        print(f"   Error: {result.stderr[:100]}")
        return False

def capture_path(path_num, path_name, rotation=0, run_num=1, condition_name=""):
    """Capture S21 data for a single antenna path with metadata"""
    
    # Create filename with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"path{path_num}_rot{rotation}_run{run_num}_{timestamp}.csv"
    
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
    global USE_SSH
    
    print_header("PULMO AI - COMPLETE PHANTOM SCANNING PROTOCOL")
    
    print("\n📋 CONFIGURATION:")
    print(f"   VNA Port: {VNA_PORT}")
    print(f"   Frequency: {START_FREQ/1e9:.1f} - {STOP_FREQ/1e9:.1f} GHz")
    print(f"   Points: {POINTS}")
    print(f"   Scans per condition: {SCANS_PER_CONDITION}")
    print(f"   Pi Control: {'SSH' if USE_SSH else 'Manual'}")
    if USE_SSH:
        print(f"   Pi IP: {PI_IP}")
        print(f"   Pi User: {PI_USER}")
    
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
    
    # Test Pi connection if using SSH
    if USE_SSH:
        if not test_pi_connection():
            response = input("\nContinue in manual mode? (y/n): ")
            if response.lower() != 'y':
                return
            USE_SSH = False
            print("\n⚠️  Switching to manual mode")
        else:
            # Test the switch script
            if not test_pi_switch_script():
                response = input("\nSwitch controller failed. Continue in manual mode? (y/n): ")
                if response.lower() != 'y':
                    return
                USE_SSH = False
                print("\n⚠️  Switching to manual mode")
    
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
        'use_ssh': USE_SSH,
        'pi_ip': PI_IP if USE_SSH else None,
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
        
        # Scan with rotations and multiple runs
        for rotation in rotations_to_use:
            for run_num in range(1, SCANS_PER_CONDITION + 1):
                print(f"\n  📸 Rotation: {rotation}°, Run: {run_num}/{SCANS_PER_CONDITION}")
                
                # Scan all 4 paths
                for path_num in range(1, 5):
                    path_names = {1: '1→3', 2: '1→4', 3: '2→3', 4: '2→4'}
                    path_name = path_names[path_num]
                    total_scans += 1
                    
                    # Set path on Pi via SSH
                    if USE_SSH:
                        if not set_path_on_pi(path_num):
                            print(f"    ⚠️ Failed to set path, skipping...")
                            continue
                    else:
                        print(f"    🔧 MANUAL: Set Pi switches to Path {path_num} ({path_name})")
                        input("      Press ENTER after setting switches...")
                    
                    # Capture data
                    success, stats = capture_path(
                        path_num, 
                        path_name,
                        rotation=rotation,
                        run_num=run_num,
                        condition_name=cond_folder
                    )
                    
                    if success:
                        successful_scans += 1
                    else:
                        print(f"    ⚠️ Failed, retrying...")
                        time.sleep(2)
                        success, stats = capture_path(
                            path_num, 
                            path_name,
                            rotation=rotation,
                            run_num=run_num,
                            condition_name=cond_folder
                        )
                        if success:
                            successful_scans += 1
                
                # Small pause between runs
                time.sleep(1)
        
        print(f"\n✅ Condition '{condition['name']}' complete!")
        print(f"   Successful scans: {successful_scans}/{total_scans}")
        
        # Store metadata
        metadata['conditions'].append({
            'name': condition['name'],
            'description': condition['description'],
            'planned_scans': total_scans,
            'successful_scans': successful_scans,
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
    
    # Count total files
    total_files = len([f for f in Path(base_folder).rglob('*.csv')])
    print(f"📊 Total CSV files: {total_files}")
    
    print("\n🔍 NEXT STEP: Run analysis script")
    print(f"python analyze_phantom_data.py {base_folder}")

if __name__ == "__main__":
    main()
