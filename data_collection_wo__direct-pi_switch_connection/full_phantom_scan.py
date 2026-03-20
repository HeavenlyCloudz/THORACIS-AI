# full_phantom_scan.py
"""
COMPLETE PHANTOM SCANNING PROTOCOL
"""
import serial
import time
import csv
import numpy as np
from datetime import datetime
import os

# Configuration
VNA_PORT = 'COM4'  
FREQ_START = 2e9
FREQ_STOP = 3e9
NUM_POINTS = 201  

# Path configurations
PATHS = [
    {'num': 1, 'name': '1→3', 'tx': 'RF1=+5V, RF2=GND', 'rx': 'RF1=+5V, RF2=GND'},
    {'num': 2, 'name': '1→4', 'tx': 'RF1=+5V, RF2=GND', 'rx': 'RF1=GND, RF2=+5V'},
    {'num': 3, 'name': '2→3', 'tx': 'RF1=GND, RF2=+5V', 'rx': 'RF1=+5V, RF2=GND'},
    {'num': 4, 'name': '2→4', 'tx': 'RF1=GND, RF2=+5V', 'rx': 'RF1=GND, RF2=+5V'}
]

# Test conditions
CONDITIONS = [
    {
        'name': '01_baseline_air',
        'description': 'EMPTY - No phantom, just air between antennas',
        'prompt': 'Remove ALL phantoms. Just air between antennas.'
    },
    {
        'name': '02_healthy_phantom_1',
        'description': 'HEALTHY PHANTOM #1 - First agar phantom (no tumor)',
        'prompt': 'Place HEALTHY phantom #1 between antennas'
    },
    {
        'name': '03_healthy_phantom_2',
        'description': 'HEALTHY PHANTOM #2 - Second agar phantom (different batch)',
        'prompt': 'Place HEALTHY phantom #2 between antennas'
    },
    {
        'name': '04_tumor_phantom',
        'description': 'TUMOR PHANTOM - Healthy phantom WITH embedded tumor simulant',
        'prompt': 'Place TUMOR phantom between antennas'
    }
]

def connect_vna():
    """Connect to VNA"""
    try:
        vna = serial.Serial(VNA_PORT, 115200, timeout=2)
        time.sleep(2)
        print(f"✅ Connected to VNA on {VNA_PORT}")
        return vna
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return None

def scan_path(vna, path_num, path_name):
    """Scan a single path and return data"""
    print(f"  Scanning Path {path_num} ({path_name})...", end='', flush=True)
    
    vna.write(b':sweep:data? s21\r\n')
    time.sleep(1)  # Waits for sweep
    
    data = vna.readline()
    if data:
        values = data.decode().strip().split(',')
        print(f" ✅ {len(values)} points")
        return values
    else:
        print(" ❌ No data")
        return None

def save_data(values, folder, path_num, path_name):
    """Save data to CSV file"""
    if not values:
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/path{path_num}_{path_name}_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frequency_Hz', 'S21_dB'])
        freqs = np.linspace(FREQ_START, FREQ_STOP, len(values))
        
        valid_points = 0
        for fq, val in zip(freqs, values):
            try:
                s21_val = float(val)
                writer.writerow([fq, s21_val])
                valid_points += 1
            except:
                pass
    
    print(f"    💾 Saved: {filename} ({valid_points} points)")
    return True

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def show_switch_guide():
    """Display switch configuration guide"""
    print("\n📋 SWITCH CONFIGURATION GUIDE:")
    print("-" * 50)
    for path in PATHS:
        print(f"Path {path['num']} ({path['name']}):")
        print(f"  TX: {path['tx']}")
        print(f"  RX: {path['rx']}")
    print("-" * 50)

def main():
    print_header("PULMO AI - COMPLETE PHANTOM SCANNING PROTOCOL")
    
    # Create main folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"phantom_data_{timestamp}"
    os.makedirs(base_folder, exist_ok=True)
    print(f"📁 Main folder: {base_folder}")
    
    # Show switch guide
    show_switch_guide()
    
    # Connect to VNA
    vna = connect_vna()
    if not vna:
        return
    
    # Run through each condition
    for cond_idx, condition in enumerate(CONDITIONS, 1):
        print_header(f"CONDITION {cond_idx}/{len(CONDITIONS)}: {condition['name']}")
        print(f"📝 {condition['description']}")
        print(f"\n👉 {condition['prompt']}")
        
        # Create condition folder
        cond_folder = f"{base_folder}/{condition['name']}"
        os.makedirs(cond_folder, exist_ok=True)
        
        input("\nPress ENTER when ready to start this condition...")
        
        # Scan all 4 paths for this condition
        for path in PATHS:
            print(f"\n  Path {path['num']} of 4")
            print(f"  Set TX: {path['tx']}")
            print(f"  Set RX: {path['rx']}")
            input("    Press ENTER after setting switches...")
            
            # Small settling delay
            time.sleep(0.5)
            
            # Scan and save
            values = scan_path(vna, path['num'], path['name'])
            if values:
                save_data(values, cond_folder, path['num'], path['name'])
        
        print(f"\n✅ Condition '{condition['name']}' complete!")
        
        # Ask if I want to continue
        if cond_idx < len(CONDITIONS):
            response = input("\nContinue to next condition? (y/n): ")
            if response.lower() != 'y':
                print("\n⚠️  Stopping early. Partial data saved.")
                break
    
    vna.close()
    
    print_header("SCANNING COMPLETE!")
    print(f"📁 All data saved in: {base_folder}")
    print("\nFolder structure:")
    for condition in CONDITIONS:
        path = f"{base_folder}/{condition['name']}"
        if os.path.exists(path):
            files = len([f for f in os.listdir(path) if f.endswith('.csv')])
            print(f"  📁 {condition['name']}: {files} files")
    
    print("\n🔍 NEXT STEP: Run analysis script")
    print(f"python analyze_phantom_data.py {base_folder}")

if __name__ == "__main__":
    main()
