# full_phantom_scan.py
"""
COMPLETE PHANTOM SCANNING PROTOCOL - WITH MULTIPLE SCANS
"""
import serial
import time
import csv
import numpy as np
from datetime import datetime
import os

# ============= CONFIGURATION =============
VNA_PORT = 'COM4' 
FREQ_START = 2e9
FREQ_STOP = 3e9
NUM_POINTS = 201

# SCANNING CONFIGURATION
SCANS_PER_CONDITION = 3      # Number of repeat scans (IMPERATIVE for reproducibility)
ROTATIONS = [0, 120, 240]    # Rotation angles in degrees (optional, for spatial variation)
USE_ROTATIONS = False         # Set to True if you can rotate phantoms

# ============= PATH CONFIGURATIONS =============
PATHS = [
    {'num': 1, 'name': '1→3', 'tx': 'RF1=+5V, RF2=GND', 'rx': 'RF1=+5V, RF2=GND'},
    {'num': 2, 'name': '1→4', 'tx': 'RF1=+5V, RF2=GND', 'rx': 'RF1=GND, RF2=+5V'},
    {'num': 3, 'name': '2→3', 'tx': 'RF1=GND, RF2=+5V', 'rx': 'RF1=+5V, RF2=GND'},
    {'num': 4, 'name': '2→4', 'tx': 'RF1=GND, RF2=+5V', 'rx': 'RF1=GND, RF2=+5V'}
]

# ============= TEST CONDITIONS =============
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
    time.sleep(1)  # Wait for sweep
    
    data = vna.readline()
    if data:
        values = data.decode().strip().split(',')
        print(f" ✅ {len(values)} points")
        return values
    else:
        print(" ❌ No data")
        return None

def save_data(values, folder, path_num, path_name, run_num, rotation=None):
    """Save data to CSV file with run number and rotation"""
    if not values:
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename with metadata
    if rotation is not None:
        filename = f"{folder}/path{path_num}_{path_name}_run{run_num}_rot{rotation}_{timestamp}.csv"
    else:
        filename = f"{folder}/path{path_num}_{path_name}_run{run_num}_{timestamp}.csv"
    
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
    
    print(f"    💾 Saved: {os.path.basename(filename)} ({valid_points} points)")
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

def quick_quality_check(condition_folder):
    """Quick check to verify data was saved correctly"""
    import glob
    
    csv_files = glob.glob(f"{condition_folder}/*.csv")
    if len(csv_files) >= 4:
        print(f"    ✅ Quality check: {len(csv_files)} files saved")
        return True
    else:
        print(f"    ⚠️ Quality check: Only {len(csv_files)} files (expected 4 per run)")
        return False

def main():
    print_header("PULMO AI - COMPLETE PHANTOM SCANNING PROTOCOL")
    print(f"\n📊 SCANNING CONFIGURATION:")
    print(f"   • Repeat scans per condition: {SCANS_PER_CONDITION}")
    if USE_ROTATIONS:
        print(f"   • Rotations: {ROTATIONS}")
    else:
        print(f"   • Rotations: Disabled")
    
    # Create main folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"phantom_data_{timestamp}"
    os.makedirs(base_folder, exist_ok=True)
    print(f"\n📁 Main folder: {base_folder}")
    
    # Create metadata file
    with open(f"{base_folder}/scan_metadata.txt", 'w') as f:
        f.write("PULMO AI PHANTOM SCANNING METADATA\n")
        f.write("="*50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scans per condition: {SCANS_PER_CONDITION}\n")
        f.write(f"Rotations: {ROTATIONS if USE_ROTATIONS else 'None'}\n")
        f.write(f"Frequency range: {FREQ_START/1e9:.1f}-{FREQ_STOP/1e9:.1f} GHz\n")
        f.write(f"Points per sweep: {NUM_POINTS}\n\n")
        f.write("Conditions:\n")
        for cond in CONDITIONS:
            f.write(f"  - {cond['name']}: {cond['description']}\n")
    
    # Show switch guide
    show_switch_guide()
    
    # Connect to VNA
    vna = connect_vna()
    if not vna:
        return
    
    # Track statistics
    total_scans = 0
    successful_scans = 0
    
    # Run through each condition
    for cond_idx, condition in enumerate(CONDITIONS, 1):
        print_header(f"CONDITION {cond_idx}/{len(CONDITIONS)}: {condition['name']}")
        print(f"📝 {condition['description']}")
        print(f"\n👉 {condition['prompt']}")
        
        # Create condition folder
        cond_folder = f"{base_folder}/{condition['name']}"
        os.makedirs(cond_folder, exist_ok=True)
        
        # Determine rotation list
        rotations_to_use = ROTATIONS if USE_ROTATIONS else [None]
        
        for rotation in rotations_to_use:
            for run_num in range(1, SCANS_PER_CONDITION + 1):
                if rotation is not None:
                    print(f"\n🔄 Rotation: {rotation}° | Run {run_num}/{SCANS_PER_CONDITION}")
                else:
                    print(f"\n📸 Run {run_num}/{SCANS_PER_CONDITION}")
                
                total_scans += 1
                
                # Scan all 4 paths for this run
                run_successful = True
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
                        save_success = save_data(values, cond_folder, path['num'], 
                                                path['name'], run_num, rotation)
                        if not save_success:
                            run_successful = False
                    else:
                        run_successful = False
                
                if run_successful:
                    successful_scans += 1
                    # Quick quality check
                    quick_quality_check(cond_folder)
                else:
                    print(f"  ⚠️ Run {run_num} had errors")
        
        print(f"\n✅ Condition '{condition['name']}' complete!")
        
        # Ask if user wants to continue
        if cond_idx < len(CONDITIONS):
            response = input("\nContinue to next condition? (y/n): ")
            if response.lower() != 'y':
                print("\n⚠️ Stopping early. Partial data saved.")
                break
    
    vna.close()
    
    # Final summary
    print_header("SCANNING COMPLETE!")
    print(f"📁 All data saved in: {base_folder}")
    print(f"\n📊 SCAN SUMMARY:")
    print(f"   • Total scans attempted: {total_scans}")
    print(f"   • Successful scans: {successful_scans}")
    print(f"   • Success rate: {successful_scans/total_scans*100:.1f}%")
    
    print("\n📁 Folder structure:")
    for condition in CONDITIONS:
        path = f"{base_folder}/{condition['name']}"
        if os.path.exists(path):
            files = len([f for f in os.listdir(path) if f.endswith('.csv')])
            expected = SCANS_PER_CONDITION * 4 * (len(ROTATIONS) if USE_ROTATIONS else 1)
            print(f"  📁 {condition['name']}: {files}/{expected} CSV files")
    
    print("\n🔍 NEXT STEP: Run analysis script")
    print(f"python analyze_phantom_data.py {base_folder}")
    
    print("\n💡 TIP: For science fair, mention:")
    print(f"   • Collected {SCANS_PER_CONDITION} repeat scans per condition")
    if USE_ROTATIONS:
        print(f"   • Tested {len(ROTATIONS)} different orientations")
    print(f"   • Total measurements: {successful_scans * 4} individual S21 traces")

if __name__ == "__main__":
    main()
