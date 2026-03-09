import serial, time, csv, math, json
from datetime import datetime

PORT = 'COM3'
BAUDRATE = 115200
START_FREQ = 2000000000
STOP_FREQ = 3000000000
POINTS = 201

def send_command(ser, cmd):
    """Send command with delay"""
    ser.write((cmd + '\n').encode())
    time.sleep(0.5)

def collect_scan(scan_name, params):
    print(f"\n=== {scan_name} ===")
    print(f"Params: {params}")
    
    try:
        with serial.Serial(PORT, BAUDRATE, timeout=3) as ser:
            time.sleep(2)
            ser.reset_input_buffer()
            
            # Send scan command
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5"
            print(f"Command: {cmd}")
            send_command(ser, cmd)
            
            data_points = []
            lines_collected = 0
            
            print(f"Waiting for {POINTS} data points...")
            
            while lines_collected < POINTS:
                line = ser.readline().decode('ascii', errors='ignore').strip()
                
                # CRITICAL: Skip command echo and prompts
                if line and not line.startswith('ch>') and not line.startswith('scan'):
                    parts = line.split()
                    
                    # KEY FIX: Accept EITHER 3 or 4 columns
                    if len(parts) >= 3:
                        try:
                            # Try 3-column format first (freq, real, imag)
                            if len(parts) == 3:
                                freq_hz = float(parts[0])
                                s21_real = float(parts[1])  # Column 2 = real
                                s21_imag = float(parts[2])  # Column 3 = imag
                            # Or 4-column format (freq, ?, real, imag)
                            elif len(parts) >= 4:
                                freq_hz = float(parts[0])
                                s21_real = float(parts[2])  # Column 3 = real
                                s21_imag = float(parts[3])  # Column 4 = imag
                            
                            magnitude = (s21_real**2 + s21_imag**2)**0.5
                            magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                            
                            data_points.append([freq_hz, magnitude_db])
                            lines_collected += 1
                            
                            if lines_collected % 40 == 0:
                                print(f"  [{lines_collected}/{POINTS}] {freq_hz/1e9:.3f} GHz: {magnitude_db:.2f} dB")
                                
                        except ValueError as e:
                            print(f"  Parse error on line {lines_collected}: {e}")
                            print(f"  Line: {line}")
                            continue
                    else:
                        print(f"  Skipping line (wrong format): {line}")
            
            if lines_collected < POINTS:
                print(f"⚠️  Warning: Collected {lines_collected}/{POINTS} points")
                if lines_collected == 0:
                    print("   No valid data received. Check VNA output format.")
                    return None
            
            # Save data
            filename = f"{scan_name}.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Frequency_Hz', 'S21_dB'])
                writer.writerows(data_points)
            
            avg_s21 = sum(d[1] for d in data_points) / len(data_points) if data_points else 0
            
            # Save metadata
            metadata = {
                'scan_name': scan_name,
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'points_collected': lines_collected,
                'average_s21_db': avg_s21,
                'min_s21_db': min(d[1] for d in data_points) if data_points else 0,
                'max_s21_db': max(d[1] for d in data_points) if data_points else 0
            }
            with open(f"{scan_name}_meta.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved: {filename} ({lines_collected} points)")
            print(f"  Average S21: {avg_s21:.2f} dB")
            return avg_s21
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    experiments = [
        ("healthy_baseline", "healthy", {"depth_cm": 5.0, "tumor": "none"}),
        ("tumor_small_center", "tumor_small", {"depth_cm": 5.0, "tumor_size_cm": 1.0, "position": "center"}),
        ("tumor_medium_center", "tumor_medium", {"depth_cm": 5.0, "tumor_size_cm": 2.0, "position": "center"}),
        ("tumor_large_center", "tumor_large", {"depth_cm": 5.0, "tumor_size_cm": 3.0, "position": "center"}),
        ("tumor_medium_left", "tumor_medium", {"depth_cm": 5.0, "tumor_size_cm": 2.0, "position": "left"}),
        ("tumor_medium_right", "tumor_medium", {"depth_cm": 5.0, "tumor_size_cm": 2.0, "position": "right"}),
        ("tumor_deep", "tumor_medium", {"depth_cm": 7.5, "tumor_size_cm": 2.0, "position": "center"}),
        ("tumor_dual", "tumor_multiple", {"depth_cm": 5.0, "tumor_size_cm": "two_1cm", "position": "spaced_4cm"}),
        ("tumor_shallow", "tumor_medium", {"depth_cm": 2.5, "tumor_size_cm": 2.0, "position": "center"}),
        ("healthy_shallow", "healthy", {"depth_cm": 2.5, "tumor": "none"}),
    ]
    
    print("="*65)
    print("PULMO AI - SYSTEMATIC PHANTOM VARIATION STUDY")
    print("="*65)
    print(f"Port: {PORT}, Freq: {START_FREQ/1e9:.1f}-{STOP_FREQ/1e9:.1f} GHz")
    print("="*65)
    
    results = []
    
    for i, (filename, label, params) in enumerate(experiments, 1):
        print(f"\n{'='*40}")
        print(f"EXPERIMENT {i}/10: {filename}")
        print(f"{'='*40}")
        
        input("Press Enter to start scan...")
        
        avg_s21 = collect_scan(filename, params)
        
        if avg_s21 is not None:
            results.append({
                'filename': filename,
                'label': label,
                'avg_s21_db': avg_s21,
                **params
            })
        else:
            print(f"❌ Failed: {filename}")
            retry = input("Try again? (y/n): ").lower()
            if retry == 'y':
                avg_s21 = collect_scan(filename, params)
                if avg_s21 is not None:
                    results.append({
                        'filename': filename,
                        'label': label,
                        'avg_s21_db': avg_s21,
                        **params
                    })
    
    # Save summary
    if results:
        summary_file = 'experiment_summary.csv'
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print("\n" + "="*65)
        print("EXPERIMENT COMPLETE!")
        print(f"Successful: {len(results)}/10 scans")
        print(f"Summary: {summary_file}")
        print("\nResults:")
        print("-" * 50)
        for r in results:
            print(f"{r['filename']:25} {r['avg_s21_db']:7.2f} dB")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()