# pi_switch_controller_all_paths.py
"""
RF Switch Controller for Raspberry Pi - ALL 4 PATHS
Run this on Pi in one terminal window
It waits for you to type path numbers
"""
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
pins = [17, 27, 18, 22]
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

def set_path(path_num):
    """Set RF switches for the requested path"""
    if path_num == 1:      # 1→3 (both switches on Path A)
        GPIO.output(17, 1); GPIO.output(27, 0)  # Switch 1: Antenna 1
        GPIO.output(18, 1); GPIO.output(22, 0)  # Switch 2: Antenna 3
        print(f"Path 1 set: Antenna 1 → Antenna 3 (opposite)")
        
    elif path_num == 2:    # 1→4 (Switch1 A, Switch2 B)
        GPIO.output(17, 1); GPIO.output(27, 0)  # Switch 1: Antenna 1
        GPIO.output(18, 0); GPIO.output(22, 1)  # Switch 2: Antenna 4
        print(f"Path 2 set: Antenna 1 → Antenna 4 (diagonal)")
        
    elif path_num == 3:    # 2→3 (Switch1 B, Switch2 A)
        GPIO.output(17, 0); GPIO.output(27, 1)  # Switch 1: Antenna 2
        GPIO.output(18, 1); GPIO.output(22, 0)  # Switch 2: Antenna 3
        print(f"Path 3 set: Antenna 2 → Antenna 3 (diagonal)")
        
    elif path_num == 4:    # 2→4 (both switches on Path B)
        GPIO.output(17, 0); GPIO.output(27, 1)  # Switch 1: Antenna 2
        GPIO.output(18, 0); GPIO.output(22, 1)  # Switch 2: Antenna 4
        print(f"Path 4 set: Antenna 2 → Antenna 4 (opposite)")
        
    else:
        print(f"Invalid path number: {path_num}")

print("="*50)
print("PULMO AI - RF SWITCH CONTROLLER (ALL 4 PATHS)")
print("="*50)
print("Enter path number (1-4) when prompted by computer")
print("  Path 1: Antenna 1 → Antenna 3 (opposite)")
print("  Path 2: Antenna 1 → Antenna 4 (diagonal)")
print("  Path 3: Antenna 2 → Antenna 3 (diagonal)")
print("  Path 4: Antenna 2 → Antenna 4 (opposite)")
print("Type 'q' to quit\n")

try:
    while True:
        cmd = input("Path: ")
        if cmd.lower() == 'q':
            break
        try:
            path = int(cmd)
            if 1 <= path <= 4:
                set_path(path)
                print("✅ Ready for capture on computer\n")
            else:
                print("Enter 1-4\n")
        except:
            print("Invalid input\n")
finally:
    GPIO.cleanup()
    print("GPIO cleaned up")