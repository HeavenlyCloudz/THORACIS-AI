# update_pi_switch_controller.py - UPDATED
"""
RF Switch Controller for Raspberry Pi - OPTIMIZED FOR PATHS 1 & 2 ONLY
Only configures paths that actually work (1→3 and 2→4)
"""
import RPi.GPIO as GPIO

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
        print(f"Path 1 set: Antenna 1 → Antenna 3 (aligned, opposite)")
        
    elif path_num == 2:    # 2→4 (both switches on Path B)
        GPIO.output(17, 0); GPIO.output(27, 1)  # Switch 1: Antenna 2
        GPIO.output(18, 0); GPIO.output(22, 1)  # Switch 2: Antenna 4
        print(f"Path 2 set: Antenna 2 → Antenna 4 (aligned, opposite)")
        
    else:
        print(f"⚠️ Path {path_num} is not optimized (use 1 or 2 only)")
        print("   Paths 3 and 4 have poor alignment - skipping")

print("="*50)
print("PULMO AI - RF SWITCH CONTROLLER (OPTIMIZED)")
print("="*50)
print("Only Paths 1 and 2 are used (aligned opposite pairs)")
print("Enter path number (1 or 2) when prompted")
print("Type 'q' to quit\n")

try:
    while True:
        cmd = input("Path: ")
        if cmd.lower() == 'q':
            break
        try:
            path = int(cmd)
            if path == 1 or path == 2:
                set_path(path)
                print("✅ Ready for capture\n")
            else:
                print(f"Enter 1 or 2 (Path {path} is not optimized)\n")
        except:
            print("Invalid input\n")
finally:
    GPIO.cleanup()
    print("GPIO cleaned up")
