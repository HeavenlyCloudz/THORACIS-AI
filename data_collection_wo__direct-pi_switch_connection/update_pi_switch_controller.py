# pi_switch_controller.py
"""
Simple RF Switch Controller for Raspberry Pi
Run this once to set up GPIO, then it waits for commands via SSH
"""
import RPi.GPIO as GPIO
import sys
import time

# GPIO pin configuration
PINS = {
    'switch1_rf1': 17,
    'switch1_rf2': 27,
    'switch2_rf1': 18,
    'switch2_rf2': 22
}

# Path configurations
PATHS = {
    1: {'rf1': [1, 0], 'rf2': [1, 0], 'name': '1→3'},
    2: {'rf1': [1, 0], 'rf2': [0, 1], 'name': '1→4'},
    3: {'rf1': [0, 1], 'rf2': [1, 0], 'name': '2→3'},
    4: {'rf1': [0, 1], 'rf2': [0, 1], 'name': '2→4'}
}

def setup_gpio():
    """Initialize GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    for pin in PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    print("✅ GPIO initialized", file=sys.stderr)

def set_path(path_num):
    """Set RF switches to specified path"""
    if path_num not in PATHS:
        print(f"❌ Invalid path: {path_num}", file=sys.stderr)
        return False
    
    config = PATHS[path_num]
    
    # Set switch 1 (TX) - pins 17 and 27
    GPIO.output(PINS['switch1_rf1'], config['rf1'][0])
    GPIO.output(PINS['switch1_rf2'], config['rf1'][1])
    
    # Set switch 2 (RX) - pins 18 and 22
    GPIO.output(PINS['switch2_rf1'], config['rf2'][0])
    GPIO.output(PINS['switch2_rf2'], config['rf2'][1])
    
    print(f"✅ Path {path_num} set: {config['name']}", file=sys.stderr)
    return True

def cleanup():
    """Clean up GPIO"""
    GPIO.cleanup()
    print("GPIO cleaned up", file=sys.stderr)

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python pi_switch_controller.py <path_number>", file=sys.stderr)
        print("Or: python pi_switch_controller.py --test", file=sys.stderr)
        sys.exit(1)
    
    setup_gpio()
    
    try:
        if sys.argv[1] == '--test':
            # Test mode: cycle through paths
            print("Testing all paths...", file=sys.stderr)
            for path_num in range(1, 5):
                set_path(path_num)
                time.sleep(1)
            print("Test complete", file=sys.stderr)
        else:
            # Set to specified path
            path_num = int(sys.argv[1])
            set_path(path_num)
    finally:
        cleanup()
