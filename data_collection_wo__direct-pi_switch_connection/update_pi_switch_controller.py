# update_pi_switch_controller.py
"""
Simple RF Switch Controller for Raspberry Pi
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
    if path_num == 1:      # 1→3
        GPIO.output(17, 1); GPIO.output(27, 0)
        GPIO.output(18, 1); GPIO.output(22, 0)
    elif path_num == 2:    # 1→4
        GPIO.output(17, 1); GPIO.output(27, 0)
        GPIO.output(18, 0); GPIO.output(22, 1)
    elif path_num == 3:    # 2→3
        GPIO.output(17, 0); GPIO.output(27, 1)
        GPIO.output(18, 1); GPIO.output(22, 0)
    elif path_num == 4:    # 2→4
        GPIO.output(17, 0); GPIO.output(27, 1)
        GPIO.output(18, 0); GPIO.output(22, 1)
    print(f"Path {path_num} set - {'1→3' if path_num==1 else '1→4' if path_num==2 else '2→3' if path_num==3 else '2→4'}")

print("="*50)
print("PULMO AI - RF SWITCH CONTROLLER")
print("="*50)
print("Enter path number (1-4) when prompted by computer")
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
                print("Enter 1-4")
        except:
            print("Invalid input")
finally:
    GPIO.cleanup()
    print("GPIO cleaned up")
