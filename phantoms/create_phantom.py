"""
PULMO AI - Phantom Fabrication Logger
Run script each time I make a phantom.
It saves a CSV log and prints a summary for my logbook.
"""

import datetime
import csv
import os
from pathlib import Path

# === CONFIGURATION ===
LOG_FILE = "pulmo_phantom_log.csv"
NOTES_FILE = "pulmo_phantom_notes.txt"

# === INITIALIZE LOG FILE IF IT DOESN'T EXIST ===
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Phantom ID",
            "Date",
            "Type (Healthy/Tumor)",
            "Water (mL)",
            "Agar (g)",
            "Salt (g)",
            "Sugar/Glycerol (g)",
            "Container Shape",
            "Container Diameter (cm)",
            "Container Height (cm)",
            "Approx Volume (mL)",
            "Notes"
        ])

# === INPUT PHANTOM DETAILS ===
print("\n=== PULMO AI PHANTOM LOGGER ===\n")

phantom_id = input("Phantom ID (e.g., H001, T001): ").strip()
phantom_type = input("Type (Healthy / Tumor): ").strip().capitalize()
water = float(input("Water volume (mL): "))
agar = float(input("Agar powder (g): "))
salt = float(input("Salt (g): "))

sugar = 0.0
if phantom_type == "Tumor":
    sugar_input = input("Sugar/Glycerol (g) [press Enter if 0]: ")
    sugar = float(sugar_input) if sugar_input.strip() else 0.0

shape = input("Container shape (cylinder/rectangle): ").strip().lower()
diameter = float(input("Diameter (cm) [if rectangle, enter width]: "))
height = float(input("Container height (cm): "))

# Estimate volume
if shape == "cylinder":
    import math
    volume = math.pi * (diameter/2)**2 * height
    volume_desc = f"π × ({diameter/2})² × {height} = {volume:.0f} mL"
else:  # rectangle approximation
    length = float(input("Length (cm) [if cylinder, press Enter]: ") or diameter)
    width = diameter
    volume = length * width * height
    volume_desc = f"{length} × {width} × {height} = {volume:.0f} mL"

notes = input("Any notes? (color, inclusions, etc.): ").strip()

# === WRITE TO CSV ===
with open(LOG_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        phantom_id,
        datetime.date.today().isoformat(),
        phantom_type,
        water,
        agar,
        salt,
        sugar,
        shape,
        diameter,
        height,
        f"{volume:.0f}",
        notes
    ])

# === GENERATE SUMMARY FOR NOTEBOOK ===
summary = f"""
========================================
PULMO AI PHANTOM FABRICATION RECORD
========================================
Phantom ID:       {phantom_id}
Date:             {datetime.date.today().isoformat()}
Type:             {phantom_type}

INGREDIENTS:
- Water:          {water} mL
- Agar powder:    {agar} g
- Salt:           {salt} g
- Sugar/Glycerol: {sugar} g

CONTAINER:
- Shape:          {shape}
- Dimensions:     {diameter} cm diameter/width × {height} cm height
- Est. volume:    {volume:.0f} mL ({volume_desc})

NOTES: {notes}

DIELECTRIC ESTIMATES (approx):
- Conductivity (σ): ~{200 * salt / water:.2f} S/m
- Relative permittivity (εᵣ): ~50-55 (healthy) or 60-70 (tumor) depending on sugar

========================================
"""
print(summary)

# Append to notes file
with open(NOTES_FILE, 'a') as f:
    f.write(summary)
    f.write("\n")

print(f"\n✅ Log saved to {LOG_FILE}")
print(f"✅ Summary appended to {NOTES_FILE}")
