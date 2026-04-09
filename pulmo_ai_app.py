#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULMO AI - Operation Oracle: Democratized Lung Screening System
- Phase-free magnitude-only processing
- Multi-angle averaging (maintains 840-dim features)
- Clinical assessment questionnaire
- Health Passport for longitudinal patient records
- Tumor localization with bounding boxes
"""

import os

# Force OpenGL ES - must be set before any Qt imports
if 'QT_OPENGL' not in os.environ:
    os.environ['QT_OPENGL'] = 'es2'
if 'QT_QPA_EGLFS_HIDECURSOR' not in os.environ:
    os.environ['QT_QPA_EGLFS_HIDECURSOR'] = '1'

import sys
import time
import json
import threading
import serial
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import math
import traceback
import csv

# Qt
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QProgressBar, QMessageBox, QTabWidget, QTextEdit,
    QFrame, QScrollArea, QGroupBox, QRadioButton, QButtonGroup,
    QCheckBox, QComboBox, QFileDialog, QInputDialog
)

# Hardware
import RPi.GPIO as GPIO
import sounddevice as sd

# ML
import tflite_runtime.interpreter as tflite

# For reconstruction
from scipy.ndimage import gaussian_filter

# For resampling
import scipy.signal

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path.home() / "pulmo_ai_app" / "models"
YAMNET_PATH = MODEL_DIR / "yamnet_working.tflite"
AUDIO_MODEL_PATH = MODEL_DIR / "lung_audio.tflite"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_xgboost_model.pkl"
FUSION_SCALER_PATH = MODEL_DIR / "fusion_scaler.pkl"

# Data directories
DATA_DIR = Path.home() / "pulmo_ai_app" / "scans"
BASELINE_DIR = DATA_DIR / "baseline"
PATIENT_DIR = DATA_DIR / "patient"
MULTI_ANGLE_DIR = DATA_DIR / "multi_angle"

# VNA Serial Settings
VNA_PORT = '/dev/ttyACM0'
BAUDRATE = 115200
START_FREQ = 2000000000
STOP_FREQ = 3000000000
POINTS = 201

# GPIO pins
SWITCH1_A = 17
SWITCH1_B = 27
SWITCH2_A = 18
SWITCH2_B = 22

PATHS = {
    1: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 1, SWITCH2_B: 0, 'name': '1->3', 'desc': 'Left to Bottom'},
    2: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 0, SWITCH2_B: 1, 'name': '1->4', 'desc': 'Left to Top'},
    3: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 1, SWITCH2_B: 0, 'name': '2->3', 'desc': 'Right to Bottom'},
    4: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 0, SWITCH2_B: 1, 'name': '2->4', 'desc': 'Right to Top'},
}

# Antenna positions for reconstruction (x,y) in mm
ANTENNA_POSITIONS = {
    1: (-75, 0),
    2: (75, 0),
    3: (0, -75),
    4: (0, 75),
}

PATH_TO_ANTENNA_PAIR = {
    1: (1, 3),
    2: (1, 4),
    3: (2, 3),
    4: (2, 4),
}

# Multi-angle scanning configuration
ROTATION_ANGLES = [0, 120, 240]

# Audio settings
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
EXPECTED_AUDIO_SAMPLES = SAMPLE_RATE * RECORD_SECONDS

# Feature dimensions - CRITICAL: Must match training!
N_FREQ_POINTS = POINTS  # 201
N_PATHS = 4
N_FREQ_FEATURES = N_PATHS * N_FREQ_POINTS  # 804
N_TIME_FEATURES_PER_PATH = 9
N_TIME_FEATURES = N_PATHS * N_TIME_FEATURES_PER_PATH  # 36
TOTAL_FEATURES = N_FREQ_FEATURES + N_TIME_FEATURES  # 840

# Model class order from training
MODEL_CLASSES = ['bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

# Educational content
EDUCATIONAL_CONTENT = {
    'asthma': {
        'description': 'Asthma causes airway inflammation and narrowing, leading to wheezing and breathing difficulties.',
        'clinical_signs': 'Wheezing, chest tightness, shortness of breath, coughing, especially at night or early morning.',
        'recommendations': 'Use prescribed inhalers, avoid triggers, create an asthma action plan.'
    },
    'copd': {
        'description': 'Chronic Obstructive Pulmonary Disease includes emphysema and chronic bronchitis, causing airflow blockage.',
        'clinical_signs': 'Chronic cough, sputum production, shortness of breath during daily activities, frequent respiratory infections.',
        'recommendations': 'Smoking cessation, pulmonary rehabilitation, oxygen therapy if needed, regular check-ups.'
    },
    'pneumonia': {
        'description': 'Pneumonia is an infection that inflames air sacs in one or both lungs, causing them to fill with fluid.',
        'clinical_signs': 'Cough with phlegm, fever, chills, shortness of breath, chest pain during breathing or coughing.',
        'recommendations': 'Antibiotics for bacterial cases, rest, hydration, follow-up chest X-ray, monitor oxygen levels.'
    },
    'healthy': {
        'description': 'Normal lung function with clear airways and effective gas exchange.',
        'clinical_signs': 'No persistent cough, normal breathing patterns, ability to perform daily activities without breathlessness.',
        'recommendations': 'Maintain healthy lifestyle, avoid smoking, regular exercise, preventive care.'
    },
    'bronchial': {
        'description': 'Bronchial issues affect the main airways to the lungs, causing inflammation and mucus production.',
        'clinical_signs': 'Persistent cough often with mucus, fatigue, chest discomfort, mild fever.',
        'recommendations': 'Rest, hydration, avoid irritants, seek medical care if symptoms persist.'
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def db_to_linear(db):
    return 10 ** (db / 10)

def linear_to_db(linear):
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def add_time_domain_features(X):
    """
    Add time-domain features via IFFT using magnitude-only data.
    This matches your training pipeline exactly.
    
    Input: X shape (n_samples, 804) - magnitude values in dB
    Output: X_augmented shape (n_samples, 840) - 804 freq + 36 time features
    """
    n_samples, n_features = X.shape
    n_freq = n_features  # 804 features = 4 paths * 201 freq points
    n_paths = 4
    freq_per_path = n_freq // n_paths  # 201
    
    time_features = []
    
    for sample in X:
        sample_time_features = []
        for path in range(n_paths):
            # Extract this path's frequency response
            start_idx = path * freq_per_path
            end_idx = (path + 1) * freq_per_path
            freq_response = sample[start_idx:end_idx]
            
            # Convert from dB to linear magnitude
            freq_response_linear = db_to_linear(freq_response)
            
            # IFFT to get time domain (magnitude-only, but consistent with training)
            time_response = np.fft.ifft(freq_response_linear)
            time_magnitude = np.abs(time_response)
            
            # Extract time-domain features
            sample_time_features.extend([
                np.max(time_magnitude),          # Peak in time domain
                np.argmax(time_magnitude),       # Location of peak
                np.mean(time_magnitude),         # Average energy
                np.std(time_magnitude),          # Variation
                np.percentile(time_magnitude, 90),  # 90th percentile
                np.percentile(time_magnitude, 10),  # 10th percentile
                np.sum(time_magnitude),          # Total energy
                np.max(time_magnitude) - np.min(time_magnitude),  # Range
                np.sum(np.square(time_magnitude)),  # Energy
            ])
        
        time_features.append(sample_time_features)
    
    time_features = np.array(time_features)
    X_augmented = np.concatenate([X, time_features], axis=1)
    
    return X_augmented

# =============================================================================
# HEALTH PASSPORT - Patient Health Records
# =============================================================================

class HealthPassportWidget(QWidget):
    """Store and display patient health history across scans"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.records_file = DATA_DIR / "health_passport.json"
        self.patient_records = self._load_records()
        self.current_patient_id = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Health Passport")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #0277bd;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Your Personal Lung Health Record")
        subtitle.setStyleSheet("font-size: 11px; color: #666;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # Patient selector / new patient
        patient_row = QHBoxLayout()
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(150)
        self.patient_combo.currentTextChanged.connect(self._on_patient_selected)
        patient_row.addWidget(QLabel("Patient:"))
        patient_row.addWidget(self.patient_combo)
        
        self.new_patient_btn = QPushButton("New Patient")
        self.new_patient_btn.setStyleSheet("""
            QPushButton {
                background: #4fc3f7;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.new_patient_btn.clicked.connect(self._create_new_patient)
        patient_row.addWidget(self.new_patient_btn)
        
        layout.addLayout(patient_row)
        
        # Summary card for most recent scan
        summary_group = QGroupBox("Most Recent Assessment")
        summary_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        summary_layout = QVBoxLayout(summary_group)
        
        self.recent_date_label = QLabel("No scans recorded")
        self.recent_date_label.setStyleSheet("font-size: 12px; color: #555;")
        summary_layout.addWidget(self.recent_date_label)
        
        self.recent_dx_label = QLabel("")
        self.recent_dx_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        summary_layout.addWidget(self.recent_dx_label)
        
        self.recent_confidence_label = QLabel("")
        summary_layout.addWidget(self.recent_confidence_label)
        
        self.recent_trend_label = QLabel("")
        self.recent_trend_label.setStyleSheet("font-size: 11px;")
        summary_layout.addWidget(self.recent_trend_label)
        
        layout.addWidget(summary_group)
        
        # Historical trends
        trends_group = QGroupBox("Health Trends")
        trends_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        trends_layout = QVBoxLayout(trends_group)
        
        self.trends_text = QTextEdit()
        self.trends_text.setReadOnly(True)
        self.trends_text.setMaximumHeight(120)
        self.trends_text.setStyleSheet("font-size: 11px;")
        trends_layout.addWidget(self.trends_text)
        
        layout.addWidget(trends_group)
        
        # Scan history table
        history_group = QGroupBox("Scan History")
        history_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTextEdit()
        self.history_table.setReadOnly(True)
        self.history_table.setMaximumHeight(150)
        self.history_table.setStyleSheet("font-size: 10px; font-family: monospace;")
        history_layout.addWidget(self.history_table)
        
        layout.addWidget(history_group)
        
        # Export button
        export_btn = QPushButton("Export Health Report")
        export_btn.setMinimumHeight(35)
        export_btn.setStyleSheet("""
            QPushButton {
                background: #66bb6a;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background: #4caf50; }
        """)
        export_btn.clicked.connect(self._export_report)
        layout.addWidget(export_btn)
        
        # Refresh patient list
        self._refresh_patient_list()
    
    def _load_records(self):
        """Load health passport records from file"""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_records(self):
        """Save health passport records to file"""
        try:
            with open(self.records_file, 'w') as f:
                json.dump(self.patient_records, f, indent=2)
        except Exception as e:
            print(f"Error saving records: {e}")
    
    def _refresh_patient_list(self):
        """Update patient dropdown"""
        self.patient_combo.clear()
        patients = list(self.patient_records.keys())
        if patients:
            self.patient_combo.addItems(patients)
            self.patient_combo.setCurrentIndex(0)
        else:
            self.patient_combo.addItem("No patients")
    
    def _create_new_patient(self):
        """Create a new patient record"""
        name, ok = QInputDialog.getText(self, "New Patient", "Enter patient name:")
        if ok and name.strip():
            patient_id = name.strip()
            if patient_id not in self.patient_records:
                self.patient_records[patient_id] = {
                    'created': datetime.now().isoformat(),
                    'scans': []
                }
                self._save_records()
                self._refresh_patient_list()
                self.patient_combo.setCurrentText(patient_id)
                QMessageBox.information(self, "Success", f"Patient {patient_id} created")
    
    def _on_patient_selected(self, patient_name):
        """Load and display selected patient's records"""
        self.current_patient_id = patient_name
        if patient_name and patient_name in self.patient_records:
            self._display_patient_records(patient_name)
    
    def _display_patient_records(self, patient_name):
        """Display all records for a patient"""
        records = self.patient_records.get(patient_name, {})
        scans = records.get('scans', [])
        
        if not scans:
            self.recent_date_label.setText("No scans recorded")
            self.recent_dx_label.setText("")
            self.recent_confidence_label.setText("")
            self.recent_trend_label.setText("")
            self.trends_text.clear()
            self.history_table.clear()
            return
        
        # Most recent scan
        latest = scans[-1]
        scan_date = latest.get('date', 'Unknown')
        dx = latest.get('diagnosis', 'Unknown')
        confidence = latest.get('confidence', 0)
        
        self.recent_date_label.setText(f"Date: {scan_date[:16]}")
        self.recent_dx_label.setText(f"Diagnosis: {dx.upper()}")
        self.recent_confidence_label.setText(f"Confidence: {confidence:.1%}")
        
        # Show trend
        if len(scans) >= 2:
            prev = scans[-2]
            prev_dx = prev.get('diagnosis', 'Unknown')
            if prev_dx == dx:
                self.recent_trend_label.setText("Trend: Stable (same diagnosis)")
                self.recent_trend_label.setStyleSheet("font-size: 11px; color: #4caf50;")
            else:
                self.recent_trend_label.setText(f"Trend: Changed from {prev_dx.upper()} to {dx.upper()}")
                self.recent_trend_label.setStyleSheet("font-size: 11px; color: #ff9800;")
        else:
            self.recent_trend_label.setText("First scan - baseline established")
            self.recent_trend_label.setStyleSheet("font-size: 11px; color: #2196f3;")
        
        # Generate trend analysis
        self._update_trend_analysis(scans)
        
        # Update history table
        self._update_history_table(scans)
    
    def _update_trend_analysis(self, scans):
        """Analyze health trends over time"""
        if len(scans) < 2:
            self.trends_text.setText("Not enough data for trend analysis.\nComplete more scans to see trends.")
            return
        
        # Count diagnoses over time
        dx_counts = {}
        for scan in scans:
            dx = scan.get('diagnosis', 'Unknown')
            dx_counts[dx] = dx_counts.get(dx, 0) + 1
        
        # Calculate trend
        recent_3 = scans[-3:] if len(scans) >= 3 else scans
        recent_dxs = [s.get('diagnosis', 'Unknown') for s in recent_3]
        
        trend_text = ""
        
        # Check for improvement/deterioration
        if 'pneumonia' in recent_dxs and 'healthy' not in recent_dxs:
            trend_text += "- Current symptoms suggest active infection\n"
        elif 'healthy' in recent_dxs and 'pneumonia' in recent_dxs[:2]:
            trend_text += "- Improving condition detected\n"
        
        if 'asthma' in recent_dxs or 'copd' in recent_dxs:
            trend_text += "- Chronic respiratory pattern detected\n"
            trend_text += "- Regular monitoring recommended\n"
        
        if len(scans) >= 3:
            first_dx = scans[0].get('diagnosis', 'Unknown')
            last_dx = scans[-1].get('diagnosis', 'Unknown')
            if first_dx != 'healthy' and last_dx == 'healthy':
                trend_text += "- Significant improvement over time\n"
            elif first_dx == 'healthy' and last_dx != 'healthy':
                trend_text += "- Health decline detected - consult provider\n"
        
        # Add stability assessment
        unique_dxs = len(set([s.get('diagnosis', 'Unknown') for s in scans]))
        if unique_dxs == 1:
            trend_text += f"- Consistent {scans[0].get('diagnosis', 'Unknown').upper()} diagnosis across all scans\n"
        elif unique_dxs <= 2:
            trend_text += "- Relatively stable health pattern\n"
        else:
            trend_text += "- Variable health pattern - discuss with provider\n"
        
        # Add compliance note
        if len(scans) >= 2:
            try:
                first_date = datetime.fromisoformat(scans[0].get('date', ''))
                last_date = datetime.fromisoformat(scans[-1].get('date', ''))
                days_span = (last_date - first_date).days
                if days_span > 0:
                    scans_per_month = len(scans) / (days_span / 30)
                    if scans_per_month >= 1:
                        trend_text += f"- Good monitoring frequency ({scans_per_month:.1f} scans/month)\n"
                    else:
                        trend_text += "- Consider more frequent monitoring\n"
            except:
                pass
        
        self.trends_text.setText(trend_text)
    
    def _update_history_table(self, scans):
        """Display scan history in table format"""
        if not scans:
            return
        
        # Create table header
        table = "Date                 | Diagnosis      | Confidence | MW Result    | Audio Result\n"
        table += "-" * 80 + "\n"
        
        # Show last 10 scans (most recent first)
        for scan in reversed(scans[-10:]):
            date = scan.get('date', 'Unknown')[:16]
            dx = scan.get('diagnosis', 'Unknown')[:14]
            conf = f"{scan.get('confidence', 0):.0%}"
            mw = scan.get('microwave_result', 'Unknown')[:12]
            audio = scan.get('audio_result', 'Unknown')[:12]
            
            table += f"{date:20s} {dx:14s} {conf:10s} {mw:14s} {audio:12s}\n"
        
        self.history_table.setText(table)
    
    def add_scan_record(self, diagnosis, confidence, microwave_result, audio_result, audio_probs=None):
        """Add a new scan record for the current patient"""
        if not self.current_patient_id or self.current_patient_id not in self.patient_records:
            # Auto-create patient if none exists
            if not self.current_patient_id:
                self._create_new_patient()
                if not self.current_patient_id:
                    return
            if self.current_patient_id not in self.patient_records:
                self.patient_records[self.current_patient_id] = {
                    'created': datetime.now().isoformat(),
                    'scans': []
                }
        
        # Create scan record
        record = {
            'date': datetime.now().isoformat(),
            'diagnosis': diagnosis,
            'confidence': confidence,
            'microwave_result': microwave_result,
            'audio_result': audio_result,
            'audio_probs': audio_probs.tolist() if audio_probs is not None else None
        }
        
        # Add to records
        self.patient_records[self.current_patient_id]['scans'].append(record)
        self._save_records()
        
        # Refresh display
        self._display_patient_records(self.current_patient_id)
    
    def _export_report(self):
        """Export health report as CSV"""
        if not self.current_patient_id or self.current_patient_id not in self.patient_records:
            QMessageBox.warning(self, "No Patient", "No patient selected")
            return
        
        # Ask for save location
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Health Report", 
            f"{self.current_patient_id}_health_report.csv",
            "CSV Files (*.csv)"
        )
        
        if filepath:
            records = self.patient_records[self.current_patient_id]
            scans = records.get('scans', [])
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Diagnosis', 'Confidence', 'Microwave Result', 'Audio Result'])
                for scan in scans:
                    writer.writerow([
                        scan.get('date', ''),
                        scan.get('diagnosis', ''),
                        scan.get('confidence', ''),
                        scan.get('microwave_result', ''),
                        scan.get('audio_result', '')
                    ])
            
            QMessageBox.information(self, "Export Complete", f"Report saved to {filepath}")

# =============================================================================
# SYMPTOM TO CONDITION MAPPING
# =============================================================================

class SymptomToConditionMapper:
    """Maps patient-reported symptoms to condition probabilities"""
    
    def __init__(self):
        self.conditions = ['asthma', 'copd', 'pneumonia', 'bronchitis', 'healthy']
        
        self.primary_mapping = {
            'asthma': {
                'chest_sensation': ['Tightness (like a band squeezing)'],
                'lung_sounds': ['Wheezing (high-pitched whistling)'],
                'symptom_pattern': ['Episodic (comes and goes)', 'Triggered by exercise/allergens', 'Worse at night'],
                'cough_type': ['Dry cough (no mucus)']
            },
            'copd': {
                'chest_sensation': ['Heavy or clogged feeling'],
                'lung_sounds': ['Coarse crackles/Rhonchi (rattling/bubbling)'],
                'symptom_pattern': ['Constant (always present)', 'Worse when lying down'],
                'cough_type': ['Productive cough with clear mucus']
            },
            'pneumonia': {
                'chest_sensation': ['Sharp pain when breathing deeply'],
                'lung_sounds': ['Fine crackles (popping sounds)'],
                'systemic_symptoms': ['Fever', 'Chills'],
                'cough_type': ['Productive cough with yellow/green mucus']
            },
            'bronchitis': {
                'chest_sensation': ['Rattling sensation when breathing'],
                'lung_sounds': ['Coarse crackles/Rhonchi (rattling/bubbling)'],
                'systemic_symptoms': ['Sore throat', 'Runny nose'],
                'cough_type': ['Productive cough with clear mucus']
            },
            'healthy': {
                'chest_sensation': ['No chest discomfort'],
                'breathing_difficulty': ['No difficulty'],
                'cough_type': ['No cough'],
                'lung_sounds': ['Normal/No unusual sounds']
            }
        }
        
        self.secondary_mapping = {
            'asthma': {
                'breathing_difficulty': ['Mild - noticeable but not limiting', 'Moderate - limits some activities'],
                'systemic_symptoms': []
            },
            'copd': {
                'breathing_difficulty': ['Moderate - limits some activities', 'Severe - difficulty at rest'],
                'systemic_symptoms': ['Fatigue', 'Weight loss', 'Swollen ankles']
            },
            'pneumonia': {
                'breathing_difficulty': ['Severe - difficulty at rest'],
                'systemic_symptoms': ['Fatigue', 'Fever', 'Chills']
            },
            'bronchitis': {
                'breathing_difficulty': ['Mild - noticeable but not limiting'],
                'systemic_symptoms': ['Sore throat', 'Runny nose']
            },
            'healthy': {
                'systemic_symptoms': ['None of these']
            }
        }
        
        self.primary_weight = 3.0
        self.secondary_weight = 1.0
    
    def calculate_condition_probabilities(self, patient_symptoms):
        """Calculate probability for each condition based on reported symptoms"""
        scores = {condition: 0.0 for condition in self.conditions}
        
        for condition in self.conditions:
            score = 0.0
            
            # Check primary symptoms
            primary_criteria = self.primary_mapping.get(condition, {})
            for symptom_type, expected_values in primary_criteria.items():
                if symptom_type in patient_symptoms:
                    reported = patient_symptoms[symptom_type]
                    if isinstance(reported, list):
                        for r in reported:
                            if r in expected_values:
                                score += self.primary_weight
                    else:
                        if reported in expected_values:
                            score += self.primary_weight
            
            # Check secondary symptoms
            secondary_criteria = self.secondary_mapping.get(condition, {})
            for symptom_type, expected_values in secondary_criteria.items():
                if symptom_type in patient_symptoms:
                    reported = patient_symptoms[symptom_type]
                    if isinstance(reported, list):
                        for r in reported:
                            if r in expected_values:
                                score += self.secondary_weight
                    else:
                        if reported in expected_values:
                            score += self.secondary_weight
            
            scores[condition] = score
        
        # Convert scores to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {c: scores[c] / total_score for c in self.conditions}
        else:
            probabilities = {c: 0.0 for c in self.conditions}
            probabilities['healthy'] = 1.0
        
        return probabilities, scores

# =============================================================================
# ENHANCED CLINICAL DECISION SUPPORT
# =============================================================================

class EnhancedClinicalDecisionSupport:
    """Combines AI acoustic analysis with patient-reported symptoms"""
    
    def __init__(self):
        self.symptom_mapper = SymptomToConditionMapper()
        self.ai_weight = 0.6
        self.symptom_weight = 0.4
    
    def assess(self, ai_probs, patient_symptoms, environmental_quality="normal"):
        """Combine AI predictions with patient symptoms"""
        if environmental_quality == "poor":
            self.ai_weight = 0.3
            self.symptom_weight = 0.7
        elif environmental_quality == "good":
            self.ai_weight = 0.7
            self.symptom_weight = 0.3
        else:
            self.ai_weight = 0.6
            self.symptom_weight = 0.4
        
        # Get symptom-based probabilities
        symptom_probs, _ = self.symptom_mapper.calculate_condition_probabilities(patient_symptoms)
        
        # Combine AI and symptom probabilities
        final_probs = {}
        for condition in ['asthma', 'copd', 'pneumonia', 'bronchitis', 'healthy']:
            condition_idx = MODEL_CLASSES.index(condition) if condition in MODEL_CLASSES else 0
            ai_score = ai_probs[condition_idx]
            symptom_score = symptom_probs.get(condition, 0)
            final_probs[condition] = (self.ai_weight * ai_score) + (self.symptom_weight * symptom_score)
        
        # Normalize
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {c: p/total for c, p in final_probs.items()}
        
        # Get final diagnosis
        final_diagnosis = max(final_probs.items(), key=lambda x: x[1])
        final_confidence = final_diagnosis[1]
        
        # Generate explanation
        explanation = self._generate_explanation(
            final_diagnosis[0], final_probs, ai_probs, symptom_probs, patient_symptoms, environmental_quality
        )
        
        return final_diagnosis[0], final_confidence, explanation, final_probs
    
    def _generate_explanation(self, final_dx, final_probs, ai_probs, symptom_probs, symptoms, quality):
        """Generate comprehensive clinical explanation"""
        explanation = "CLINICAL DECISION ANALYSIS\n\n"
        explanation += "=" * 50 + "\n\n"
        
        if quality == "poor":
            explanation += "NOTE: Audio quality was poor. Diagnosis relies more heavily on patient-reported symptoms.\n\n"
        
        # AI Analysis Section
        ai_dx_idx = np.argmax(ai_probs)
        ai_dx = MODEL_CLASSES[ai_dx_idx]
        ai_conf = ai_probs[ai_dx_idx]
        
        explanation += "1. ACOUSTIC ANALYSIS (AI)\n"
        explanation += f"   Model detected: {ai_dx.upper()} with {ai_conf:.1%} confidence\n"
        explanation += "   Sound patterns identified:\n"
        
        sound_findings = {
            'asthma': "   - High-pitched wheezing during exhalation\n   - Prolonged expiratory phase\n",
            'copd': "   - Coarse crackles and rhonchi\n   - Reduced breath sounds\n",
            'pneumonia': "   - Fine crackles (popping sounds)\n   - Bronchial breath sounds\n",
            'bronchitis': "   - Rattling/coarse sounds\n   - Mucus-related noises\n",
            'healthy': "   - Clear breath sounds\n   - No abnormal adventitious sounds\n"
        }
        explanation += sound_findings.get(ai_dx, "   - Abnormal lung sounds detected\n")
        explanation += f"   Confidence in acoustic diagnosis: {ai_conf:.1%}\n\n"
        
        # Patient Symptoms Section
        explanation += "2. PATIENT-REPORTED SYMPTOMS\n"
        
        symptom_descriptions = {
            'chest_sensation': 'Chest sensation',
            'lung_sounds': 'Lung sounds noticed',
            'cough_type': 'Cough type',
            'breathing_difficulty': 'Breathing difficulty',
            'symptom_pattern': 'Symptom pattern',
            'systemic_symptoms': 'Additional symptoms'
        }
        
        for symptom_type, description in symptom_descriptions.items():
            if symptom_type in symptoms:
                value = symptoms[symptom_type]
                if isinstance(value, list) and value:
                    explanation += f"   - {description}: {', '.join(value)}\n"
                elif value:
                    explanation += f"   - {description}: {value}\n"
        
        # Symptom-based diagnosis
        symptom_dx = max(symptom_probs.items(), key=lambda x: x[1])
        explanation += f"\n   Symptoms most consistent with: {symptom_dx[0].upper()} ({symptom_dx[1]:.1%})\n\n"
        
        # Combined Analysis Section
        explanation += "3. COMBINED DIAGNOSIS\n\n"
        explanation += f"   Final Diagnosis: {final_dx.upper()}\n"
        explanation += f"   Overall Confidence: {final_confidence:.1%}\n\n"
        
        explanation += "   Decision weights:\n"
        explanation += f"   - AI Analysis: {self.ai_weight:.0%}\n"
        explanation += f"   - Patient Symptoms: {self.symptom_weight:.0%}\n\n"
        
        if final_dx == ai_dx and final_dx == symptom_dx[0]:
            explanation += f"   AGREEMENT: AI and symptoms both indicate {final_dx.upper()}\n"
            explanation += "   - High confidence in this diagnosis\n"
        elif final_dx == ai_dx:
            explanation += f"   AI analysis supports {final_dx.upper()}\n"
            explanation += "   - Your reported symptoms differ somewhat\n"
            explanation += "   - The acoustic patterns are the primary driver\n"
        elif final_dx == symptom_dx[0]:
            explanation += f"   Your symptoms strongly suggest {final_dx.upper()}\n"
            explanation += "   - The AI detected different patterns\n"
            explanation += "   - Consider clinical correlation\n"
        else:
            explanation += "   MIXED FINDINGS - consider clinical correlation\n"
            explanation += f"   - AI suggests {ai_dx.upper()}\n"
            explanation += f"   - Symptoms suggest {symptom_dx[0].upper()}\n"
        
        # Detailed probabilities table
        explanation += "\n4. DETAILED PROBABILITIES\n\n"
        explanation += f"   {'Condition':12s} {'AI':>8s} {'Symptoms':>10s} {'Combined':>10s}\n"
        explanation += f"   {'-'*12} {'-'*8} {'-'*10} {'-'*10}\n"
        
        for condition in ['asthma', 'copd', 'pneumonia', 'bronchitis', 'healthy']:
            condition_idx = MODEL_CLASSES.index(condition) if condition in MODEL_CLASSES else 0
            ai_pct = ai_probs[condition_idx] * 100
            sym_pct = symptom_probs.get(condition, 0) * 100
            final_pct = final_probs.get(condition, 0) * 100
            explanation += f"   {condition:12s} {ai_pct:7.1f}% {sym_pct:9.1f}% {final_pct:9.1f}%\n"
        
        return explanation

# =============================================================================
# CLINICAL ASSESSMENT WIDGET
# =============================================================================

class ClinicalAssessmentWidget(QFrame):
    assessment_complete = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            ClinicalAssessmentWidget {
                background-color: #fff8e1;
                border: 2px solid #ffb74d;
                border-radius: 15px;
                padding: 10px;
            }
        """)
        self.assessment_data = {}
        self._setup_ui()
        self.hide()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        title = QLabel("Clinical Assessment Questionnaire")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e65100;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Please answer these questions based on your symptoms")
        subtitle.setStyleSheet("font-size: 11px; color: #666;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: transparent;")
        
        content_widget = QWidget()
        self.form_layout = QVBoxLayout(content_widget)
        self.form_layout.setSpacing(15)
        
        # Question 1: Breathing Difficulty
        self._add_question("1. How would you describe your breathing difficulty?",
                          ["No difficulty", "Mild - noticeable but not limiting",
                           "Moderate - limits some activities", "Severe - difficulty at rest"],
                          "breathing_difficulty")
        
        # Question 2: Chest Sensation
        self._add_question("2. What chest sensations are you experiencing?",
                          ["No chest discomfort", "Tightness (like a band squeezing)",
                           "Sharp pain when breathing deeply", "Heavy or clogged feeling",
                           "Rattling sensation when breathing"],
                          "chest_sensation")
        
        # Question 3: Cough Type
        self._add_question("3. What best describes your cough?",
                          ["No cough", "Dry cough (no mucus)",
                           "Productive cough with clear mucus",
                           "Productive cough with yellow/green mucus",
                           "Persistent cough that won't go away"],
                          "cough_type")
        
        # Question 4: Sound Characteristic
        self._add_question("4. What lung sounds did you notice?",
                          ["Normal/No unusual sounds",
                           "Wheezing (high-pitched whistling)",
                           "Fine crackles (popping sounds)",
                           "Coarse crackles/Rhonchi (rattling/bubbling)",
                           "Rubbing/grating sound"],
                          "lung_sounds")
        
        # Question 5: Systemic Symptoms (multi-select)
        self._add_question("5. Do you have any of these symptoms? (Select all that apply)",
                          ["Fever", "Chills", "Fatigue", "Weight loss", "Swollen ankles",
                           "Sore throat", "Runny nose", "None of these"],
                          "systemic_symptoms",
                          multi_select=True)
        
        # Question 6: Symptom Pattern
        self._add_question("6. When do symptoms typically occur?",
                          ["Constant (always present)",
                           "Episodic (comes and goes)",
                           "Worse at night",
                           "Triggered by exercise/allergens",
                           "Worse when lying down"],
                          "symptom_pattern")
        
        self.submit_btn = QPushButton("SUBMIT ASSESSMENT")
        self.submit_btn.setMinimumHeight(40)
        self.submit_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                background: #ff9800;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px;
                margin-top: 10px;
            }
            QPushButton:hover { background: #f57c00; }
        """)
        self.submit_btn.clicked.connect(self._on_submit)
        self.form_layout.addWidget(self.submit_btn)
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
    
    def _add_question(self, question, options, key, multi_select=False):
        """Add a question to the form"""
        group = QGroupBox(question)
        group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        layout = QVBoxLayout(group)
        
        if multi_select:
            buttons = []
            for option in options:
                cb = QCheckBox(option)
                layout.addWidget(cb)
                buttons.append(cb)
            self.assessment_data[key] = buttons
        else:
            button_group = QButtonGroup()
            for i, option in enumerate(options):
                rb = QRadioButton(option)
                layout.addWidget(rb)
                button_group.addButton(rb, i)
            self.assessment_data[key] = button_group
        
        self.form_layout.addWidget(group)
    
    def _on_submit(self):
        """Collect and emit assessment results"""
        results = {}
        
        # Breathing difficulty
        if "breathing_difficulty" in self.assessment_data:
            bg = self.assessment_data["breathing_difficulty"]
            selected = bg.checkedButton()
            if selected:
                results["breathing_difficulty"] = selected.text()
        
        # Chest sensation
        if "chest_sensation" in self.assessment_data:
            bg = self.assessment_data["chest_sensation"]
            selected = bg.checkedButton()
            if selected:
                results["chest_sensation"] = selected.text()
        
        # Cough type
        if "cough_type" in self.assessment_data:
            bg = self.assessment_data["cough_type"]
            selected = bg.checkedButton()
            if selected:
                results["cough_type"] = selected.text()
        
        # Lung sounds
        if "lung_sounds" in self.assessment_data:
            bg = self.assessment_data["lung_sounds"]
            selected = bg.checkedButton()
            if selected:
                results["lung_sounds"] = selected.text()
        
        # Systemic symptoms (multi-select)
        if "systemic_symptoms" in self.assessment_data:
            selected = []
            for cb in self.assessment_data["systemic_symptoms"]:
                if cb.isChecked():
                    selected.append(cb.text())
            results["systemic_symptoms"] = selected
        
        # Symptom pattern
        if "symptom_pattern" in self.assessment_data:
            bg = self.assessment_data["symptom_pattern"]
            selected = bg.checkedButton()
            if selected:
                results["symptom_pattern"] = selected.text()
        
        self.assessment_complete.emit(results)
        self.hide()
    
    def show_assessment(self):
        """Show the assessment widget and reset selections"""
        for key, widget in self.assessment_data.items():
            if isinstance(widget, list):
                for cb in widget:
                    cb.setChecked(False)
            else:
                widget.setExclusive(False)
                for btn in widget.buttons():
                    btn.setAutoExclusive(False)
                    btn.setChecked(False)
                widget.setExclusive(True)
        
        self.show()
        self.raise_()

# =============================================================================
# TUMOR LOCALIZER CLASS
# =============================================================================

class TumorLocalizer:
    """Localize tumors using path attenuation analysis"""
    
    def __init__(self):
        self.antenna_positions = ANTENNA_POSITIONS
    
    def analyze_path_attenuation(self, s21_data, baseline_data):
        """Analyze which paths have abnormal attenuation"""
        path_attenuation = {}
        path_ratios = {}
        
        for path_num in [1, 2, 3, 4]:
            if path_num in s21_data and baseline_data and path_num in baseline_data:
                patient_avg = np.mean(s21_data[path_num])
                baseline_avg = np.mean(baseline_data[path_num])
                
                diff = baseline_avg - patient_avg
                path_attenuation[path_num] = diff
                path_ratios[path_num] = patient_avg / baseline_avg if baseline_avg != 0 else 1.0
        
        sorted_paths = sorted(path_attenuation.items(), key=lambda x: x[1], reverse=True)
        tumor_location = self._estimate_location_from_paths(sorted_paths)
        
        return {
            'path_attenuation': path_attenuation,
            'path_ratios': path_ratios,
            'most_affected_paths': sorted_paths[:2],
            'tumor_location': tumor_location,
            'confidence': self._calculate_confidence(sorted_paths, path_ratios)
        }
    
    def _estimate_location_from_paths(self, sorted_paths):
        """Estimate tumor location by intersecting most affected paths"""
        if len(sorted_paths) < 1:
            return {'x': 0, 'y': 0, 'description': 'Unable to localize', 'quadrant': 'unknown'}
        
        top_paths = [p[0] for p in sorted_paths[:2]]
        
        intersections = []
        for path_num in top_paths:
            if path_num in PATH_TO_ANTENNA_PAIR:
                tx, rx = PATH_TO_ANTENNA_PAIR[path_num]
                tx_pos = self.antenna_positions[tx]
                rx_pos = self.antenna_positions[rx]
                
                mid_x = (tx_pos[0] + rx_pos[0]) / 2
                mid_y = (tx_pos[1] + rx_pos[1]) / 2
                intersections.append((mid_x, mid_y))
        
        if len(intersections) >= 2:
            avg_x = np.mean([p[0] for p in intersections])
            avg_y = np.mean([p[1] for p in intersections])
        elif len(intersections) == 1:
            avg_x, avg_y = intersections[0]
        else:
            avg_x, avg_y = 0, 0
        
        quadrant = self._get_quadrant(avg_x, avg_y)
        
        return {
            'x': avg_x,
            'y': avg_y,
            'quadrant': quadrant,
            'description': f"Abnormal presence detected in {quadrant} region of the chest"
        }
    
    def _get_quadrant(self, x, y):
        if x > 0 and y > 0:
            return "upper right"
        elif x < 0 and y > 0:
            return "upper left"
        elif x > 0 and y < 0:
            return "lower right"
        elif x < 0 and y < 0:
            return "lower left"
        else:
            return "central"
    
    def _calculate_confidence(self, sorted_paths, path_ratios):
        if len(sorted_paths) < 2:
            return 0.3
        
        top1_atten = sorted_paths[0][1] if sorted_paths else 0
        top2_atten = sorted_paths[1][1] if len(sorted_paths) > 1 else 0
        
        all_atten = [a for _, a in sorted_paths]
        if len(all_atten) > 1 and np.std(all_atten) > 0:
            spread_ratio = (top1_atten - top2_atten) / (np.std(all_atten) + 1e-6)
        else:
            spread_ratio = 1
        
        confidence = min(0.95, 0.3 + spread_ratio * 0.3)
        return confidence
    
    def generate_bounding_box(self, tumor_location, image_width=350, image_height=350):
        """Generate bounding box coordinates for visualization"""
        px = int((tumor_location['x'] + 100) / 200 * image_width)
        py = int((tumor_location['y'] + 100) / 200 * image_height)
        
        box_size = 40
        x1 = max(0, px - box_size // 2)
        y1 = max(0, py - box_size // 2)
        x2 = min(image_width, px + box_size // 2)
        y2 = min(image_height, py + box_size // 2)
        
        return (x1, y1, x2, y2)

# =============================================================================
# RECONSTRUCTION WIDGET
# =============================================================================

class ReconstructionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 350)
        self.reconstruction_data = None
        self.tumor_location = None
        self.bounding_box = None
        self.localization_confidence = 0
        self.setStyleSheet("background-color: white; border: 2px solid #4fc3f7; border-radius: 10px;")
    
    def reconstruct_image(self, s21_data, frequencies, baseline_data=None):
        """Simple delay-and-sum reconstruction using magnitude data"""
        try:
            grid_size = 80
            x_grid = np.linspace(-100, 100, grid_size)
            y_grid = np.linspace(-100, 100, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            image = np.zeros((grid_size, grid_size))
            c = 3e8
            
            for path_num, s21 in s21_data.items():
                if path_num not in PATH_TO_ANTENNA_PAIR:
                    continue
                
                tx_ant, rx_ant = PATH_TO_ANTENNA_PAIR[path_num]
                tx_pos = ANTENNA_POSITIONS[tx_ant]
                rx_pos = ANTENNA_POSITIONS[rx_ant]
                
                # Convert dB to linear
                s21_linear = db_to_linear(s21)
                
                if baseline_data and path_num in baseline_data:
                    baseline_linear = db_to_linear(baseline_data[path_num])
                    s21_linear = s21_linear - baseline_linear
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        point = (X[i, j], Y[i, j])
                        
                        d_tx = np.sqrt((tx_pos[0] - point[0])**2 + (tx_pos[1] - point[1])**2)
                        d_rx = np.sqrt((rx_pos[0] - point[0])**2 + (rx_pos[1] - point[1])**2)
                        total_dist = (d_tx + d_rx) / 1000
                        
                        # Simple delay compensation (magnitude-based)
                        delay = total_dist / c
                        freq_idx = int(np.clip(delay * 1e9 / (STOP_FREQ/1e9) * POINTS, 0, POINTS-1))
                        
                        if freq_idx < len(s21_linear):
                            image[i, j] += s21_linear[freq_idx]
            
            image /= len(s21_data)
            image = gaussian_filter(image, sigma=2)
            
            # Normalize for display
            if image.max() > 0:
                image = np.clip(image, 0, np.percentile(image, 95))
                image = (image / image.max()) * 255
            
            self.reconstruction_data = image.astype(np.uint8)
            self.update()
            return self.reconstruction_data
            
        except Exception as e:
            print(f"Reconstruction error: {e}")
            traceback.print_exc()
            return None
    
    def set_tumor_localization(self, tumor_location, confidence, bounding_box):
        self.tumor_location = tumor_location
        self.localization_confidence = confidence
        self.bounding_box = bounding_box
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, QColor(255, 255, 255))
        
        if self.reconstruction_data is not None:
            img_height, img_width = self.reconstruction_data.shape
            scale_x = w / img_width
            scale_y = h / img_height
            
            for i in range(img_width):
                for j in range(img_height):
                    val = self.reconstruction_data[i, j]
                    r = int(val)
                    g = int(val * 0.5)
                    b = int(255 - val)
                    color = QColor(r, g, b)
                    x = int(i * scale_x)
                    y = int(j * scale_y)
                    painter.fillRect(x, y, int(scale_x) + 1, int(scale_y) + 1, color)
            
            if self.bounding_box and self.localization_confidence > 0.5:
                x1, y1, x2, y2 = self.bounding_box
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(center_x - 10, center_y, center_x + 10, center_y)
                painter.drawLine(center_x, center_y - 10, center_x, center_y + 10)
                
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                painter.drawRect(x1, y1 - 25, 120, 20)
                painter.drawText(x1 + 5, y1 - 10, f"Confidence: {self.localization_confidence:.0%}")
        
        else:
            painter.setPen(QPen(QColor(150, 150, 150), 2))
            painter.drawRect(10, 10, w - 20, h - 20)
            painter.drawText(w//2 - 100, h//2, "Reconstruction will appear here after scan")
        
        painter.setPen(QPen(QColor(0, 0, 255), 3))
        painter.setBrush(QBrush(QColor(0, 0, 255)))
        for ant, pos in ANTENNA_POSITIONS.items():
            x = int((pos[0] + 100) / 200 * w)
            y = int((pos[1] + 100) / 200 * h)
            painter.drawEllipse(x - 5, y - 5, 10, 10)
            painter.drawText(x - 15, y - 10, f"{ant}")
        
        painter.end()
    
    def clear(self):
        self.reconstruction_data = None
        self.tumor_location = None
        self.bounding_box = None
        self.localization_confidence = 0
        self.update()

# =============================================================================
# AUDIO PROCESSOR
# =============================================================================

class AudioProcessor(QThread):
    result_ready = Signal(np.ndarray)
    waveform_ready = Signal(np.ndarray)
    finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, record_seconds=3, device_id=None):
        super().__init__()
        self.record_seconds = record_seconds
        self.device_id = device_id
        self.yamnet_interpreter = None
        self.classifier_interpreter = None
        self._load_models()
    
    def _load_models(self):
        try:
            if not YAMNET_PATH.exists():
                self.error_occurred.emit(f"YAMNet model not found: {YAMNET_PATH}")
                return
            
            self.yamnet_interpreter = tflite.Interpreter(str(YAMNET_PATH))
            self.yamnet_interpreter.allocate_tensors()
            print("YAMNet TFLite loaded")
            
            if not AUDIO_MODEL_PATH.exists():
                self.error_occurred.emit(f"Classifier not found: {AUDIO_MODEL_PATH}")
                return
            
            self.classifier_interpreter = tflite.Interpreter(str(AUDIO_MODEL_PATH))
            self.classifier_interpreter.allocate_tensors()
            print("Classifier loaded")
            
        except Exception as e:
            self.error_occurred.emit(f"Model loading error: {e}")
            traceback.print_exc()
    
    def run(self):
        try:
            print(f"Recording {self.record_seconds} seconds...")
            
            sample_count = EXPECTED_AUDIO_SAMPLES
            
            try:
                recording = sd.rec(sample_count, samplerate=SAMPLE_RATE, channels=1,
                                   dtype='float32', device=self.device_id, blocking=True)
                sd.wait()
                audio = recording.flatten()
            except Exception as e:
                device_info = sd.query_devices(self.device_id) if self.device_id else sd.query_devices(sd.default.device[0])
                device_sr = int(device_info['default_samplerate'])
                recording = sd.rec(int(self.record_seconds * device_sr), samplerate=device_sr,
                                   channels=1, dtype='float32', device=self.device_id, blocking=True)
                sd.wait()
                audio = recording.flatten()
                new_length = int(len(audio) * SAMPLE_RATE / device_sr)
                audio = scipy.signal.resample(audio, new_length)
            
            audio_max = np.max(np.abs(audio))
            if audio_max > 0.01:
                audio = audio / audio_max
            else:
                self.error_occurred.emit("Low audio input detected")
                self.result_ready.emit(np.zeros(5, dtype=np.float32))
                self.finished.emit()
                return
            
            if len(audio) < EXPECTED_AUDIO_SAMPLES:
                audio = np.pad(audio, (0, EXPECTED_AUDIO_SAMPLES - len(audio)))
            elif len(audio) > EXPECTED_AUDIO_SAMPLES:
                audio = audio[:EXPECTED_AUDIO_SAMPLES]
            
            input_details = self.yamnet_interpreter.get_input_details()[0]
            self.yamnet_interpreter.set_tensor(input_details['index'], audio.astype(np.float32))
            self.yamnet_interpreter.invoke()
            
            output_details = self.yamnet_interpreter.get_output_details()[0]
            embeddings = self.yamnet_interpreter.get_tensor(output_details['index'])
            
            if len(embeddings.shape) == 2:
                pooled_embedding = embeddings[0]
            elif len(embeddings.shape) == 1:
                pooled_embedding = embeddings
            else:
                pooled_embedding = np.mean(embeddings, axis=0)
            
            input_data = pooled_embedding.reshape(1, -1).astype(np.float32)
            classifier_input = self.classifier_interpreter.get_input_details()[0]
            self.classifier_interpreter.set_tensor(classifier_input['index'], input_data)
            self.classifier_interpreter.invoke()
            
            classifier_output = self.classifier_interpreter.get_output_details()[0]
            probs = self.classifier_interpreter.get_tensor(classifier_output['index'])[0]
            
            self.result_ready.emit(probs.astype(np.float32))
            
            if len(audio) > 800:
                step = max(1, len(audio) // 800)
                self.waveform_ready.emit(audio[::step][:800])
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.result_ready.emit(np.zeros(5, dtype=np.float32))
        finally:
            self.finished.emit()

# =============================================================================
# EDUCATIONAL WIDGET (Fixed Scrolling)
# =============================================================================

class EducationalWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            EducationalWidget {
                background-color: #f5f5f5;
                border: 2px solid #4fc3f7;
                border-radius: 15px;
                padding: 10px;
            }
        """)
        self._setup_ui()
        self.hide()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.title_label = QLabel("Clinical Education")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0277bd;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background-color: #4fc3f7;")
        layout.addWidget(divider)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #4fc3f7;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setSpacing(15)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        self.condition_label = QLabel()
        self.condition_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #0277bd;")
        self.condition_label.setAlignment(Qt.AlignCenter)
        self.condition_label.setWordWrap(True)
        self.content_layout.addWidget(self.condition_label)
        
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("font-size: 13px; line-height: 1.5; padding: 5px;")
        self.desc_label.setAlignment(Qt.AlignTop)
        self.content_layout.addWidget(self.desc_label)
        
        signs_group = QGroupBox("Clinical Signs and Symptoms")
        signs_group.setStyleSheet("""
            QGroupBox { 
                font-size: 14px; 
                font-weight: bold; 
                border: 1px solid #ccc; 
                border-radius: 8px; 
                margin-top: 10px; 
                padding-top: 10px;
            }
        """)
        signs_layout = QVBoxLayout(signs_group)
        self.signs_label = QLabel()
        self.signs_label.setWordWrap(True)
        self.signs_label.setStyleSheet("font-size: 12px; padding: 5px;")
        self.signs_label.setAlignment(Qt.AlignTop)
        signs_layout.addWidget(self.signs_label)
        self.content_layout.addWidget(signs_group)
        
        rec_group = QGroupBox("Recommendations")
        rec_group.setStyleSheet("""
            QGroupBox { 
                font-size: 14px; 
                font-weight: bold; 
                border: 1px solid #ccc; 
                border-radius: 8px; 
                margin-top: 10px; 
                padding-top: 10px;
            }
        """)
        rec_layout = QVBoxLayout(rec_group)
        self.rec_label = QLabel()
        self.rec_label.setWordWrap(True)
        self.rec_label.setStyleSheet("font-size: 12px; padding: 5px;")
        self.rec_label.setAlignment(Qt.AlignTop)
        rec_layout.addWidget(self.rec_label)
        self.content_layout.addWidget(rec_group)
        
        literacy_note = QLabel(
            "Clinical literacy empowers patients to recognize symptoms early and seek appropriate care."
        )
        literacy_note.setWordWrap(True)
        literacy_note.setStyleSheet(
            "font-size: 11px; font-style: italic; color: #666; "
            "background-color: #e3f2fd; padding: 8px; border-radius: 8px;"
        )
        literacy_note.setAlignment(Qt.AlignTop)
        self.content_layout.addWidget(literacy_note)
        
        self.content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
    
    def show_condition(self, condition_name, confidence):
        condition = condition_name.lower()
        content = EDUCATIONAL_CONTENT.get(condition, EDUCATIONAL_CONTENT.get('healthy'))
        
        self.condition_label.setText(f"{condition_name.upper()} ({confidence:.1%} confidence)")
        self.desc_label.setText(content['description'])
        self.signs_label.setText(content['clinical_signs'])
        self.rec_label.setText(content['recommendations'])
        
        self.show()
        self.raise_()

# =============================================================================
# VNA CONTROLLER
# =============================================================================

class VNADirectController:
    def __init__(self, port=VNA_PORT, baudrate=BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.frequencies = None
        self.connect()
    
    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            print(f"VNA connected on {self.port}")
            return True
        except Exception as e:
            print(f"VNA connection failed: {e}")
            return False
    
    def capture_s21(self, progress_callback=None):
        """Capture S21 magnitude data"""
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return None
        
        try:
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5\r\n"
            self.serial_conn.write(cmd.encode())
            
            time.sleep(2.0)
            
            data_points = []
            lines_collected = 0
            timeout_start = time.time()
            max_timeout = 20
            
            while lines_collected < POINTS and (time.time() - timeout_start) < max_timeout:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
                    
                    if not line or line.startswith('ch>') or line.startswith('scan') or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            s21_real = float(parts[1])
                            s21_imag = float(parts[2])
                            
                            magnitude = math.sqrt(s21_real**2 + s21_imag**2)
                            magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                            
                            data_points.append(magnitude_db)
                            lines_collected += 1
                            
                            if progress_callback and lines_collected % 20 == 0:
                                progress_callback(lines_collected, POINTS)
                        except ValueError:
                            continue
                else:
                    time.sleep(0.05)
            
            if len(data_points) == POINTS:
                if self.frequencies is None:
                    self.frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, POINTS)
                return np.array(data_points)
            else:
                print(f"Only {len(data_points)}/{POINTS} points captured")
                return None
                
        except Exception as e:
            print(f"VNA capture error: {e}")
            return None
    
    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("VNA disconnected")

# =============================================================================
# RF SWITCH CONTROLLER
# =============================================================================

class RFSwitchController:
    def __init__(self):
        self._setup_gpio()
    
    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in [SWITCH1_A, SWITCH1_B, SWITCH2_A, SWITCH2_B]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)
        print("GPIO initialized for RF switches")
    
    def set_path(self, path_num):
        if path_num not in PATHS:
            raise ValueError(f"Invalid path: {path_num}")
        states = PATHS[path_num]
        for pin, state in states.items():
            if pin in [SWITCH1_A, SWITCH1_B, SWITCH2_A, SWITCH2_B]:
                GPIO.output(pin, state)
        time.sleep(0.1)
        print(f"Path {path_num} set: {PATHS[path_num]['desc']}")
    
    def cleanup(self):
        GPIO.cleanup()
        print("GPIO cleaned up")

# =============================================================================
# CSV DATA MANAGER
# =============================================================================

class CSVDataManager:
    def __init__(self):
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        PATIENT_DIR.mkdir(parents=True, exist_ok=True)
        MULTI_ANGLE_DIR.mkdir(parents=True, exist_ok=True)
        print("Data directories created")
    
    def save_scan(self, data, path_num, directory, frequencies=None, angle=0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"path{path_num}_angle{angle}_{timestamp}.csv"
        filepath = directory / filename
        
        if frequencies is None:
            frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, len(data))
        
        rows = []
        for i, (freq, s21) in enumerate(zip(frequencies, data)):
            rows.append([freq, s21])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_GHz', 'S21_dB'])
            writer.writerows(rows)
        
        return filepath
    
    def load_latest_from_directory(self, directory):
        data = {}
        for path_num in [1, 2, 3, 4]:
            files = list(directory.glob(f"path{path_num}_*.csv"))
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest)
                data[path_num] = df['S21_dB'].values
        return data if data else None
    
    def has_baseline(self):
        for path_num in [1, 2, 3, 4]:
            files = list(BASELINE_DIR.glob(f"path{path_num}_*.csv"))
            if files:
                return True
        return False
    
    def clear_all(self):
        for d in [BASELINE_DIR, PATIENT_DIR, MULTI_ANGLE_DIR]:
            if d.exists():
                for f in d.glob("*.csv"):
                    f.unlink()
        print("All data cleared")

# =============================================================================
# MICROWAVE SCANNER
# =============================================================================

class MicrowaveScanner:
    def __init__(self, vna_controller):
        self.vna = vna_controller
        self.switch = RFSwitchController()
        self.csv_manager = CSVDataManager()
        self.frequencies = None
        self._baseline_data = None
    
    def scan_all_paths(self, save_dir, angle=0, progress_callback=None):
        """Scan all paths and return magnitude data"""
        data = {}
        total_paths = len(PATHS)
        
        for idx, path_num in enumerate(PATHS.keys(), 1):
            if progress_callback:
                progress_callback(f"Setting Path {path_num} at {angle} deg", idx / total_paths)
            
            self.switch.set_path(path_num)
            time.sleep(0.2)
            
            if progress_callback:
                progress_callback(f"Capturing Path {path_num}", idx / total_paths)
            
            s21_data = self.vna.capture_s21()
            if s21_data is None:
                raise RuntimeError(f"Failed to capture path {path_num}")
            
            data[path_num] = s21_data
            
            if self.frequencies is None:
                self.frequencies = self.vna.frequencies
            
            self.csv_manager.save_scan(s21_data, path_num, save_dir, self.frequencies, angle)
            
            if progress_callback:
                progress_callback(f"Path {path_num} complete", idx / total_paths)
        
        return data
    
    def set_baseline(self, baseline_data):
        self._baseline_data = baseline_data.copy()
        print(f"Baseline stored for {len(self._baseline_data)} paths")
    
    def load_baseline(self):
        if self._baseline_data is not None:
            return self._baseline_data
        return self.csv_manager.load_latest_from_directory(BASELINE_DIR)
    
    def has_baseline(self):
        return self.csv_manager.has_baseline() or (self._baseline_data is not None)
    
    def extract_features(self, s21_data):
        """
        Extract 840-dim feature vector from magnitude data.
        This matches your training pipeline exactly.
        """
        # Step 1: Concatenate frequency features (4 paths * 201 points = 804)
        freq_features = np.array([s21_data[p] for p in [1, 2, 3, 4]]).reshape(1, -1)
        
        # Step 2: Add time-domain features (36 features)
        augmented_features = add_time_domain_features(freq_features)
        
        return augmented_features[0]
    
    def combine_rotation_features(self, rotation_data):
        """
        Combine features from multiple rotations by AVERAGING.
        This maintains the 840-dim feature vector your model expects.
        
        Args:
            rotation_data: dict of {angle: {path_num: s21_array}}
        
        Returns:
            840-dim feature vector (averaged across rotations)
        """
        all_freq_features = []
        
        for angle, data in rotation_data.items():
            # Extract frequency features for this rotation
            freq_features = np.array([data[p] for p in [1, 2, 3, 4]]).reshape(1, -1)
            all_freq_features.append(freq_features)
        
        # Average across all rotations (preserves 804 dimensions)
        avg_freq_features = np.mean(all_freq_features, axis=0)  # Shape: (1, 804)
        
        # Add time-domain features (36 features)
        augmented_features = add_time_domain_features(avg_freq_features)
        
        return augmented_features[0]
    
    def cleanup(self):
        self.switch.cleanup()
        self.vna.close()

# =============================================================================
# FUSION CLASSIFIER
# =============================================================================

class FusionClassifier:
    def __init__(self):
        if not FUSION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Fusion model not found: {FUSION_MODEL_PATH}")
        if not FUSION_SCALER_PATH.exists():
            raise FileNotFoundError(f"Fusion scaler not found: {FUSION_SCALER_PATH}")
        
        with open(FUSION_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(FUSION_SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Fusion model loaded")
    
    def predict(self, mw_features, audio_probs):
        """Combine 840-dim microwave features + 5-dim audio probs = 845-dim"""
        fusion_vec = np.concatenate([mw_features, audio_probs]).reshape(1, -1)
        scaled = self.scaler.transform(fusion_vec)
        pred = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        return pred, np.max(proba)

# =============================================================================
# MAIN GUI APPLICATION
# =============================================================================

class PulmoAIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PULMO AI: Operation Oracle - Lung Screening System")
        
        self.vna = VNADirectController()
        self.scanner = MicrowaveScanner(self.vna)
        
        self.audio_device_id = None
        self._setup_audio_device()
        
        self.educational_widget = EducationalWidget(self)
        self.reconstruction_widget = ReconstructionWidget(self)
        self.clinical_assessment = ClinicalAssessmentWidget(self)
        self.clinical_assessment.assessment_complete.connect(self._on_clinical_assessment)
        self.clinical_decision_support = EnhancedClinicalDecisionSupport()
        
        # Health Passport
        self.health_passport = HealthPassportWidget()
        
        try:
            self.fusion = FusionClassifier()
        except Exception as e:
            print(f"Fusion not loaded: {e}")
            self.fusion = None
        
        self.current_mw_features = None
        self.current_audio_probs = None
        self.last_ai_probs = None
        self.current_s21_data = None
        self.baseline_data = None
        self.rotation_scans = {}
        
        self._setup_ui()
        self.showFullScreen()
    
    def _setup_audio_device(self):
        try:
            devices = sd.query_devices()
            print("Available audio devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device['name']} - {device['max_input_channels']} in")
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and ('USB' in device['name'] or 'Mic' in device['name']):
                    self.audio_device_id = i
                    print(f"Selected audio input device: {i} - {device['name']}")
                    break
            
            if self.audio_device_id is None:
                print("Using default audio input device")
        except Exception as e:
            print(f"Error detecting audio devices: {e}")
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left panel (70%)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        title = QLabel("PULMO AI: Operation Oracle")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #0277bd;
            background-color: #e1f5fe;
            padding: 15px;
            border-radius: 20px;
            margin-bottom: 10px;
        """)
        left_layout.addWidget(title)
        
        subtitle = QLabel("Democratized Lung Screening | Explainable AI | Clinical Education")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #555; margin-bottom: 10px;")
        left_layout.addWidget(subtitle)
        
        vna_status = "Connected" if self.vna.serial_conn else "Disconnected"
        audio_status = "USB" if self.audio_device_id is not None else "Default"
        self.status_bar = QLabel(f"VNA: {vna_status} | Audio: {audio_status} | Multi-Angle: {len(ROTATION_ANGLES)} positions")
        self.status_bar.setStyleSheet("font-size: 11px; color: #666; padding: 5px; background: #f0f0f0; border-radius: 8px;")
        left_layout.addWidget(self.status_bar)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #4fc3f7;
                border-radius: 10px;
                background: white;
            }
            QTabBar::tab {
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                background: #e1f5fe;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: #4fc3f7;
                color: white;
            }
        """)
        left_layout.addWidget(self.tabs)
        
        self._add_microwave_tab()
        self._add_audio_tab()
        self._add_fusion_tab()
        self._add_education_tab()
        self._add_health_passport_tab()
        
        exit_btn = QPushButton("EXIT")
        exit_btn.setMinimumHeight(40)
        exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                background: #ef5350;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px;
                margin-top: 10px;
            }
            QPushButton:hover { background: #ff6659; }
        """)
        exit_btn.clicked.connect(self.close)
        left_layout.addWidget(exit_btn)
        
        # Right panel (30%)
        right_panel = QWidget()
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 10, 5, 10)
        right_layout.setSpacing(15)
        
        recon_title = QLabel("Microwave Reconstruction")
        recon_title.setAlignment(Qt.AlignCenter)
        recon_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #0277bd;")
        right_layout.addWidget(recon_title)
        
        self.reconstruction_widget.setMinimumHeight(300)
        right_layout.addWidget(self.reconstruction_widget)
        
        right_layout.addWidget(self.clinical_assessment)
        right_layout.addWidget(self.educational_widget)
        
        mission = QLabel("""
        <b>Operation Oracle Mission</b><br>
        Transforming early detection from a scarce, opaque resource 
        into a portable, explainable, and truly accessible practice.
        <br><br>
        <i>The most effective diagnostic tools don't just process data, 
        but build the literacy necessary to understand it.</i>
        """)
        mission.setWordWrap(True)
        mission.setStyleSheet("font-size: 10px; color: #555; background: #f5f5f5; padding: 10px; border-radius: 10px;")
        mission.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(mission)
        
        main_layout.addWidget(left_panel, 7)
        main_layout.addWidget(right_panel, 3)
    
    def _add_microwave_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.mw_status = QLabel("Ready - Place antennas and connect VNA")
        self.mw_status.setStyleSheet("font-size: 13px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.mw_status)
        
        self.baseline_btn = QPushButton("1. RECORD BASELINE (AIR)")
        self.baseline_btn.setMinimumHeight(45)
        self.baseline_btn.setStyleSheet(self._button_style("#81d4fa"))
        self.baseline_btn.clicked.connect(self._record_baseline)
        layout.addWidget(self.baseline_btn)
        
        self.scan_btn = QPushButton("2. SCAN PATIENT (MULTI-ANGLE)")
        self.scan_btn.setMinimumHeight(45)
        self.scan_btn.setStyleSheet(self._button_style("#4fc3f7"))
        self.scan_btn.clicked.connect(self._run_multi_angle_scan)
        layout.addWidget(self.scan_btn)
        
        self.mw_progress = QProgressBar()
        self.mw_progress.setVisible(False)
        layout.addWidget(self.mw_progress)
        
        self.mw_result = QTextEdit()
        self.mw_result.setReadOnly(True)
        self.mw_result.setMinimumHeight(200)
        self.mw_result.setStyleSheet("font-size: 11px; font-family: monospace;")
        layout.addWidget(self.mw_result)
        
        clear_btn = QPushButton("CLEAR ALL DATA")
        clear_btn.setMinimumHeight(35)
        clear_btn.setStyleSheet(self._button_style("#ff9800"))
        clear_btn.clicked.connect(self._clear_all_data)
        layout.addWidget(clear_btn)
        
        self.tabs.addTab(tab, "Microwave")
    
    def _add_audio_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.audio_status = QLabel("Ready - Place stethoscope on patient's back")
        self.audio_status.setStyleSheet("font-size: 13px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.audio_status)
        
        self.waveform_label = QLabel()
        self.waveform_label.setMinimumHeight(150)
        self.waveform_label.setStyleSheet("background-color: black; border-radius: 8px;")
        layout.addWidget(self.waveform_label)
        
        self.audio_btn = QPushButton("ANALYZE LUNG SOUNDS")
        self.audio_btn.setMinimumHeight(50)
        self.audio_btn.setStyleSheet(self._button_style("#66bb6a"))
        self.audio_btn.clicked.connect(self._run_acoustic_analysis)
        layout.addWidget(self.audio_btn)
        
        self.audio_result = QTextEdit()
        self.audio_result.setReadOnly(True)
        self.audio_result.setMinimumHeight(250)
        self.audio_result.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.audio_result)
        
        self.tabs.addTab(tab, "Acoustic")
    
    def _add_fusion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.fusion_status = QLabel("Perform both scans for combined diagnosis")
        self.fusion_status.setStyleSheet("font-size: 13px; padding: 8px; background: #fff3e0; border-radius: 8px;")
        layout.addWidget(self.fusion_status)
        
        self.fusion_mw_btn = QPushButton("1. SCAN MICROWAVE (MULTI-ANGLE)")
        self.fusion_mw_btn.setMinimumHeight(45)
        self.fusion_mw_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_mw_btn.clicked.connect(self._fusion_microwave)
        layout.addWidget(self.fusion_mw_btn)
        
        self.fusion_audio_btn = QPushButton("2. ANALYZE ACOUSTIC")
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_audio_btn.setMinimumHeight(45)
        self.fusion_audio_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_audio_btn.clicked.connect(self._fusion_acoustic)
        layout.addWidget(self.fusion_audio_btn)
        
        self.fusion_combine_btn = QPushButton("3. RUN FUSION DIAGNOSIS")
        self.fusion_combine_btn.setEnabled(False)
        self.fusion_combine_btn.setMinimumHeight(50)
        self.fusion_combine_btn.setStyleSheet(self._button_style("#4fc3f7"))
        self.fusion_combine_btn.clicked.connect(self._run_fusion)
        layout.addWidget(self.fusion_combine_btn)
        
        self.fusion_result = QTextEdit()
        self.fusion_result.setReadOnly(True)
        self.fusion_result.setMinimumHeight(200)
        self.fusion_result.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.fusion_result)
        
        self.tabs.addTab(tab, "Fusion")
    
    def _add_education_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        welcome = QLabel("Welcome to Clinical Education")
        welcome.setStyleSheet("font-size: 18px; font-weight: bold; color: #0277bd;")
        welcome.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome)
        
        oracle_desc = QLabel("""
        <b>Operation Oracle: Democratizing Early Detection</b><br><br>
        <b>Against Clinical Delay:</b> Rapid, multimodal screening at point-of-care (under 5 minutes)<br><br>
        <b>Against Access Barriers:</b> Portable platforms using democratized hardware (Raspberry Pi + NanoVNA)<br><br>
        <b>Against Systemic Exclusion:</b> Explainable AI + clinical education + complete open-source documentation<br><br>
        <i>Together, we transform early detection from a scarce, opaque resource into a portable, explainable, and truly accessible practice.</i>
        """)
        oracle_desc.setWordWrap(True)
        oracle_desc.setStyleSheet("font-size: 12px; line-height: 1.5; padding: 15px; background: #f5f5f5; border-radius: 10px;")
        layout.addWidget(oracle_desc)
        
        how_it_works = QLabel("""
        <b>How It Works:</b><br><br>
        <b>1. Microwave Scan:</b> 4 antennas measure tissue dielectric properties - tumors appear as contrast anomalies<br><br>
        <b>2. Acoustic Analysis:</b> Lung sound recording detects wheezing, crackles, and breath patterns<br><br>
        <b>3. Fusion Diagnosis:</b> Combines structural (microwave) and functional (acoustic) data for comprehensive assessment
        """)
        how_it_works.setWordWrap(True)
        how_it_works.setStyleSheet("font-size: 11px; padding: 15px; background: #e3f2fd; border-radius: 10px;")
        layout.addWidget(how_it_works)
        
        opensource = QLabel("""
        <b>Open-Source Documentation</b><br><br>
        This project includes comprehensive documentation, online courses, and community forums 
        to enable anyone to build, modify, and improve the system. Democratization of innovation 
        is a fundamental pillar of Operation Oracle.
        """)
        opensource.setWordWrap(True)
        opensource.setStyleSheet("font-size: 11px; padding: 15px; background: #e8f5e9; border-radius: 10px;")
        layout.addWidget(opensource)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Education")
    
    def _add_health_passport_tab(self):
        """Add Health Passport tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        layout.addWidget(self.health_passport)
        
        self.tabs.addTab(tab, "Health Passport")
    
    def _button_style(self, color):
        return f"""
            QPushButton {{
                font-size: 14px;
                font-weight: bold;
                background: {color};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: {color}cc;
            }}
            QPushButton:disabled {{
                background: #cccccc;
                color: #888;
            }}
        """
    
    def _update_mw_progress(self, msg, frac):
        self.mw_status.setText(msg)
        self.mw_progress.setValue(int(frac * 100))
        QApplication.processEvents()
    
    def _clear_all_data(self):
        reply = QMessageBox.question(self, "Clear Data", "Delete all saved baseline and patient scans?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.scanner.csv_manager.clear_all()
            self.scanner._baseline_data = None
            self.current_mw_features = None
            self.current_audio_probs = None
            self.last_ai_probs = None
            self.current_s21_data = None
            self.rotation_scans = {}
            self.reconstruction_widget.clear()
            self.mw_result.setText("All data cleared. You can now record a new baseline.")
            self.mw_status.setText("Ready")
            QMessageBox.information(self, "Data Cleared", "All saved scans have been deleted.")
    
    def _record_baseline(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected.")
            return
        
        self.baseline_btn.setEnabled(False)
        self.mw_status.setText("Recording baseline (air)...")
        self.mw_progress.setVisible(True)
        self.mw_progress.setValue(0)
        self.mw_result.setText("Starting baseline scan - Place nothing between antennas...")
        
        def worker():
            try:
                data = self.scanner.scan_all_paths(
                    BASELINE_DIR, angle=0,
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.scanner.set_baseline(data)
                self.baseline_data = data
                
                result_text = "BASELINE RECORDED SUCCESSFULLY\n\n"
                result_text += "Baseline data saved.\n"
                result_text += f"\nFiles saved to: {BASELINE_DIR}\n"
                result_text += "\nNow place patient between antennas and click SCAN PATIENT (MULTI-ANGLE)."
                
                self.mw_result.setText(result_text)
                self.mw_status.setText("Baseline complete")
                
            except Exception as e:
                self.mw_result.setText(f"Error: {e}\n\nCheck VNA connection and try again.")
                self.mw_status.setText("Baseline failed")
                traceback.print_exc()
            finally:
                self.baseline_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _run_multi_angle_scan(self):
        """Perform scans at multiple rotation angles and combine via averaging"""
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected.")
            return
        
        if not self.scanner.has_baseline():
            QMessageBox.warning(self, "Missing Baseline", "Please record baseline (air) first!")
            return
        
        self.scan_btn.setEnabled(False)
        self.mw_progress.setVisible(True)
        self.mw_progress.setValue(0)
        
        def worker():
            try:
                all_rotation_data = {}
                baseline = self.scanner.load_baseline()
                
                for angle_idx, angle in enumerate(ROTATION_ANGLES):
                    self.mw_status.setText(f"Scanning at {angle} deg rotation...")
                    self.mw_progress.setValue(int(angle_idx / len(ROTATION_ANGLES) * 50))
                    
                    if angle_idx > 0:
                        # In production, use a dialog. For now, auto-continue with delay
                        time.sleep(2)
                    
                    data = self.scanner.scan_all_paths(
                        MULTI_ANGLE_DIR, angle=angle,
                        progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                    )
                    
                    all_rotation_data[angle] = data
                    
                    # Reconstruct image for this rotation
                    self.reconstruction_widget.reconstruct_image(data, self.scanner.frequencies, baseline)
                    
                    self.mw_progress.setValue(int((angle_idx + 1) / len(ROTATION_ANGLES) * 50))
                
                self.rotation_scans = all_rotation_data
                
                # COMBINE FEATURES BY AVERAGING ACROSS ROTATIONS
                # This preserves the 840-dim feature vector
                combined_features = self.scanner.combine_rotation_features(all_rotation_data)
                self.current_mw_features = combined_features
                
                # Final reconstruction using averaged data
                self._reconstruct_from_rotations(all_rotation_data)
                
                # Generate localization analysis using averaged data
                localizer = TumorLocalizer()
                
                # Average attenuation across rotations for localization
                combined_attenuation = {}
                for angle, data in all_rotation_data.items():
                    analysis = localizer.analyze_path_attenuation(data, baseline)
                    for path, atten in analysis['path_attenuation'].items():
                        if path not in combined_attenuation:
                            combined_attenuation[path] = []
                        combined_attenuation[path].append(atten)
                
                avg_attenuation = {p: np.mean(v) for p, v in combined_attenuation.items()}
                sorted_paths = sorted(avg_attenuation.items(), key=lambda x: x[1], reverse=True)
                
                tumor_location = localizer._estimate_location_from_paths(sorted_paths)
                confidence = localizer._calculate_confidence(sorted_paths, {})
                
                w = self.reconstruction_widget.width()
                h = self.reconstruction_widget.height()
                bounding_box = localizer.generate_bounding_box(tumor_location, w, h)
                
                self.reconstruction_widget.set_tumor_localization(tumor_location, confidence, bounding_box)
                
                result_text = "MULTI-ANGLE PATIENT SCAN COMPLETE\n\n"
                result_text += f"Scanned at angles: {ROTATION_ANGLES}\n"
                result_text += f"Total transmission paths: {len(ROTATION_ANGLES) * 4}\n"
                result_text += f"Features combined via averaging to maintain 840-dim vector\n\n"
                
                result_text += "TUMOR LOCALIZATION:\n"
                if confidence > 0.5:
                    result_text += f"   Location: {tumor_location['description']}\n"
                    result_text += f"   Coordinates: ({tumor_location['x']:.0f} mm, {tumor_location['y']:.0f} mm)\n"
                    result_text += f"   Confidence: {confidence:.0%}\n\n"
                    result_text += "   Most affected paths (averaged across rotations):\n"
                    for path_num, attenuation in sorted_paths[:2]:
                        result_text += f"     Path {path_num}: {attenuation:.1f} dB attenuation\n"
                    result_text += "\nCLINICAL INTERPRETATION:\n"
                    result_text += f"   The model identifies an abnormal presence in the {tumor_location['quadrant']} quadrant.\n"
                    result_text += "   This area shows increased attenuation compared to baseline,\n"
                    result_text += "   suggesting potential tissue abnormality.\n"
                    result_text += "   Recommended: Further clinical correlation advised.\n"
                else:
                    result_text += "   No significant abnormality detected\n"
                    result_text += f"   Confidence: {confidence:.0%}\n"
                
                result_text += f"\nData saved to: {MULTI_ANGLE_DIR}\n"
                result_text += "\nReady for fusion with acoustic analysis."
                
                self.mw_result.setText(result_text)
                self.mw_status.setText("Multi-angle scan complete")
                
                if self.fusion is not None:
                    self.fusion_audio_btn.setEnabled(True)
                
            except Exception as e:
                self.mw_result.setText(f"Error: {e}")
                self.mw_status.setText("Scan failed")
                traceback.print_exc()
            finally:
                self.scan_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _reconstruct_from_rotations(self, rotation_data):
        """Reconstruct image using averaged data from all rotations"""
        try:
            # Average S21 data across all rotations
            avg_s21 = {p: [] for p in [1, 2, 3, 4]}
            
            for angle, data in rotation_data.items():
                for path_num in [1, 2, 3, 4]:
                    if path_num in data:
                        avg_s21[path_num].append(data[path_num])
            
            # Compute average for each path
            for path_num in avg_s21:
                if avg_s21[path_num]:
                    avg_s21[path_num] = np.mean(avg_s21[path_num], axis=0)
                else:
                    avg_s21[path_num] = np.zeros(POINTS)
            
            baseline = self.scanner.load_baseline()
            self.reconstruction_widget.reconstruct_image(avg_s21, self.scanner.frequencies, baseline)
            
        except Exception as e:
            print(f"Rotation reconstruction error: {e}")
            traceback.print_exc()
    
    def _run_acoustic_analysis(self):
        self.audio_btn.setEnabled(False)
        self.audio_status.setText("Recording lung sounds (3 seconds)...")
        self.audio_result.setText("Processing...")
        self.waveform_label.setText("")
        
        self.audio_thread = AudioProcessor(RECORD_SECONDS, self.audio_device_id)
        self.audio_thread.result_ready.connect(self._on_audio_result)
        self.audio_thread.waveform_ready.connect(self._draw_waveform)
        self.audio_thread.error_occurred.connect(self._on_audio_error)
        self.audio_thread.finished.connect(self._on_audio_finished)
        self.audio_thread.start()
    
    def _on_audio_result(self, probs):
        self.last_ai_probs = probs
        self.clinical_assessment.show_assessment()
    
    def _on_clinical_assessment(self, patient_symptoms):
        """Handle completed clinical assessment with enhanced decision support"""
        if self.last_ai_probs is None:
            return
        
        environmental_quality = "normal"
        
        final_dx, confidence, explanation, all_probs = self.clinical_decision_support.assess(
            self.last_ai_probs, patient_symptoms, environmental_quality
        )
        
        recommendations = {
            'asthma': "\n\n5. MANAGEMENT RECOMMENDATIONS\n\n   - Use prescribed rescue inhaler as needed\n   - Identify and avoid triggers\n   - Consider daily controller medication\n   - Create an asthma action plan\n",
            'copd': "\n\n5. MANAGEMENT RECOMMENDATIONS\n\n   - Smoking cessation (if applicable)\n   - Pulmonary rehabilitation\n   - Use prescribed bronchodilators\n   - Monitor oxygen saturation\n",
            'pneumonia': "\n\n5. MANAGEMENT RECOMMENDATIONS\n\n   - Seek medical evaluation promptly\n   - Antibiotics if bacterial cause confirmed\n   - Rest and hydration\n   - Follow-up chest X-ray\n",
            'bronchitis': "\n\n5. MANAGEMENT RECOMMENDATIONS\n\n   - Rest and increased fluid intake\n   - Honey or cough suppressants for symptom relief\n   - Avoid irritants (smoke, dust)\n   - See doctor if symptoms persist >3 weeks\n",
            'healthy': "\n\n5. MANAGEMENT RECOMMENDATIONS\n\n   - No specific treatment needed\n   - Maintain healthy lifestyle\n   - Regular exercise and good nutrition\n   - Annual check-ups recommended\n"
        }
        
        result_text = "PULMO AI CLINICAL ASSESSMENT\n\n"
        result_text += explanation
        result_text += recommendations.get(final_dx, "")
        result_text += "\n\n" + "=" * 50 + "\n"
        result_text += "DISCLAIMER: This is an AI-assisted screening tool.\n"
        result_text += "Always consult a qualified healthcare provider for medical decisions."
        
        self.audio_result.setText(result_text)
        self.audio_status.setText(f"Diagnosis: {final_dx.upper()} ({confidence:.1%})")
        
        self.educational_widget.show_condition(final_dx, confidence)
        
        # Store for fusion
        final_probs = np.zeros(5)
        for i, condition in enumerate(MODEL_CLASSES):
            if condition in all_probs:
                final_probs[i] = all_probs[condition]
        
        self.current_audio_probs = final_probs
        
        # Save to Health Passport
        mw_result = "Abnormal" if confidence > 0.7 else "Normal"
        if self.current_mw_features is not None:
            try:
                # If microwave features exist, you could run a quick microwave prediction
                pass
            except:
                mw_result = "Not analyzed"
        else:
            mw_result = "Not scanned"
        
        self.health_passport.add_scan_record(
            diagnosis=final_dx,
            confidence=confidence,
            microwave_result=mw_result,
            audio_result=final_dx,
            audio_probs=final_probs
        )
        
        if self.current_mw_features is not None and self.fusion is not None:
            self.fusion_combine_btn.setEnabled(True)
    
    def _on_audio_error(self, error_msg):
        self.audio_result.setText(f"Audio Error: {error_msg}\n\nCheck microphone and try again.")
        self.audio_status.setText("Audio error")
        print(f"Audio error: {error_msg}")
    
    def _on_audio_finished(self):
        self.audio_btn.setEnabled(True)
    
    def _draw_waveform(self, data):
        if data is None or len(data) == 0:
            return
        w = self.waveform_label.width()
        h = self.waveform_label.height()
        if w < 10 or h < 10:
            return
        
        pixmap = QtGui.QPixmap(w, h)
        pixmap.fill(Qt.black)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.cyan, 2))
        
        mid = h // 2
        data = (data - np.mean(data)) / (np.std(data) + 1e-6)
        data = np.clip(data, -1, 1)
        step = w / len(data)
        
        for i in range(1, len(data)):
            x1 = int((i-1) * step)
            x2 = int(i * step)
            y1 = int(mid + data[i-1] * mid * 0.8)
            y2 = int(mid + data[i] * mid * 0.8)
            painter.drawLine(x1, y1, x2, y2)
        
        painter.end()
        self.waveform_label.setPixmap(pixmap)
    
    def _fusion_microwave(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected.")
            return
        
        self.fusion_mw_btn.setEnabled(False)
        self.fusion_status.setText("Running multi-angle microwave scan...")
        
        def worker():
            try:
                if not self.scanner.has_baseline():
                    self.fusion_status.setText("Need baseline first! Go to Microwave tab.")
                    self.fusion_mw_btn.setEnabled(True)
                    return
                
                all_rotation_data = {}
                baseline = self.scanner.load_baseline()
                
                for angle_idx, angle in enumerate(ROTATION_ANGLES):
                    if angle_idx > 0:
                        time.sleep(2)
                    
                    data = self.scanner.scan_all_paths(MULTI_ANGLE_DIR, angle=angle)
                    all_rotation_data[angle] = data
                    
                    self.reconstruction_widget.reconstruct_image(data, self.scanner.frequencies, baseline)
                
                self.rotation_scans = all_rotation_data
                
                # Combine features by averaging across rotations
                combined_features = self.scanner.combine_rotation_features(all_rotation_data)
                self.current_mw_features = combined_features
                
                # Final reconstruction using averaged data
                self._reconstruct_from_rotations(all_rotation_data)
                
                self.fusion_status.setText("Microwave complete. Now perform acoustic analysis.")
                self.fusion_audio_btn.setEnabled(True)
                
            except Exception as e:
                self.fusion_status.setText(f"Error: {e}")
                traceback.print_exc()
            finally:
                self.fusion_mw_btn.setEnabled(True)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _fusion_acoustic(self):
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_status.setText("Recording acoustic...")
        
        self.audio_thread = AudioProcessor(RECORD_SECONDS, self.audio_device_id)
        self.audio_thread.result_ready.connect(self._on_fusion_audio_result)
        self.audio_thread.error_occurred.connect(self._on_fusion_audio_error)
        self.audio_thread.finished.connect(self._on_fusion_audio_finished)
        self.audio_thread.start()
    
    def _on_fusion_audio_result(self, probs):
        self.last_ai_probs = probs
        self.clinical_assessment.show_assessment()
        # Connect assessment to fusion
        self.clinical_assessment.assessment_complete.disconnect(self._on_clinical_assessment)
        self.clinical_assessment.assessment_complete.connect(self._on_fusion_clinical_assessment)
    
    def _on_fusion_clinical_assessment(self, patient_symptoms):
        """Handle clinical assessment for fusion path"""
        if self.last_ai_probs is None:
            return
        
        environmental_quality = "normal"
        
        final_dx, confidence, explanation, all_probs = self.clinical_decision_support.assess(
            self.last_ai_probs, patient_symptoms, environmental_quality
        )
        
        self.current_audio_probs = np.zeros(5)
        for i, condition in enumerate(MODEL_CLASSES):
            if condition in all_probs:
                self.current_audio_probs[i] = all_probs[condition]
        
        # Save to Health Passport
        mw_result = "Abnormal" if confidence > 0.7 else "Normal"
        self.health_passport.add_scan_record(
            diagnosis=final_dx,
            confidence=confidence,
            microwave_result=mw_result,
            audio_result=final_dx,
            audio_probs=self.current_audio_probs
        )
        
        self.fusion_status.setText("Acoustic complete. Ready for fusion diagnosis.")
        self.fusion_combine_btn.setEnabled(True)
        
        # Reconnect the original handler
        self.clinical_assessment.assessment_complete.disconnect(self._on_fusion_clinical_assessment)
        self.clinical_assessment.assessment_complete.connect(self._on_clinical_assessment)
    
    def _on_fusion_audio_error(self, error_msg):
        self.fusion_status.setText(f"Acoustic error: {error_msg}")
        self.fusion_audio_btn.setEnabled(True)
    
    def _on_fusion_audio_finished(self):
        self.fusion_audio_btn.setEnabled(True)
    
    def _run_fusion(self):
        if self.current_mw_features is None or self.current_audio_probs is None:
            QMessageBox.warning(self, "Missing Data", "Perform both scans first!")
            return
        
        if self.fusion is None:
            self.fusion_result.setText("Fusion model not loaded!")
            return
        
        try:
            pred, conf = self.fusion.predict(self.current_mw_features, self.current_audio_probs)
            class_names = ['baseline', 'healthy', 'tumor']
            result = class_names[pred] if pred < 3 else "unknown"
            
            audio_dx = np.argmax(self.current_audio_probs)
            audio_class = MODEL_CLASSES[audio_dx] if audio_dx < len(MODEL_CLASSES) else "unknown"
            audio_conf = self.current_audio_probs[audio_dx] if audio_dx < len(self.current_audio_probs) else 0
            
            result_text = f"FUSION DIAGNOSIS: {result.upper()}\n\n"
            result_text += f"Overall Confidence: {conf:.1%}\n\n"
            result_text += "Multimodal Interpretation:\n"
            result_text += f"   Microwave (Structural): {['Normal', 'Possible Anomaly', 'Definite Anomaly'][pred]}\n"
            result_text += f"   Acoustic (Functional): {audio_class.upper()} ({audio_conf:.1%})\n\n"
            
            if hasattr(self.reconstruction_widget, 'tumor_location') and self.reconstruction_widget.tumor_location:
                loc = self.reconstruction_widget.tumor_location
                loc_conf = self.reconstruction_widget.localization_confidence
                if loc_conf > 0.5:
                    result_text += "TUMOR LOCALIZATION:\n"
                    result_text += f"   {loc['description']} ({loc['x']:.0f}, {loc['y']:.0f}) mm\n"
                    result_text += f"   Localization Confidence: {loc_conf:.0%}\n\n"
                    result_text += "   The model believes this area to be the location of\n"
                    result_text += "   abnormal tissue presence requiring further investigation.\n\n"
            
            result_text += "Clinical Recommendation:\n"
            
            if result == 'tumor':
                result_text += "   - Further evaluation recommended\n"
                result_text += "   - Consider referral for detailed imaging\n"
                result_text += "   - Schedule follow-up within 2-4 weeks\n"
            elif result == 'healthy':
                result_text += "   - Normal findings\n"
                result_text += "   - Continue regular health maintenance\n"
                result_text += "   - Annual check-ups recommended\n"
            else:
                result_text += "   - Baseline established\n"
                result_text += "   - Repeat scan periodically for monitoring\n"
            
            result_text += "\nClinical education content is available on the right panel."
            
            self.fusion_result.setText(result_text)
            self.fusion_status.setText(f"Diagnosis: {result} ({conf:.1%})")
            
            self.educational_widget.show_condition(audio_class, audio_conf)
            
        except Exception as e:
            self.fusion_result.setText(f"Fusion error: {e}")
            traceback.print_exc()
    
    def closeEvent(self, event):
        print("\nShutting down PULMO AI: Operation Oracle...")
        self.scanner.cleanup()
        event.accept()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    platform = os.environ.get('QT_QPA_PLATFORM', '')
    if platform == 'eglfs':
        QApplication.setAttribute(Qt.AA_UseOpenGLES, True)

    app = QApplication(sys.argv)
    window = PulmoAIMainWindow()
    window.showFullScreen()
    sys.exit(app.exec())
