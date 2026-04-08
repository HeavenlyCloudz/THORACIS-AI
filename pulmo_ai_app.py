#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULMO AI - Operation Oracle: Democratized Lung Screening System
- Phase-based microwave reconstruction
- Multi-angle scanning (0°, 120°, 240°)
- Tumor localization with bounding boxes
- Confidence-based heatmap visualization
- YAMNet audio processing
"""

import os

# Forces OpenGL ES bc it must be set before any Qt imports
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
import struct
import wave
import subprocess

# Qt
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QProgressBar, QMessageBox, QTabWidget, QTextEdit, QLineEdit,
    QComboBox, QFrame, QScrollArea, QGroupBox
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
ROTATION_ANGLES = [0, 120, 240]  # Three rotations for full coverage like in my phantom scans training

# Audio settings
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
EXPECTED_AUDIO_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
YAMNET_EMBEDDING_SIZE = 1024

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
    """Add IFFT-based time-domain features"""
    n_samples, n_features = X.shape
    n_paths = 4
    freq_per_path = n_features // n_paths
    
    time_features = []
    for sample in X:
        sample_time = []
        for p in range(n_paths):
            start = p * freq_per_path
            end = (p + 1) * freq_per_path
            freq_resp = sample[start:end]
            time_resp = np.abs(np.fft.ifft(freq_resp))
            sample_time.extend([
                np.max(time_resp), np.argmax(time_resp),
                np.mean(time_resp), np.std(time_resp),
                np.percentile(time_resp, 90), np.percentile(time_resp, 10),
                np.sum(time_resp), np.max(time_resp) - np.min(time_resp),
                np.sum(time_resp ** 2)
            ])
        time_features.append(sample_time)
    time_features = np.array(time_features)
    return np.concatenate([X, time_features], axis=1)

# =============================================================================
# TUMOR LOCALIZER CLASS
# =============================================================================

class TumorLocalizer:
    """Localize tumors using path attenuation analysis and image reconstruction"""
    
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
# RECONSTRUCTION WIDGET WITH PHASE AND BOUNDING BOX
# =============================================================================

class ReconstructionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 350)
        self.reconstruction_data = None
        self.heatmap_data = None
        self.tumor_location = None
        self.bounding_box = None
        self.localization_confidence = 0
        self.setStyleSheet("background-color: white; border: 2px solid #4fc3f7; border-radius: 10px;")
    
    def reconstruct_image_with_phase(self, s21_complex_data, frequencies, baseline_complex_data=None):
        """
        Phase-based coherent reconstruction for sharper images
        s21_complex_data: dict of path_num -> complex numpy array
        """
        try:
            grid_size = 80
            x_grid = np.linspace(-100, 100, grid_size)
            y_grid = np.linspace(-100, 100, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            image_complex = np.zeros((grid_size, grid_size), dtype=np.complex64)
            c = 3e8  # speed of light in m/s
            
            for path_num, s21 in s21_complex_data.items():
                if path_num not in PATH_TO_ANTENNA_PAIR:
                    continue
                
                tx_ant, rx_ant = PATH_TO_ANTENNA_PAIR[path_num]
                tx_pos = ANTENNA_POSITIONS[tx_ant]
                rx_pos = ANTENNA_POSITIONS[rx_ant]
                
                # Subtract baseline in complex domain if available
                if baseline_complex_data and path_num in baseline_complex_data:
                    s21 = s21 - baseline_complex_data[path_num]
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        point = (X[i, j], Y[i, j])
                        
                        d_tx = np.sqrt((tx_pos[0] - point[0])**2 + (tx_pos[1] - point[1])**2)
                        d_rx = np.sqrt((rx_pos[0] - point[0])**2 + (rx_pos[1] - point[1])**2)
                        total_dist = (d_tx + d_rx) / 1000  # Convert to meters
                        
                        # Coherent summation across frequencies
                        for f_idx, freq in enumerate(frequencies):
                            omega = 2 * np.pi * freq * 1e9
                            phase_shift = np.exp(-1j * omega * total_dist / c)
                            image_complex[i, j] += s21[f_idx] * phase_shift
            
            # Normalize
            image_complex /= len(s21_complex_data) * len(frequencies)
            
            # Take magnitude for display
            magnitude = np.abs(image_complex)
            
            # Apply Gaussian filter for smoothing
            magnitude = gaussian_filter(magnitude, sigma=1.5)
            
            # Clip to 95th percentile for better contrast
            magnitude = np.clip(magnitude, 0, np.percentile(magnitude, 95))
            
            # Normalize to 0-255
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max()) * 255
            
            self.reconstruction_data = magnitude.astype(np.uint8)
            self.heatmap_data = magnitude.astype(np.uint8)
            self.update()
            
            return self.reconstruction_data, image_complex
            
        except Exception as e:
            print(f"Phase reconstruction error: {e}")
            traceback.print_exc()
            return None, None
    
    def set_tumor_localization(self, tumor_location, confidence, bounding_box):
        """Set tumor localization data for visualization"""
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
            
            # Draw heatmap
            for i in range(img_width):
                for j in range(img_height):
                    val = self.reconstruction_data[i, j]
                    # Use colormap: blue (low) -> red (high)
                    r = int(val)
                    g = int(val * 0.5)
                    b = int(255 - val)
                    color = QColor(r, g, b)
                    x = int(i * scale_x)
                    y = int(j * scale_y)
                    painter.fillRect(x, y, int(scale_x) + 1, int(scale_y) + 1, color)
            
            # Draw bounding box if tumor detected with good confidence
            if self.bounding_box and self.localization_confidence > 0.5:
                x1, y1, x2, y2 = self.bounding_box
                
                # Draw red bounding box
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                
                # Draw crosshair at center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(center_x - 10, center_y, center_x + 10, center_y)
                painter.drawLine(center_x, center_y - 10, center_x, center_y + 10)
                
                # Draw confidence label background
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                painter.drawRect(x1, y1 - 25, 120, 20)
                painter.drawText(x1 + 5, y1 - 10, f"Confidence: {self.localization_confidence:.0%}")
        
        else:
            painter.setPen(QPen(QColor(150, 150, 150), 2))
            painter.drawRect(10, 10, w - 20, h - 20)
            painter.drawText(w//2 - 100, h//2, "Reconstruction will appear here after scan")
        
        # Draw antenna positions
        painter.setPen(QPen(QColor(0, 0, 255), 3))
        painter.setBrush(QBrush(QColor(0, 0, 255)))
        for ant, pos in ANTENNA_POSITIONS.items():
            x = int((pos[0] + 100) / 200 * w)
            y = int((pos[1] + 100) / 200 * h)
            painter.drawEllipse(x - 5, y - 5, 10, 10)
            painter.drawText(x - 15, y - 10, f"{ant}")
        
        painter.end()
    
    def clear(self):
        """Clear reconstruction data"""
        self.reconstruction_data = None
        self.heatmap_data = None
        self.tumor_location = None
        self.bounding_box = None
        self.localization_confidence = 0
        self.update()

# =============================================================================
# AUDIO PROCESSOR WITH YAMNET
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
            print("✅ YAMNet TFLite loaded")
            
            if not AUDIO_MODEL_PATH.exists():
                self.error_occurred.emit(f"Classifier not found: {AUDIO_MODEL_PATH}")
                return
            
            self.classifier_interpreter = tflite.Interpreter(str(AUDIO_MODEL_PATH))
            self.classifier_interpreter.allocate_tensors()
            print("✅ Classifier loaded")
            
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
# EDUCATIONAL WIDGET
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
        scroll.setStyleSheet("border: none; background-color: transparent;")
        
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setSpacing(15)
        
        self.condition_label = QLabel()
        self.condition_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.condition_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(self.condition_label)
        
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("font-size: 13px; line-height: 1.5;")
        self.content_layout.addWidget(self.desc_label)
        
        signs_group = QGroupBox("Clinical Signs and Symptoms")
        signs_group.setStyleSheet("""
            QGroupBox { font-size: 14px; font-weight: bold; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; padding-top: 10px; }
        """)
        signs_layout = QVBoxLayout(signs_group)
        self.signs_label = QLabel()
        self.signs_label.setWordWrap(True)
        self.signs_label.setStyleSheet("font-size: 12px;")
        signs_layout.addWidget(self.signs_label)
        self.content_layout.addWidget(signs_group)
        
        rec_group = QGroupBox("Recommendations")
        rec_group.setStyleSheet("""
            QGroupBox { font-size: 14px; font-weight: bold; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; padding-top: 10px; }
        """)
        rec_layout = QVBoxLayout(rec_group)
        self.rec_label = QLabel()
        self.rec_label.setWordWrap(True)
        self.rec_label.setStyleSheet("font-size: 12px;")
        rec_layout.addWidget(self.rec_label)
        self.content_layout.addWidget(rec_group)
        
        literacy_note = QLabel("Clinical literacy empowers patients to recognize symptoms early and seek appropriate care.")
        literacy_note.setWordWrap(True)
        literacy_note.setStyleSheet("font-size: 11px; font-style: italic; color: #666; background-color: #e3f2fd; padding: 8px; border-radius: 8px;")
        self.content_layout.addWidget(literacy_note)
        
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
# VNA CONTROLLER WITH COMPLEX DATA
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
    
    def capture_s21_complex(self, progress_callback=None):
        """Capture complex S21 data (real and imaginary parts)"""
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return None, None
        
        try:
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5\r\n"
            self.serial_conn.write(cmd.encode())
            
            time.sleep(2.0)
            
            real_parts = []
            imag_parts = []
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
                            freq_hz = float(parts[0])
                            s21_real = float(parts[1])
                            s21_imag = float(parts[2])
                            
                            real_parts.append(s21_real)
                            imag_parts.append(s21_imag)
                            lines_collected += 1
                            
                            if progress_callback and lines_collected % 20 == 0:
                                progress_callback(lines_collected, POINTS)
                        except ValueError:
                            continue
                else:
                    time.sleep(0.05)
            
            if len(real_parts) == POINTS:
                if self.frequencies is None:
                    self.frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, POINTS)
                return np.array(real_parts), np.array(imag_parts)
            else:
                print(f"Only {len(real_parts)}/{POINTS} points captured")
                return None, None
                
        except Exception as e:
            print(f"VNA capture error: {e}")
            return None, None
    
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
        print(f"Data directories created")
    
    def save_scan_complex(self, real_data, imag_data, path_num, directory, frequencies=None, angle=0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"path{path_num}_angle{angle}_{timestamp}.csv"
        filepath = directory / filename
        
        if frequencies is None:
            frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, len(real_data))
        
        rows = []
        for i, (freq, real, imag) in enumerate(zip(frequencies, real_data, imag_data)):
            rows.append([freq, real, imag])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_GHz', 'S21_real', 'S21_imag'])
            writer.writerows(rows)
        
        return filepath
    
    def load_complex_from_directory(self, directory):
        """Load all complex data from a directory"""
        complex_data = {}
        for path_num in [1, 2, 3, 4]:
            files = list(directory.glob(f"path{path_num}_*.csv"))
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest)
                complex_data[path_num] = df['S21_real'].values + 1j * df['S21_imag'].values
        return complex_data if complex_data else None
    
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
# MICROWAVE SCANNER WITH MULTI-ANGLE SUPPORT
# =============================================================================

class MicrowaveScanner:
    def __init__(self, vna_controller):
        self.vna = vna_controller
        self.switch = RFSwitchController()
        self.csv_manager = CSVDataManager()
        self.frequencies = None
        self._baseline_complex = None
        self._last_complex_data = None
    
    def scan_all_paths_complex(self, save_dir, angle=0, progress_callback=None):
        """Scan all paths and return complex data"""
        data_complex = {}
        data_mag_db = {}
        total_paths = len(PATHS)
        
        for idx, path_num in enumerate(PATHS.keys(), 1):
            if progress_callback:
                progress_callback(f"Setting Path {path_num} at {angle}°", idx / total_paths)
            
            self.switch.set_path(path_num)
            time.sleep(0.2)
            
            if progress_callback:
                progress_callback(f"Capturing Path {path_num}", idx / total_paths)
            
            real_data, imag_data = self.vna.capture_s21_complex()
            if real_data is None:
                raise RuntimeError(f"Failed to capture path {path_num}")
            
            complex_data = real_data + 1j * imag_data
            data_complex[path_num] = complex_data
            data_mag_db[path_num] = 20 * np.log10(np.abs(complex_data) + 1e-12)
            
            if self.frequencies is None:
                self.frequencies = self.vna.frequencies
            
            self.csv_manager.save_scan_complex(real_data, imag_data, path_num, save_dir, self.frequencies, angle)
            
            if progress_callback:
                progress_callback(f"Path {path_num} complete", idx / total_paths)
        
        return data_complex, data_mag_db
    
    def set_baseline(self, baseline_complex):
        self._baseline_complex = baseline_complex.copy()
        print(f"Baseline stored for {len(self._baseline_complex)} paths")
    
    def load_baseline(self):
        if self._baseline_complex is not None:
            return self._baseline_complex
        return self.csv_manager.load_complex_from_directory(BASELINE_DIR)
    
    def has_baseline(self):
        return self.csv_manager.has_baseline() or (self._baseline_complex is not None)
    
    def extract_features_from_complex(self, complex_data):
        """Extract features from complex data (magnitude only for ML consistency)"""
        magnitude = [np.abs(complex_data[p]) for p in [1, 2, 3, 4]]
        raw = np.array(magnitude).reshape(1, -1)
        return add_time_domain_features(raw)[0]
    
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
        
        try:
            self.fusion = FusionClassifier()
        except Exception as e:
            print(f"Fusion not loaded: {e}")
            self.fusion = None
        
        self.current_mw_features = None
        self.current_audio_probs = None
        self.current_complex_data = None
        self.baseline_complex = None
        self.rotation_scans = {}  # Store scans at different angles
        
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
        
        recon_title = QLabel("Microwave Reconstruction (Phase-Based)")
        recon_title.setAlignment(Qt.AlignCenter)
        recon_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #0277bd;")
        right_layout.addWidget(recon_title)
        
        self.reconstruction_widget.setMinimumHeight(300)
        right_layout.addWidget(self.reconstruction_widget)
        
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
        self.audio_result.setMinimumHeight(180)
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
            self.scanner._baseline_complex = None
            self.current_mw_features = None
            self.current_audio_probs = None
            self.current_complex_data = None
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
                data_complex, data_mag = self.scanner.scan_all_paths_complex(
                    BASELINE_DIR, angle=0,
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.scanner.set_baseline(data_complex)
                self.baseline_complex = data_complex
                
                result_text = f"BASELINE RECORDED SUCCESSFULLY\n\n"
                result_text += "Complex baseline data saved.\n"
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
        """Perform scans at multiple rotation angles with user prompts"""
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
                all_complex_data = {}
                all_mag_data = {}
                
                for angle_idx, angle in enumerate(ROTATION_ANGLES):
                    # Update UI for this angle
                    self.mw_status.setText(f"Scanning at {angle}° rotation...")
                    self.mw_progress.setValue(int(angle_idx / len(ROTATION_ANGLES) * 50))
                    
                    # Prompt user to rotate
                    if angle_idx > 0:
                        self.mw_result.setText(f"Please rotate the phantom to {angle}° and click OK")
                        # Note: In a real app, you'd use a dialog here
                        # For now, we'll just wait 3 seconds
                        time.sleep(3)
                    
                    # Scan at this angle
                    data_complex, data_mag = self.scanner.scan_all_paths_complex(
                        MULTI_ANGLE_DIR, angle=angle,
                        progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                    )
                    
                    all_complex_data[angle] = data_complex
                    all_mag_data[angle] = data_mag
                    
                    self.mw_progress.setValue(int((angle_idx + 1) / len(ROTATION_ANGLES) * 50))
                
                self.rotation_scans = all_complex_data
                self.current_complex_data = all_complex_data
                
                # Combine features from all rotations
                combined_features = self._combine_rotation_features(all_mag_data)
                self.current_mw_features = combined_features
                
                # Reconstruct image using all rotations (phase-based)
                self._reconstruct_from_rotations(all_complex_data)
                
                # Generate localization analysis
                baseline = self.scanner.load_baseline()
                localizer = TumorLocalizer()
                
                # Combine attenuation across rotations
                combined_attenuation = {}
                for angle, data in all_mag_data.items():
                    analysis = localizer.analyze_path_attenuation(data, baseline)
                    for path, atten in analysis['path_attenuation'].items():
                        if path not in combined_attenuation:
                            combined_attenuation[path] = []
                        combined_attenuation[path].append(atten)
                
                # Average across rotations
                avg_attenuation = {p: np.mean(v) for p, v in combined_attenuation.items()}
                sorted_paths = sorted(avg_attenuation.items(), key=lambda x: x[1], reverse=True)
                
                tumor_location = localizer._estimate_location_from_paths(sorted_paths)
                confidence = localizer._calculate_confidence(sorted_paths, {})
                
                # Generate bounding box
                w = self.reconstruction_widget.width()
                h = self.reconstruction_widget.height()
                bounding_box = localizer.generate_bounding_box(tumor_location, w, h)
                
                # Update reconstruction widget with localization
                self.reconstruction_widget.set_tumor_localization(tumor_location, confidence, bounding_box)
                
                # Format result text
                result_text = f"MULTI-ANGLE PATIENT SCAN COMPLETE\n\n"
                result_text += f"Scanned at angles: {ROTATION_ANGLES}\n"
                result_text += f"Total transmission paths: {len(ROTATION_ANGLES) * 4}\n\n"
                
                result_text += f"TUMOR LOCALIZATION:\n"
                if confidence > 0.5:
                    result_text += f"   Location: {tumor_location['description']}\n"
                    result_text += f"   Coordinates: ({tumor_location['x']:.0f} mm, {tumor_location['y']:.0f} mm)\n"
                    result_text += f"   Confidence: {confidence:.0%}\n\n"
                    result_text += f"   Most affected paths (averaged across rotations):\n"
                    for path_num, attenuation in sorted_paths[:2]:
                        result_text += f"     Path {path_num}: {attenuation:.1f} dB attenuation\n"
                    result_text += f"\nClinical Interpretation:\n"
                    result_text += f"   The model identifies an abnormal presence in the {tumor_location['quadrant']} quadrant.\n"
                    result_text += f"   This area shows increased attenuation compared to baseline,\n"
                    result_text += f"   suggesting potential tissue abnormality.\n"
                    result_text += f"   Recommended: Further clinical correlation advised.\n"
                else:
                    result_text += f"   No significant abnormality detected (confidence: {confidence:.0%})\n"
                    result_text += f"   Multi-angle scan appears within normal baseline range.\n"
                
                result_text += f"\nData saved to: {MULTI_ANGLE_DIR}\n"
                result_text += f"\nReady for fusion with acoustic analysis."
                
                self.mw_result.setText(result_text)
                self.mw_status.setText("Multi-angle scan complete")
                
                # Enable fusion buttons
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
    
    def _combine_rotation_features(self, rotation_mag_data):
        """Combine magnitude features from multiple rotations"""
        all_raw_features = []
        
        for angle, data in rotation_mag_data.items():
            raw = np.array([data[p] for p in [1, 2, 3, 4]]).reshape(1, -1)
            all_raw_features.append(raw)
        
        combined = np.concatenate(all_raw_features, axis=1)
        return add_time_domain_features(combined)[0]
    
    def _reconstruct_from_rotations(self, rotation_complex_data):
        """Reconstruct image using phase-based method from all rotations"""
        try:
            # Combine complex data from all rotations
            all_complex = {}
            for angle, data in rotation_complex_data.items():
                for path_num, complex_val in data.items():
                    if path_num not in all_complex:
                        all_complex[path_num] = []
                    all_complex[path_num].append(complex_val)
            
            # Average complex data across rotations
            avg_complex = {}
            for path_num, values in all_complex.items():
                avg_complex[path_num] = np.mean(np.array(values), axis=0)
            
            baseline = self.scanner.load_baseline()
            
            # Perform phase-based reconstruction
            image, _ = self.reconstruction_widget.reconstruct_image_with_phase(
                avg_complex, self.scanner.frequencies, baseline
            )
            
            return image
            
        except Exception as e:
            print(f"Rotation reconstruction error: {e}")
            traceback.print_exc()
            return None
    
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
        self.current_audio_probs = probs
        
        model_class_idx = np.argmax(probs)
        confidence = probs[model_class_idx]
        class_name = MODEL_CLASSES[model_class_idx]
        
        self.educational_widget.show_condition(class_name, confidence)
        
        result_text = f"ACOUSTIC DIAGNOSIS\n\n"
        result_text += f"PREDICTION: {class_name.upper()}\n"
        result_text += f"Confidence: {confidence:.1%}\n\n"
        result_text += "Detailed Probabilities:\n"
        for i, (c, p) in enumerate(zip(MODEL_CLASSES, probs)):
            result_text += f"   {c}: {p:.1%}\n"
        result_text += "\nClinical education content displayed on the right panel."
        
        self.audio_result.setText(result_text)
        self.audio_status.setText(f"Diagnosis: {class_name} ({confidence:.1%})")
        
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
                
                # Perform multi-angle scan
                all_complex_data = {}
                all_mag_data = {}
                
                for angle_idx, angle in enumerate(ROTATION_ANGLES):
                    if angle_idx > 0:
                        time.sleep(2)  # Simulate rotation delay
                    
                    data_complex, data_mag = self.scanner.scan_all_paths_complex(MULTI_ANGLE_DIR, angle=angle)
                    all_complex_data[angle] = data_complex
                    all_mag_data[angle] = data_mag
                
                self.rotation_scans = all_complex_data
                self.current_complex_data = all_complex_data
                
                # Combine features
                combined_features = self._combine_rotation_features(all_mag_data)
                self.current_mw_features = combined_features
                
                # Reconstruct
                self._reconstruct_from_rotations(all_complex_data)
                
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
        self.current_audio_probs = probs
        self.fusion_status.setText("Acoustic complete. Ready for fusion diagnosis.")
        self.fusion_combine_btn.setEnabled(True)
    
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
            
            audio_class_idx = np.argmax(self.current_audio_probs)
            audio_class = MODEL_CLASSES[audio_class_idx]
            audio_conf = self.current_audio_probs[audio_class_idx]
            
            result_text = f"FUSION DIAGNOSIS: {result.upper()}\n\n"
            result_text += f"Overall Confidence: {conf:.1%}\n\n"
            result_text += "Multimodal Interpretation:\n"
            result_text += f"   Microwave (Structural): {['Normal', 'Possible Anomaly', 'Definite Anomaly'][pred]}\n"
            result_text += f"   Acoustic (Functional): {audio_class.upper()} ({audio_conf:.1%})\n\n"
            
            # Add localization info if available
            if hasattr(self.reconstruction_widget, 'tumor_location') and self.reconstruction_widget.tumor_location:
                loc = self.reconstruction_widget.tumor_location
                loc_conf = self.reconstruction_widget.localization_confidence
                if loc_conf > 0.5:
                    result_text += f"TUMOR LOCALIZATION:\n"
                    result_text += f"   {loc['description']} ({loc['x']:.0f}, {loc['y']:.0f}) mm\n"
                    result_text += f"   Localization Confidence: {loc_conf:.0%}\n\n"
                    result_text += f"   The model believes this area to be the location of\n"
                    result_text += f"   abnormal tissue presence requiring further investigation.\n\n"
            
            result_text += "Clinical Recommendation:\n"
            
            if result == 'tumor':
                result_text += "   • Further evaluation recommended\n"
                result_text += "   • Consider referral for detailed imaging\n"
                result_text += "   • Schedule follow-up within 2-4 weeks\n"
            elif result == 'healthy':
                result_text += "   • Normal findings\n"
                result_text += "   • Continue regular health maintenance\n"
                result_text += "   • Annual check-ups recommended\n"
            else:
                result_text += "   • Baseline established\n"
                result_text += "   • Repeat scan periodically for monitoring\n"
            
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
