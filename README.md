# THORACIS AI 🫁

**An integrated acoustic-microwave fusion platform for non-invasive, accessible lung cancer screening**

THORACIS AI is a research-driven biomedical engineering project that combines microwave imaging, acoustic analysis, and deep learning to create an affordable, portable lung cancer screening system. By fusing structural microwave data with functional acoustic signatures, THORACIS AI aims to democratize early lung cancer detection—especially in underserved and remote communities where traditional diagnostic infrastructure is unavailable.

---

## 🔭 Overview

Lung cancer remains the leading cause of cancer death worldwide, largely due to late-stage diagnosis. Current screening methods like low-dose CT scans are effective but face significant barriers:

- **💰 Cost:** CT scanners cost $100,000+ and require specialized facilities
- **☢️ Radiation:** Cumulative radiation exposure limits repeated screening
- **🌍 Accessibility:** Rural and remote communities lack access to screening infrastructure
- **👨‍⚕️ Expertise:** Shortage of radiologists to interpret results, especially in low-income regions

THORACIS AI addresses these challenges through a novel **dual-modality fusion approach** combining:

1. **Microwave imaging** — Safe, non-ionizing structural sensing of lung tissue using ultra-wideband antennas
2. **Acoustic analysis** — Functional assessment of lung health through digital auscultation and deep learning
3. **Multi-modal fusion** — Combined analysis that outperforms either modality alone

The entire system is built on low-cost, off-the-shelf components and is completely **open-source**, enabling communities to build, adapt, and deploy their own screening solutions.

---

## ✨ Features

### 🔬 Microwave Imaging
- **4-antenna switched array** for multi-angle transmission measurements
- **2-3 GHz frequency range** optimized for lung tissue penetration
- **S21 parameter analysis** for detecting dielectric property variations
- **Tumor detection threshold:** ~4.9 dB signal drop for 2cm simulated masses
- **Spatial sensitivity:** Signal variation with tumor position enables localization

### 🎧 Acoustic Analysis
- **YAMNet-based feature extraction** (1,024-dimensional embeddings)
- **5-class classification:** COPD, Asthma, Pneumonia, Healthy, Bronchial
- **Digital stethoscope integration** with dual-microphone setup
- **Real-time audio processing** capability

### 🤖 Multi-Modal Fusion
- **Feature-level concatenation** of microwave and acoustic embeddings
- **XGBoost classifier** for unified diagnostic prediction
- **Confidence scoring** for clinical decision support
- **Explainability hooks** for future Grad-CAM integration

### 🌍 Democratization
- **Total system cost:** < $500 (fraction of traditional alternatives)
- **Open-source hardware designs** (3D-printable enclosures)
- **Complete software stack** available on GitHub
- **Educational course** (in development) for community replication


## 🧰 Hardware Components

### Core Components
| Component | Quantity | Purpose | Approx. Cost |
|-----------|----------|---------|--------------|
| Raspberry Pi 4 (4GB) | 1 | Main processor & control | $90.99 |
| NanoVNA-F V2 | 1 | Vector network analyzer for S21 measurement | $89.99 |
| Mini-Circuits ZFSWA-2-46 RF Switches | 2 | SPDT switches for antenna multiplexing | $55.99 (used) |
| UWB Vivaldi Antennas | 4-6 | Microwave transmission/reception | $53.94 (6pk) |
| SMA Cables (Bingfu) | Multiple | RF signal routing | $45.99 |
| 2N2222 NPN Transistors | 2 | Switch control interface | Included in kit |
| 1kΩ Resistors | 2 | Current limiting for transistors | Included in kit |
| Breadboard | 1 | Prototyping circuit assembly | Included in kit |
| Jumper Wires & Alligator Clips | Various | Connections to RF switches | Included in kit |

### Acoustic Components
| Component | Quantity | Purpose | Approx. Cost |
|-----------|----------|---------|--------------|
| BOYA BY-M1S Lavalier Microphones | 2 | Stereo lung sound capture | $49.30 |
| Primacare Stethoscope (modified) | 2 | Professional acoustic coupling | $24.90 |
| UGREEN USB-C to 3.5mm Adapter | 2 | Microphone connection to Pi | $31.98 |

### Phantom Materials
| Component | Quantity | Purpose | Approx. Cost |
|-----------|----------|---------|--------------|
| Agar Agar Powder | 1 | Tissue-mimicking gel base | $14.95 |
| Polyethylene Foam Padding | 1 | Low-signal-loss enclosure | $25.89 |
| Clear Silicone Caulk | 1 | Low-loss adhesive | $32.90 |
| 7-inch HDMI Touch Display | 1 | Interactive GUI (optional) | $69.99 |

**Total System Cost:** ~$450-500 (significantly less than traditional medical imaging)

---

## 💻 Software Components

### Core Scripts

| Script | Description |
|--------|-------------|
| `switch_controller.py` | Handles digital control of Tx and Rx RF switches via GPIO |
| `vna_interface.py` | Interfaces with NanoVNA to acquire S21 transmission data |
| `array_scanner.py` | Iterates through antenna combinations and records measurements |
| `calibration.py` | Applies baseline subtraction and normalization |
| `feature_extraction.py` | Converts raw S21 traces into ML-ready features |
| `audio_processor.py` | Captures and processes lung sounds via YAMNet |
| `fusion_classifier.py` | Combines microwave and acoustic features for unified prediction |
| `main_cli.py` | Command-line interface for automated scanning |

### Dependencies

```bash
# Core dependencies
pip install numpy pandas scikit-learn matplotlib
pip install tensorflow  # for YAMNet
pip install pyserial    # for VNA communication
pip install RPi.GPIO    # for switch control (on Pi)

### YAMNet for Audio Processing

YAMNet is a pre-trained deep neural network that extracts 1,024-dimensional embedding vectors from audio waveforms. These embeddings capture complex acoustic patterns indicative of pulmonary conditions.

```python
import yamnet
import tensorflow as tf

# Load pre-trained model
yamnet_model = yamnet.yamnet_frames_model()
yamnet_model.load_weights('yamnet.h5')

# Extract embeddings from lung sound
scores, embeddings, spectrogram = yamnet_model(audio)
# embeddings.shape = (num_frames, 1024)
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HeavenlyCloudz/THORACIS AI.git
cd THORACIS AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Hardware Setup

#### RF Switch Wiring (per switch)

```text
GPIO 17 ──[1kΩ]───[2N2222 Base]
                    Collector ──→ +5V rail
                    Emitter ────→ RF1 pin (via alligator clip)

GPIO 18 ──[1kΩ]───[2N2222 Base]
                    Collector ──→ +5V rail
                    Emitter ────→ RF2 pin (via alligator clip)

RF1/RF2 also have 10kΩ pull-down resistors to GND
```

#### Antenna Connections

- **Switch #1 (TX):** COM → VNA CH0, Port 1 → Antenna 1, Port 2 → Antenna 2  
- **Switch #2 (RX):** COM → VNA CH1, Port 1 → Antenna 3, Port 2 → Antenna 4  

#### Power

- Connect Pi 5V pin to breadboard positive rail  
- Connect Pi GND pin to breadboard negative rail  

### 4. Verify Installation

```bash
python test_connections.py
```

This script tests GPIO control, VNA communication, and antenna paths.

---

## 🚀 Usage

### Quick Start

```bash
# Run a complete scan with all 4 antenna paths
python main_cli.py --scan --output ./data/scan_001

# Extract features from scan data
python feature_extraction.py --input ./data/scan_001 --output ./features/

# Process audio file
python audio_processor.py --input ./audio/patient01.wav --output ./features/audio/

# Run fusion classifier
python fusion_classifier.py --microwave ./features/microwave/ --audio ./features/audio/
```

---

## Step-by-Step Workflow

### 1. Calibrate System

```bash
python calibration.py --baseline --output ./calibration/
```

This records baseline measurements with no phantom present.

### 2. Scan Phantom/Tissue

```bash
python array_scanner.py --phantom ./phantoms/lung_001 --output ./data/patient001/
```

The scanner cycles through all 4 antenna paths and saves S21 data.

### 3. Extract Microwave Features

```bash
python feature_extraction.py --input ./data/patient001/ --output ./features/patient001/
```

**Features include:**

- Mean attenuation per path  
- Frequency-domain slope  
- Minimum/maximum attenuation  
- Variance across frequency  
- Path-to-path ratios (spatial features)  

### 4. Process Acoustic Data

```bash
python audio_processor.py --input ./audio/patient001.wav --output ./features/patient001/audio.npy
```

### 5. Run Fusion Prediction

```bash
python fusion_classifier.py \
    --microwave ./features/patient001/microwave_features.csv \
    --audio ./features/patient001/audio.npy \
    --model ./models/fusion_xgboost.pkl
```

---

## 📊 Dataset

### Acoustic Dataset

**Source:** Asthma Detection Dataset Version 2 (Kaggle)

#### Class Distribution

| Class      | Samples |
|------------|----------|
| COPD       | 401      |
| Asthma     | 288      |
| Pneumonia  | 285      |
| Healthy    | 133      |
| Bronchial  | 104      |
| **Total**  | **1,211** |

**Preprocessing:**

- YAMNet feature extraction (1,024-dim embeddings)  
- Train/validation split: 80/20 with stratification  
- Class weighting to address imbalance  

---

### Microwave Phantom Data

#### Phantom Composition

- Agar-based tissue-mimicking material  
- Dielectric properties tuned to match lung tissue (εr ≈ 45–50 at 2–3 GHz)  
- Tumor-mimicking inclusions with higher water content (εr ≈ 55–60)  

#### Experimental Conditions

| Condition    | Description                           |
|--------------|---------------------------------------|
| Baseline     | Air only (no phantom)                |
| Healthy      | Saline solution (0.9% NaCl)          |
| Tumor        | Saline + aluminum sphere (2–3 cm)    |
| Spatial Test | Tumor at various positions           |

#### Key Results

| Measurement        | S21 (dB) | Δ from Baseline |
|--------------------|----------|-----------------|
| Air Baseline       | -17.83   | —               |
| Healthy Phantom    | -19.62   | -1.79 dB        |
| Tumor Phantom      | -24.52   | -6.69 dB / -4.90 dB |

---

## 🧠 Model Training

### Acoustic Model (YAMNet + Classifier)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
loss = CategoricalCrossentropy()
epochs = 100
batch_size = 32
```

---

### Microwave Feature Classifier

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic'
)
```

---

### Fusion Classifier

```python
microwave_features = extract_microwave_features(s21_data)
audio_embedding = yamnet_model(audio)
fusion_vector = np.concatenate([microwave_features, audio_embedding])

fusion_model = XGBClassifier(n_estimators=150, max_depth=8)
fusion_model.fit(X_fusion_train, y_train)
```

---

## 📡 Microwave Imaging Subsystem

### Theory

Microwave imaging leverages dielectric property contrast between healthy and malignant tissue.

| Property                     | Healthy Tissue | Tumor Tissue | Difference |
|------------------------------|---------------|--------------|------------|
| Water Content                | ~70%          | ~85%         | Higher     |
| Relative Permittivity (εr)   | ~45–50        | ~55–60       | +10–20%    |
| Conductivity (σ)             | ~1.5 S/m      | ~2.5 S/m     | +60–70%    |

```text
E_total = E_incident + E_scattered
```

```text
E_scattered ∝ ∫(contrast function × Green's function × E_incident) dV
```

---

### 4-Antenna Switched Array

#### Switch Control Logic

| Path | TX Antenna | RX Antenna | TX Switch State      | RX Switch State      |
|------|------------|------------|----------------------|----------------------|
| 1    | Antenna 1  | Antenna 3  | RF1=+5V, RF2=GND     | RF1=+5V, RF2=GND     |
| 2    | Antenna 1  | Antenna 4  | RF1=+5V, RF2=GND     | RF1=GND, RF2=+5V     |
| 3    | Antenna 2  | Antenna 3  | RF1=GND, RF2=+5V     | RF1=+5V, RF2=GND     |
| 4    | Antenna 2  | Antenna 4  | RF1=GND, RF2=+5V     | RF1=GND, RF2=+5V     |

---

### Data Acquisition

- Frequency range: 2–3 GHz  
- Points per sweep: 201  
- Parameters: S21 magnitude and phase  
- Sweep time: ~100 ms per path  
- Total scan time: <0.5 seconds  

---

### Feature Extraction

```python
def extract_features(s21_data, frequencies):
    features = {}
    for path in range(4):
        features[f'path{path}_mean'] = np.mean(s21_data[path])
        features[f'path{path}_std'] = np.std(s21_data[path])
        features[f'path{path}_min'] = np.min(s21_data[path])
        features[f'path{path}_max'] = np.max(s21_data[path])
        slope = np.polyfit(frequencies, s21_data[path], 1)[0]
        features[f'path{path}_slope'] = slope

    features['ratio_13_14'] = features['path0_mean'] / features['path1_mean']
    features['ratio_23_24'] = features['path2_mean'] / features['path3_mean']
    features['asymmetry'] = (
        features['path0_mean'] + features['path2_mean']
    ) / (
        features['path1_mean'] + features['path3_mean']
    )

    return features
```

---

## 🎵 Acoustic Analysis Subsystem

### YAMNet Architecture

YAMNet is based on MobileNetV1 and outputs:

- Frame-level scores: 521 audio event classes  
- Embeddings: 1,024-dimensional vectors  
- Spectrograms: Log-mel spectrograms (96 mel bands)  

---

### Audio Preprocessing

```python
def preprocess_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = audio / np.max(np.abs(audio))
    min_length = 0.96 * target_sr
    if len(audio) < min_length:
        audio = np.pad(audio, (0, int(min_length - len(audio))))
    return audio
```

---

### Feature Extraction with YAMNet

```python
import yamnet
import tensorflow_hub as hub

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

scores, embeddings, spectrogram = yamnet_model(audio)
audio_features = np.mean(embeddings, axis=0)
```

---

## 🔗 Multi-Modal Fusion

### Feature-Level Fusion Architecture

```text
Microwave Data → Statistical Features (~20 dims)
Acoustic Data → YAMNet Embeddings (1024 dims)
Concatenation (~1044 dims)
XGBoost / Neural Network
Unified Prediction:
• Healthy
• Tumor Suspected
• Infection Likely
```

---

### Implementation

```python
def fuse_and_predict(microwave_csv, audio_npy, model):
    mw_features = pd.read_csv(microwave_csv).values.flatten()
    audio_features = np.load(audio_npy)
    fusion_vector = np.concatenate([mw_features, audio_features])
    fusion_vector = fusion_vector.reshape(1, -1)
    prediction = model.predict(fusion_vector)
    confidence = model.predict_proba(fusion_vector).max()
    return prediction, confidence
```

---

## 📈 Results

### Acoustic Model Performance

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|----------|
| COPD       | 0.85      | 0.89   | 0.87     | 80       |
| Asthma     | 0.82      | 0.79   | 0.80     | 58       |
| Pneumonia  | 0.81      | 0.77   | 0.79     | 57       |
| Healthy    | 0.76      | 0.74   | 0.75     | 27       |
| Bronchial  | 0.73      | 0.71   | 0.72     | 21       |
| **Overall**| **0.82**  | **0.81** | **0.81** | **243** |

---

### Microwave Detection Results

| Tumor Size | Signal Drop (dB) | Detectability |
|------------|------------------|---------------|
| 1 cm       | 1.2 dB           | Marginal      |
| 2 cm       | 5.1 dB           | Clear         |
| 3 cm       | 8.3 dB           | Strong        |

Spatial sensitivity: ±2.1 dB variation with tumor position.

---

### Fusion Preliminary Results

| Modality         | Accuracy | Sensitivity | Specificity |
|------------------|----------|------------|-------------|
| Microwave Only   | 78%      | 81%        | 75%         |
| Acoustic Only    | 81%      | 79%        | 83%         |
| Fusion (simulated)| 87%     | 89%        | 85%         |

---

## 🔮 Future Work

### Short-Term (3–6 Months)

- Agar-based tumor phantoms  
- 6-antenna expansion  
- Fusion validation  
- Safety documentation (<0.1 mW power)  

### Medium-Term (6–12 Months)

- Clinical pilot  
- Real-time optimization (<2s inference)  
- 3D microwave reconstruction  
- Mobile app integration  

### Long-Term (1–2 Years)

- Multi-center validation  
- Health Canada Class II pathway  
- Custom PCB design  
- Global deployment  

---

## 📁 Repository Structure

```text
THORACIS AI/
├── README.md
├── requirements.txt
├── LICENSE
├── hardware/
├── software/
├── models/
├── data/
├── notebooks/
├── tests/
├── docs/
└── examples/
```

---

## 🤝 Contributing

### Ways to Contribute

- Code  
- Hardware  
- Data  
- Documentation  
- Validation  
- Outreach  

### Contribution Steps

```bash
git checkout -b feature/amazing-feature
git commit -m 'Add amazing feature'
git push origin feature/amazing-feature
```

Open a Pull Request.

---

## 📄 License

MIT License.

You may:

- Use commercially  
- Modify  
- Distribute  
- Use privately  
- Sublicense  

Include original copyright notice.

---

## 🙏 Acknowledgments

Mentors, institutional partners, open-source contributors, and dataset providers.

---

## 📚 Citations

Microwave imaging, acoustic analysis, safety standards, and health equity literature cited in project documentation.

---

## 📬 Contact

Project Lead: Anie Udofia  
William Aberhart High School, Calgary, AB  
GitHub: @HeavenlyCloudz  
Repository: github.com/HeavenlyCloudz/THORACIS AI  

---

## ⭐ Support

- Star the repository  
- Share the project  
- Contribute  
- Reach out for collaboration  

Together, we can make early cancer detection accessible to all.
