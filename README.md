# ⚡ ML and DL-Based Intelligent Detection, Classification, and Localization of Faults in Power Transmission Systems

![MATLAB](https://img.shields.io/badge/MATLAB-Simulink-orange)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![IEEE](https://img.shields.io/badge/Published-IEEE_ICIICS_2026-blue)
![Accuracy](https://img.shields.io/badge/Best_Accuracy-99.78%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📄 IEEE Publication

**"ML and DL-Based Intelligent Detection, Classification, and Localization of Faults in Power Transmission Systems"**
Achuoth Akol Achuoth Deng, Sharveshwaran S B, K. Manik Shivram, Narthana B, Angel T S, Sreelekshmi R S
Department of EEE, Amrita Vishwa Vidyapeetham, Amritapuri, India

**Published — 2026 3rd International Conference on Integrated Intelligence and Communication Systems (ICIICS), IEEE Bangalore Section**

🔗 [View on IEEE Xplore](https://ieeexplore.ieee.org/document/11483490)

---

## 📌 Overview
An intelligent fault diagnosis framework for three-phase power transmission
lines using RMS-based ML and DL models. Detects, classifies, and localizes
faults using only RMS voltage and current measurements — avoiding complex
waveform or time-frequency processing for low computational cost.

A dataset of **27,540 labeled samples** was generated using MATLAB/Simulink
covering 6 fault types, 9 fault resistances, and 10 fault distances
across a 300 km transmission line.

---

## 🎯 Three Classification Tasks

| Task | Best Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Fault Location | **LightGBM** | **99.78%** | **99.78%** | **99.78%** | **99.78%** |
| Fault Type | **CatBoost** | **85.86%** | **85.87%** | **85.86%** | **85.86%** |
| Phase Classification | **CatBoost** | **85.60%** | **85.60%** | **85.60%** | **85.60%** |

---

## 🧠 Models Compared

| Model | Type |
|---|---|
| Decision Tree | Classical ML |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |
| Dense Multi-task MLP | Deep Learning (TensorFlow) |

---

## 📊 Full Results

### Fault Location Classification
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **LightGBM** | **99.78%** | **99.78%** | **99.78%** | **99.78%** |
| XGBoost | 99.73% | 99.73% | 99.73% | 99.73% |
| Random Forest | 99.56% | 99.56% | 99.56% | 99.56% |
| CatBoost | 98.79% | 98.79% | 98.79% | 98.79% |
| Decision Tree | 98.67% | 98.67% | 98.67% | 98.67% |
| Dense MLP | 87.82% | 90.35% | 87.82% | 88.21% |

### Fault Type Classification
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **CatBoost** | **85.86%** | **85.87%** | **85.86%** | **85.86%** |
| XGBoost | 85.43% | 85.43% | 85.43% | 85.43% |
| LightGBM | 85.40% | 85.40% | 85.40% | 85.40% |
| Random Forest | 84.94% | 84.94% | 84.94% | 84.94% |
| Decision Tree | 84.87% | 84.87% | 84.87% | 84.87% |
| Dense MLP | 83.15% | 83.15% | 83.15% | 83.14% |

### Phase Classification
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **CatBoost** | **85.60%** | **85.60%** | **85.60%** | **85.60%** |
| Decision Tree | 85.48% | 85.48% | 85.48% | 85.48% |
| XGBoost | 85.43% | 85.43% | 85.42% | 85.42% |
| LightGBM | 85.14% | 85.14% | 85.14% | 85.14% |
| Random Forest | 85.14% | 85.14% | 85.14% | 85.14% |
| Dense MLP | 84.34% | 84.51% | 84.34% | 83.81% |

---

## 🔌 Power System Specifications
| Parameter | Value |
|---|---|
| System Rating | 100 MVA, 110 kV |
| Frequency | 50 Hz |
| Line Length | 300 km (distributed parameters) |
| Fault Types | NOFAULT, LG, LL, LLG, LLL, LLLG |
| Fault Distances | 10–270 km (10 locations) |
| Fault Resistance | 0, 0.01, 0.1, 1, 5, 7, 10, 50, 100 Ω |
| Location Zones | near_source / mid_line / near_load |
| Total Samples | 27,540 labeled samples |

---

## ⚙️ Methodology

### 1. Power System Modeling
3-phase 300 km transmission line modelled in MATLAB/Simulink.
Three-phase voltages and currents measured at relay point.

### 2. Dataset Generation
Automated using `generate_dataset_Project_sim.m` — simulates all
fault types, distances and resistances. Saves RMS features as CSV.

### 3. Feature Engineering (`main.py`)
35+ features extracted from RMS measurements:
- **Base:** Ia, Ib, Ic, Va, Vb, Vc
- **Power:** P1, P2, P3, P_total
- **Differences:** dV_ab, dV_bc, dV_ca, dI_ab, dI_bc, dI_ca
- **Zero-sequence:** I0, V0, I0_ratio, V0_ratio, ground_imbalance
- **Negative sequence:** neg_seq_approx
- **Statistical:** V_mean, V_std, I_mean, I_std
- **Interactions:** Va×Ia, Vb×Ib, Vc×Ic, log transforms

### 4. Model Training
- RobustScaler normalization (robust to outliers)
- 70:15:15 Train/Val/Test split (stratified)
- Optional SMOTE oversampling for imbalanced classes
- Per-class sample weights for DL training

### 5. Deep Multi-task MLP
Single TensorFlow model with 3 output heads:
- Head 1 → Fault Type classification
- Head 2 → Fault Location classification
- Head 3 → Phase classification

---

## 📁 Project Structure
```
fault-detection-power-transmission/
├── main.py                              # Full ML/DL training pipeline
├── requirements.txt                     # Python dependencies
├── matlab/
│   ├── Project_sim.slx                  # Simulink transmission model
│   ├── generate_dataset_Project_sim.m   # Automated dataset generator
│   ├── set_fault_type.m                 # Fault type configuration
│   └── fault_to_flags.m                 # Phase flag converter
├── dataset/
│   └── fault_data.csv                   # 27,540 labeled RMS samples
├── saved_models/                        # Trained .pkl & .h5 models
└── results/                             # Confusion matrices & plots
```

---

## 🚀 How to Run

### Step 1 — Generate Dataset (MATLAB)
```matlab
% Open MATLAB
% Open matlab/Project_sim.slx
% Run:
generate_dataset_Project_sim()
```

### Step 2 — Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train All Models
```bash
python main.py
```

### Output Files
```
results/
├── cm_FT_*.png               # Fault type confusion matrices
├── cm_FL_*.png               # Fault location confusion matrices
├── cm_PH_*.png               # Phase confusion matrices
├── dl_training_curves.png
└── model_ranking_improved.csv

saved_models/
├── FT_*.pkl                  # Fault type models
├── FL_*.pkl                  # Fault location models
├── PH_*.pkl                  # Phase models
└── dl_multitask_best.h5      # Deep learning model
```

---

## 📋 Dataset Features
| Feature | Description |
|---|---|
| Ia, Ib, Ic | Three-phase RMS currents (per unit) |
| Va, Vb, Vc | Three-phase RMS voltages (per unit) |
| fault_type | NOFAULT / LG / LL / LLG / LLL / LLLG |
| location | near_source / mid_line / near_load |
| A, B, C, G | Faulted phase flags (binary) |

---

## 👤 Authors
**Achuoth Akol Achuoth Deng**, Sharveshwaran S B, K. Manik Shivram,
Narthana B, Angel T S, Sreelekshmi R S
Department of EEE, Amrita Vishwa Vidyapeetham, Amritapuri, India
[LinkedIn](https://linkedin.com/in/achuoth-akol-achuoth-deng) · [GitHub](https://github.com/Achuoth11)
