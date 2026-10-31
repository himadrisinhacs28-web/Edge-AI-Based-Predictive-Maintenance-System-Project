# Predictive Maintenance of Industrial Motors Using Vibration and Acoustic Sensor Fusion

## Project Overview

This project develops a software-based predictive maintenance system for industrial motors using vibration and acoustic sensor data. Machine learning and deep learning models are used to classify motor operating conditions and detect potential faults. A lightweight TensorFlow Lite version of the best model was also prepared as a first step toward edge deployment.

## Problem Statement

Unexpected motor failures cause production downtime and maintenance costs. Predictive maintenance helps identify abnormal motor conditions before major failures occur.

## Objectives

- Analyze vibration and acoustic sensor data.
- Identify different motor operating conditions.
- Develop individual-sensor and sensor-fusion deep learning models.
- Evaluate and compare model performance.
- Explore lightweight model deployment using TensorFlow Lite.

## Dataset

**MAFAULDA** (Machinery Fault Database), sampled at 50 kHz.

Four of the six available conditions are used (overhang and underhang faults are excluded):

| Label | Condition |
|---|---|
| 0 | Normal |
| 1 | Imbalance |
| 2 | Horizontal Misalignment |
| 3 | Vertical Misalignment |

**Signals used:** three accelerometer axes (columns 4, 5, 6) and the microphone (column 7).
**Preprocessing:** each recording is split into non-overlapping windows of 1024 samples.

## Technologies Used

Python, Google Colab, NumPy, Pandas, SciPy, Scikit-learn, TensorFlow / Keras, TensorFlow Lite, Streamlit

## Models Implemented

- Acoustic CNN
- Vibration CNN
- Random Forest (time-domain and time + FFT features)
- Sensor Fusion CNN (two-branch 1D CNN, feature-level fusion of vibration and acoustic inputs)

## Results

| Model | Test accuracy |
|---|---|
| Vibration CNN (single axis) | ~46% |
| Random Forest (10 features, single axis) | 45.37% |
| Acoustic CNN | 65.95% |
| Random Forest (27 time-domain features, 3 axes) | 62.55% |
| Random Forest (57 time + FFT features, 3 axes) | 78.42% |
| **Sensor Fusion CNN** | **97.03%** |

**Sensor Fusion CNN, per-class results (42,944 test windows from 176 held-out files):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.89 | 0.98 | 0.94 |
| Imbalance | 0.99 | 0.97 | 0.98 |
| Horizontal Misalignment | 0.99 | 0.92 | 0.96 |
| Vertical Misalignment | 0.95 | 1.00 | 0.97 |

Macro F1: 0.96.

### Notes on the comparison

- The Acoustic CNN, Vibration CNN and Random Forest baselines were trained and evaluated on a smaller subset of the data (30 files per class, 24 test files). The Sensor Fusion CNN used all 880 files with a separate file-level split. The baselines and the fusion model are therefore not evaluated on an identical test set, and part of the improvement may come from the larger training set.
- The test set is class-imbalanced (Normal: 2,440 windows vs. Imbalance: 16,348). Normal precision (0.89) is the weakest metric: about 11% of windows predicted as Normal are actually faulty, i.e. missed faults, which matters most in a maintenance setting.

## Edge Deployment (TensorFlow Lite)

| Version | Size |
|---|---|
| Float32 TFLite | ~388 KB |
| Quantized TFLite | ~119 KB |

Average inference time of the float32 TFLite model was ~0.78 ms per window, measured on Google Colab CPU. Accuracy of the quantized model has not been evaluated on the full test set.

## Limitations

- **Dataset-based only.** Results come from an existing dataset and do not represent validation on a physical industrial motor.
- **Software-only implementation.** No physical sensors or microcontrollers were used. TFLite conversion and timing were tested in Google Colab only, not on an edge device.
- **Generalization.** Train/validation/test splits are by file, which avoids window-level leakage. However, test files may come from fault severities and rotation speeds similar to those in training, so accuracy on unseen operating conditions may be lower than reported.
- **Limited fault classes.** Only four of the six MAFAULDA conditions are covered.

## Future Scope

- Evaluate on held-out fault severities and RPM ranges.
- Re-run baselines on the same split as the fusion model for a fair comparison.
- Evaluate the quantized TFLite model on the full test set.
- Improve the Streamlit dashboard.
- Benchmark inference on edge hardware.
- Explore physical sensor and microcontroller integration.
