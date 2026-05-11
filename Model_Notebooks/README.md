# Model Notebooks

**Run these notebooks in order: 01 → 02 → 03 → 04 → 05**

Each notebook saves its outputs to `artifacts/` so the next one can pick them up. Do not skip steps.

---

## Prerequisites

Before running anything, download the UNSW-NB15 dataset and place it at:
```
data/
└── UNSW_NB15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```
See the root [README](../README.md#datasets) for the download link.

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 01 — Data Preprocessing

**Reference:** Musthafa et al., IEEE Access 2025 | **Dataset:** UNSW-NB15

What it does:
- Loads UNSW-NB15 train/test CSVs
- Drops `id` and `attack_cat` columns
- Label-encodes categorical features: `proto`, `service`, `state`
- Applies `StandardScaler` to all numeric features
- Runs ANOVA F-test and selects the **top 36 features** (paper-aligned)
- Generates PCA 2D visualization of the selected feature space

**Config to set at the top of the notebook:**
```python
BASE_DIR   = "../"         # root of the project
TRAIN_PATH = "data/UNSW_NB15/UNSW_NB15_training-set.csv"
TEST_PATH  = "data/UNSW_NB15/UNSW_NB15_testing-set.csv"
```

**Outputs saved to `artifacts/`:**
| File | Shape | Description |
|---|---|---|
| `X_train.npy` | (82332, 36) float32 | Scaled training features |
| `X_test.npy` | (175341, 36) float32 | Scaled test features |
| `y_train.npy` | (82332,) int | Training labels |
| `y_test.npy` | (175341,) int | Test labels |
| `scaler.pkl` | — | Fitted `StandardScaler` |
| `label_encoders.pkl` | — | Fitted `LabelEncoder` for proto/service/state |
| `selected_features.json` | — | List of 36 ANOVA-selected feature names |
| `anova_feature_ranking.png` | — | Feature importance bar chart |
| `pca_visualization.png` | — | 2D PCA scatter (Normal vs Attack) |

---

## 02 — Model Training

What it does:
- Loads `X_train.npy`, `y_train.npy` from `artifacts/`
- Creates an 80/20 train/validation split (stratified)
- Builds a **lightweight MLP** (~5k parameters) designed for Raspberry Pi deployment
- Trains with `ReduceLROnPlateau`, `EarlyStopping`, and `ModelCheckpoint`
- Plots training/validation loss and accuracy curves

**Architecture:** Small feed-forward network on tabular flow features, input shape `(36,)`.

**Outputs saved to `artifacts/models/`:**
| File | Description |
|---|---|
| `best_model.keras` | Checkpoint with highest validation accuracy |
| `final_model.h5` | Model at final epoch |
| `final_model.keras` | Same, Keras v3 format |

**Training result:** ~96.5% validation accuracy, converges stably by ~60 epochs (see `artifacts/training_curves.png`).

---

## 03 — Pruning & Quantization

What it does:
- Loads the trained model from `artifacts/models/` (tries `IDSmodel.h5` → `final_model.h5` → `final_model.keras` → `best_model.keras` in order)
- Note: notebook 03 reshapes input to `(36, 1)` sequence format for an internal LSTM path — handled automatically
- Applies **magnitude-based structured pruning** (target sparsity 0.5) using TensorFlow Model Optimization Toolkit
- Converts to two TFLite variants:
  - **Float TFLite** — standard conversion, no precision loss
  - **INT8 TFLite** — post-training quantization with representative dataset
- Benchmarks all 4 variants (Baseline Keras, Pruned Keras, Float TFLite, INT8 TFLite) on accuracy, size, and latency

**Outputs saved to `artifacts/models/`:**
| File | Size | Description |
|---|---|---|
| `IDSmodel_pruned.h5` | 43.2 KB | Pruned Keras model (50% sparsity) |
| `IDSmodel_float.tflite` | 22.0 KB | Float32 TFLite |
| `IDSmodel.tflite` | **8.2 KB** | **INT8 quantized TFLite — primary deployment model** |

Also saves `artifacts/quantization_results.csv`.

---

## 04 — Evaluation

What it does:
- Loads all 4 model variants and evaluates on the held-out test set (`X_test.npy`, `y_test.npy`)
- Computes: Accuracy, F1-Score, ROC-AUC, confusion matrix, inference latency
- Auto-detects whether each model expects input shape `(36,)` or `(36, 1)` — no manual change needed
- Generates comparison plots: confusion matrices, ROC curves, size vs accuracy bar charts

**Outputs saved to `artifacts/`:**
| File | Description |
|---|---|
| `evaluation_summary.csv` | Full metrics table for all 4 models |
| `plots/confusion_matrices.png` | Side-by-side confusion matrices |
| `plots/roc_curves.png` | ROC curves for all models |
| `plots/size_accuracy_bars.png` | Size vs accuracy tradeoff |

**Key results:**

| Model | Accuracy | F1 | Size |
|---|---|---|---|
| Baseline Keras | 89.03% | 0.9134 | 96.9 KB |
| Pruned Keras | 91.52% | 0.9351 | 43.2 KB |
| Float TFLite | 91.52% | 0.9351 | 22.0 KB |
| **INT8 TFLite** | **93.08%** | **0.9514** | **8.2 KB** |

---

## 05 — Raspberry Pi Inference

**Reference:** Musthafa et al., IEEE Access 2025

What it does:
- Verifies TFLite runtime on the device (auto-detects `tflite-runtime` on Pi vs full `tensorflow` on laptop)
- Benchmarks `IDSmodel.tflite` (INT8) and `IDSmodel_float.tflite`: load time, per-sample inference (mean, P95, P99), CPU usage via `psutil`
- Runs full test-set evaluation to confirm on-device accuracy
- Simulates a **real-time Kafka-style streaming scenario** with 9 experiments varying producer count (1/2/3) and network delay (1.0/0.1/0.0 ms)
- Prints a deployment summary matching paper Table 7 format

**Inputs required (all generated by earlier notebooks):**
```
artifacts/X_test.npy
artifacts/y_test.npy
artifacts/models/IDSmodel.tflite
artifacts/models/IDSmodel_float.tflite
artifacts/evaluation_summary.csv
```

**To run on actual Raspberry Pi:**
```bash
pip install tflite-runtime psutil
# Copy artifacts/models/ and artifacts/*.npy to the Pi, then run this notebook
```
The notebook auto-switches to `tflite-runtime` if available — no code change needed.

**Outputs saved to `artifacts/`:**
| File | Description |
|---|---|
| `pi_benchmark_results.csv` | Load time, latency (mean/P95/P99), accuracy per model |
| `pi_stream_results.csv` | 9-experiment streaming simulation results |
| `pi_inference_log.txt` | Full deployment report |
| `pi_latency_distribution.png` | Latency histogram (INT8 vs Float) |
| `pi_cpu_usage.png` | CPU usage during INT8 inference |
| `pi_stream_simulation.png` | Sending time & inference time per experiment |
| `pi_confusion_matrix.png` | Confusion matrices for both TFLite models |

**Benchmark results (simulated Pi environment):**

| Model | Avg Inference | P95 | Accuracy | CPU |
|---|---|---|---|---|
| INT8 TFLite | 0.012 ms | 0.026 ms | 90.35% | 53.5% |
| Float TFLite | 0.014 ms | 0.030 ms | 91.52% | — |

Paper targets (actual Pi 3B+): 2.4 ms inference · 97.26% accuracy · 111.8% CPU

---

## Artifact Dependency Graph

```
UNSW-NB15 CSVs
     │
     ▼
[01_data_preprocessing]
     │
     ├── X_train/test.npy, y_train/test.npy
     ├── scaler.pkl, label_encoders.pkl
     └── selected_features.json
          │
          ▼
     [02_model_training]
          │
          └── models/best_model.keras, final_model.h5
               │
               ▼
          [03_pruning_quantization]
               │
               └── models/IDSmodel.tflite, IDSmodel_float.tflite, IDSmodel_pruned.h5
                    │
                    ▼
               [04_evaluation]
                    │
                    └── evaluation_summary.csv, plots/
                         │
                         ▼
                    [05_raspberry_pi_inference]
                         │
                         └── pi_benchmark_results.csv, pi_stream_results.csv, pi_*.png
```
