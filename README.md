This repository contains the implementation of the machine learning pipeline used in my diploma thesis entitled:

**“Binary Classification of Multi-Class Malware”** 

Diploma Thesis – Electrical and Computer Engineering  

University of Patras, Greece (2025)

---


## Project Overview

This repository contains my MSc thesis work on **multi-label malware behavior binary classification** using the MalDICT-Behavior dataset and EMBER feature vectors.


Starting from the official MalDICT resources, I:
Build a large-scale, sparse multi-label dataset for malware behaviors.
Apply label balancing and capping strategies to handle extreme class imbalance.
Train and compare several models:
- LightGBM (per-label baseline with calibration & tuned thresholds)
- Random Forest
- XGBoost
- CatBoost
- TabNet (Tabular Neural Network)
- 
Evaluate all methods on a temporal test split, reporting micro / macro / weighted Precision, Recall, F1-score, AUC.
All training / evaluation scripts and result files are included in this repository.

---

## Repository Structure

| File | Description |
|------|-------------|
| `01_vectorize_behavior_new.ipynb` | Feature vectorization and creation of sparse `.npz` behavior feature chunks |
| `02_train_ovr_new.ipynb` | Per-label LightGBM model training and validation threshold estimation |
| `03_train_eval_capped.ipynb` | Calibration, test evaluation, and computation of final performance metrics |
| `04_train_catboost_capped.ipynb`| Per-label  |
| `04_train_rf_capped.ipynb`|  |
| `04_train_tabnet_capped.ipynb`|  |
| `04_train_xgboost_capped.ipynb`|  |



---
## Dataset & Feature Pipeline

# MalDICT-Behavior
Based on MalDICT-Behavior, which provides malware samples annotated with one or more
behavior / category labels.
After cleaning and consistency checks, the working setup uses **64 active labels**.

# EMBER Feature Extraction

**1. Streaming JSONL**
Read EMBER metadata (*_features.jsonl) line-by-line.
Avoid loading the full dataset into memory at once.

**2. MD5 Filtering**
Build a keep_set of hashes from MalDICT-Behavior tag files.
Keep only samples whose md5 appears in this set.
Ensures every feature vector has valid behavior labels.

**3. Vectorization**
Use PEFeatureExtractor (EMBER v2) to convert each sample to a fixed-length feature vector.
Store the result as sparse .npz matrices in parts (e.g. 5k samples per file).

**4. Multi-Label Targets**
Normalize behavior tags.
Build a global label -> index mapping over 64 labels.
Encode each sample as a multi-hot vector.
Save Y in parts aligned 1:1 with the X parts and validate row alignment.

Result:
X_train, X_test: high-dimensional sparse EMBER features.
Y_train, Y_test: aligned multi-label targets for 64 behaviors.


---

## Requirements

Two Python environments are used:

| Purpose | Python version |
|---------|----------------|
| Feature vectorization | 3.6 |
| Model training & evaluation | 3.12 |

## Main dependencies:

lightgbm

numpy

scipy

pandas

scikit-learn

tqdm

--- 


The code should be executed in the order of the numbered notebooks.


## Current Results (Test Performance After Calibration)

| Metric | Micro | Macro | Weighted |
|--------|------:|------:|---------:|
| Precision | 0.5386 | 0.4626 | 0.6648 |
| Recall | 0.2926 | 0.3843 | 0.2926 |
| F1-score | 0.3792 | 0.3348 | 0.3685 |
| AUC | 0.9125 | 0.8648 | 0.8025 |

Interpretation:
- High discriminative capability (AUC above 0.90)
- Significant improvement in macro-level performance after calibration and optimized threshold selection

---


## Author

Georgios Konstantinos Ktenas  
Diploma Thesis – Electrical and Computer Engineering  
University of Patras  
2025

---

## License

Only the code in this repository is provided for academic and research purposes.  
The dataset is excluded and governed by its original license.

