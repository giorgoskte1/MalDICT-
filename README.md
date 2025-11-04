This repository contains the implementation of the machine learning pipeline used in my diploma thesis entitled:
**“Binary Classification of Multi-Class Malware”** 
Diploma Thesis – Electrical and Computer Engineering  
University of Patras, Greece (2025)

---

## Project Overview

The goal of this research is to detect malware and classify samples into multiple behavioral categories using static behavior features extracted from Portable Executable (PE) files.

Since malware can exhibit more than one malicious behavior simultaneously, the task is formulated as a **multi-label binary classification problem**. Each classifier decides independently whether a specific malware behavior is present or not in a sample.

Main methodological components:
- One-vs-Rest (OvR) training with LightGBM binary models per label
- Handling high class imbalance through per-label sampling strategies
- Platt scaling for probability calibration
- Per-label decision threshold optimization based on validation performance
- Macro-metric oriented evaluation to ensure fair performance across all behaviors

---

## Repository Structure

| File | Description |
|------|-------------|
| `01_vectorize_behavior_new.ipynb` | Feature vectorization and creation of sparse `.npz` behavior feature chunks |
| `02_train_ovr_new.ipynb` | Per-label LightGBM model training and validation threshold estimation |
| `03_train_eval_capped.ipynb` | Calibration, test evaluation, and computation of final performance metrics |

---

## Dataset

The project uses the **MalDICT behavior subset** derived from the EMBER benchmark dataset.

Dataset Source: https://github.com/elastic/ember

The dataset itself is not included in this repository due to size and licensing restrictions.

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

