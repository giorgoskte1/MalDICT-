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
  
Evaluate all methods on a temporal test split, reporting micro / macro / weighted Precision, Recall, F1-score, AUC.
All training / evaluation scripts and result files are included in this repository.


---


## Repository Structure

* `data/`

  * Input data (MalDICT tags, EMBER metadata) and generated matrices (not all tracked in git).
* `vectorization/` and notebooks

  * Scripts / notebooks for converting EMBER JSONL → sparse feature matrices.
* `models_*` / `LightGBM_Benchmark/` / `04_train_*_capped.ipynb`

  * Training & evaluation for LightGBM, Random Forest, XGBoost, CatBoost, TabNet.
* `results/`

  * Saved metrics (CSV/JSON) for all models and threshold strategies.
* `README.md`

  * Overview and instructions.

Exact paths are documented inside each notebook.

---

## Dataset & Feature Pipeline

### MalDICT-Behavior

* Based on **MalDICT-Behavior**, which provides malware samples annotated with one or more
  behavior / category labels.
* After cleaning and consistency checks, the working setup uses **64 active labels**.

### EMBER Feature Extraction

1. **Streaming JSONL**

   * Read EMBER metadata (`*_features.jsonl`) line-by-line.
   * Avoid loading the full dataset into memory at once.

2. **MD5 Filtering**

   * Build a `keep_set` of hashes from MalDICT-Behavior tag files.
   * Keep only samples whose `md5` appears in this set.
   * Ensures every feature vector has valid behavior labels.

3. **Vectorization**

   * Use `PEFeatureExtractor` (EMBER v2) to convert each sample to a fixed-length feature vector.
   * Store the result as **sparse** `.npz` matrices in parts (e.g. 5k samples per file).

4. **Multi-Label Targets**

   * Normalize behavior tags.
   * Build a global `label -> index` mapping over 64 labels.
   * Encode each sample as a **multi-hot** vector.
   * Save `Y` in parts aligned **1:1** with the `X` parts and validate row alignment.

Result:

* `X_train`, `X_test`: high-dimensional sparse EMBER features.
* `Y_train`, `Y_test`: aligned multi-label targets for 64 behaviors.

---

## Handling Class Imbalance

MalDICT-Behavior is highly imbalanced (a few very frequent behaviors, many rare).

Key design choices:

* **No feature capping**: all feature dimensions are kept.
* **Label capping & balanced sampling**:

  * Count positives per label.
  * Apply caps on maximum positives to reduce dominance of very common labels.
  * For each per-label model:

    * Include all positives,
    * Sample negatives with a fixed ratio (e.g. 1:3 or 1:2),
    * Shuffle and split into train/validation.

This strategy is reused across LightGBM, Random Forest, XGBoost and CatBoost.

---

## Models

Below is a concise description of each family. Exact hyperparameters and
implementation details are in the corresponding notebooks.

### LightGBM (Per-Label Baseline)

LightGBM is used as a strong baseline.

* For each of the 64 labels:

  * Construct a balanced binary dataset (positives + sampled negatives).
  * Train a **LightGBM** classifier with early stopping on a validation split.
* After training:

  * Use validation predictions to perform **Platt scaling** (probability calibration).
  * Select an **optimal decision threshold per label**
    (via Precision–Recall analysis / β-tuning) to handle imbalance.
* On the temporal test set:

  * Apply per-label thresholds,
  * Compute micro / macro / weighted **Precision, Recall, F1, AUC**.
* Store:

  * Model files,
  * Calibration parameters,
  * Thresholds,
  * Aggregated metrics in `results/`.

### Random Forest (Per-Label)

* Same **one-vs-rest** setup as LightGBM:

  * One `RandomForestClassifier` per label,
  * Trained on the capped / balanced dataset for that label.
* Validation sets are used to monitor performance and tune simple thresholds.
* Final metrics on the temporal test set are saved for comparison.

Random Forest serves as a simpler ensemble baseline.

### XGBoost (Per-Label)

* Per-label binary classifiers using **XGBoost** (with sparse matrix support).
* For each label:

  * Train on capped / balanced data with early stopping.
* Threshold selection:

  * Scan validation probabilities and select the threshold that maximizes F1
    (optionally with β > 1 for recall-sensitive settings).
* On the test set:

  * Apply tuned thresholds,
  * Report micro / macro / weighted metrics.

XGBoost is a strong competitor to LightGBM and is evaluated under the same protocol.

### CatBoost (Per-Label, Threshold Tuning)

* Per-label **CatBoostClassifier** on the capped / balanced datasets.
* Observations:

  * Raw probabilities are often **not well calibrated**;
    a fixed 0.5 threshold is suboptimal for many labels.
* Approach:

  * For each label, perform **threshold tuning** on validation predictions:

    * choose the threshold that maximizes F1 (or Fβ),
    * especially useful for rare behaviors.
* Tuned thresholds are then applied on test predictions.
* Final micro / macro / weighted metrics are stored in `results/`.

With proper thresholding, CatBoost becomes competitive with the other GBDT models
and highlights the importance of decision thresholds in multi-label malware tasks.

### TabNet (Multi-Task Neural Network)

TabNet is used as a neural baseline.

* Use `TabNetMultiTaskClassifier` to predict all 64 labels jointly.
* Constraints:

  * TabNet requires **dense** inputs; sparse EMBER features must be densified.
  * Full dense training on the entire dataset is too expensive on CPU.
* Practical setup:

  * Train on a **documented random subset** of the training data
    (e.g. 300k train / 60k validation samples).
  * Use a **regularized configuration**:

    * moderate `n_d`, `n_a`, `n_steps`,
    * `lambda_sparse` and `weight_decay`,
    * early stopping on validation AUC,
    * `compute_importance=False` to avoid full-dataset explanations.
  * Save the multi-task model for reuse.
* Evaluation:

  * Predict probabilities for all labels on validation and test sets.
  * Use a global threshold (0.5) and investigate alternative global thresholds.
  * Compute micro / macro / weighted metrics on the temporal test set.

Under these realistic constraints, TabNet does **not** outperform the tree-based models
on this sparse, high-dimensional malware dataset, but provides a meaningful neural baseline
and shows the trade-offs of deep tabular models at MalDICT scale.

---

## How to Run 

This is a brief outline; see the notebooks for exact commands and paths.

1. **Set up environments**

   * Python environment for EMBER feature extraction.
   * Python environment for model training (LightGBM, XGBoost, CatBoost, TabNet).

2. **Download data**

   * MalDICT-Behavior tag files.
   * EMBER metadata for the corresponding samples.
   * Update paths in notebooks / scripts.

3. **Vectorize features**

   * Run the vectorization notebook:

     * stream EMBER JSONL files,
     * filter by MalDICT hashes,
     * write sparse `.npz` matrices for train/test.

4. **Build labels**

   * Generate multi-hot label matrices (`Y_train`, `Y_test`) for the 64 behaviors.
   * Ensure consistent indexing and alignment with `X`.

5. **Train models**

   * Run `04_train_lightgbm_capped.ipynb` (or equivalent).
   * Run corresponding `04_train_*_capped.ipynb` for:

     * Random Forest,
     * XGBoost,
     * CatBoost,
     * TabNet.

6. **Evaluate & collect results**

   * Each notebook saves metrics into `results/`
     (per-label and aggregated micro/macro/weighted scores).
   * Use these files directly for tables and plots.

---

## Notes

* All experiments use the **temporal split** of MalDICT-Behavior
  (train on earlier samples, test on later samples) to better simulate deployment.
* The setup is **multi-label**, **imbalanced**, and **large-scale** by design.
* Threshold selection and probability calibration are treated as integral parts
  of the pipeline, not post-processing.

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
Please check:

* The license of this repository, and
* The license/terms of the upstream [MalDICT](https://github.com/rjjoyce/MalDICT) project

before redistributing data or models.













