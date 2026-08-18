Lightweight Machine Learning Approaches for Network Intrusion Detection

MSc Cyber Security dissertation project (University of the West of Scotland).

This project develops and compares two lightweight classifiers — Logistic Regression and a Decision Tree — for network intrusion detection on the CICIDS2017 dataset. Both models are evaluated on detection quality (macro-averaged F1 and per-class recall) and on computational cost (training time, inference speed, model size and peak memory), so that the "lightweight" claim is measured rather than assumed.

Headline result: the Decision Tree reaches a macro-F1 of 0.9488 against the Logistic Regression's 0.7670, while also being faster to train, faster at inference, smaller on disk and lighter on memory.

Dataset

This repository does not include the CICIDS2017 data (it is large and licensed). Download the eight CSV traffic files from the Canadian Institute for Cybersecurity:

https://www.unb.ca/cic/datasets/ids-2017.html

Place the CSV files in a local data/ folder (this folder is gitignored). Notebook 01_baseline.ipynb shows exactly where the files are expected.

Environment
Python 3.13.2

Install the dependencies (pinned to the versions used in the study):

pip install -r requirements.txt
How to reproduce

Run the notebooks in order:

notebooks/01_baseline.ipynb — data consolidation, cleaning and an initial binary baseline
notebooks/02_feature_selection.ipynb — mutual-information ranking and feature-subset benchmark
notebooks/03_smote_multiclass.ipynb — leakage-safe SMOTE oversampling experiment
notebooks/04_tuning.ipynb — cross-validated grid search (scored on macro-F1)
notebooks/05_evaluation.ipynb — final evaluation on the held-out test set and cost profiling

The two heaviest tuning searches are also provided as standalone scripts:

python src/dt_tuning.py
python src/lr_tuning.py
Repository layout
notebooks/ — the numbered analysis notebooks (01–05)
src/ — standalone tuning scripts for the Decision Tree and Logistic Regression
results/ — saved best-parameter JSONs, tuning logs and confusion-matrix images
figures/ — figures used in the dissertation (ML vs DL, mutual information, feature trade-off)
Key saved outputs
results/dt_tuning_best.json, results/lr_tuning_*_best.json — tuned hyper-parameters and CV scores
results/dt_final_confusion_matrix.png, results/lr_final_confusion_matrix.png — test-set confusion matrices
results/lightweight_comparison.png — cost comparison of the two models
Author

Jaya Prasad Phuyel — MSc Cyber Security, University of the West of Scotland.