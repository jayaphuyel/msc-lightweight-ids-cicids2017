# Lightweight Machine Learning Approaches for Network Intrusion Detection

MSc Cyber Security dissertation project — University of the West of Scotland

**Author:** Jaya Prasad Phuyel (B01831178)
**Supervisor:** Ahamed Tuani

## Overview

A controlled comparison of two lightweight classifiers — Logistic Regression
and Decision Tree — for network intrusion detection on the CICIDS2017 dataset.
The study measures both detection quality (per-class recall, F1, ROC-AUC,
false alarm rate) and computational cost (training time, prediction time,
memory use).

## Dataset

CICIDS2017 is **not** included in this repository due to its size.
Download the eight CSV files from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

Place them in the `data/` folder.

## Setup

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_baseline.ipynb` | Merge, clean, stratified split, baseline LR + DT (binary) |
| `02_feature_selection.ipynb` | Filter-based feature selection |
| `03_smote_multiclass.ipynb` | SMOTE (training folds only) + multiclass |
| `04_tuning.ipynb` | Grid Search, stratified 10-fold cross validation |
| `05_evaluation_cost.ipynb` | Final test-set evaluation + cost profiling |

## Reproducibility

Random seed fixed at 42 for the train/test split, SMOTE and model
initialisation. Cost measurements taken on an AMD Ryzen 5 5500U with
8 GB RAM running Windows 11.

## Structure