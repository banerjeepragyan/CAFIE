# CAFIE: Counterfactually Aware Fair InferencE


## Setup
All requirements can be found in ```requirements.txt```. All packages can be installed by running
```bash
pip install requirements.txt
```
Counterfactual word lists are on ```act.txt```, ```cnt.txt```, and ```cnt2.txt```. These lists can be modified as and when required for more flexibility on debiasing. The following metrics have been used in the paper. Hyperparameters ```alpha``` and ```lambda``` and the vanilla model can be changed by changing the relevant command line arguments. 

## Metrics and Experiments
CAFIE can be evaluated on each of the metrics using-
1. StereoSet
```bash
python stereoset_scoring.py
```
2. CrowS-Pairs
```bash
python crows_scoring.py
```
4. BOLD
```bash
python bold_batched_scoring.py
python bold_get_scores.py
```
6. Fluency
```bash
python fluency_scoring.py
```

## Free form generation
CAFIE can be used for free form generation using
```bash
python generation.py
```
