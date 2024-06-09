# All Should Be Equal in the Eyes of Language Models: Counterfactually Aware Fair Text Generation

## Setup
All requirements can be found in ```requirements.txt```. All packages can be installed by running
```bash
pip install -r requirements.txt
```
Counterfactual word lists are on ```list_1.txt```, ```list_2.txt```, and ```list_3.txt``` in the folder ```data\word_lists```. These lists can be modified as and when required for more flexibility on debiasing. The following metrics have been used in the paper. Hyperparameters ```alpha``` and ```lambda``` and the vanilla model can be changed by changing the relevant command line arguments. 

## Metrics and Experiments
To evaluate CAFIE on StereoSet, run
```bash
python cafie_stereoset.py
```

## Free form generation
CAFIE can be used for free form generation using
```bash
python generation.py
```

If you use our work, please cite using the following:
```
Banerjee, P., Java, A., Jandial, S., Shahid, S., Furniturewala, S., Krishnamurthy, B., & Bhatia, S. (2024, March). All Should Be Equal in the Eyes of LMs: Counterfactually Aware Fair Text Generation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 16, pp. 17673-17681).
```
