# IN6227 Assignment 1

Variant 1: logistic regression and random forest classification.

## Run

Requires Python 3.12 and the supplied `train.csv` / `test.csv` files.

```sh
python -m pip install -r requirements.txt
python run_experiment.py --data-dir /path/to/dataset
python verify_results.py
```

## Structure

- `run_experiment.py`: experiment workflow.
- `data.py`: loading, cleaning and data audit.
- `modeling.py`: preprocessing, tuning, evaluation and bootstrap.
- `verify_results.py`: independent metric checks.
- `results/`: configurations, CV scores and test results.

## Method and results

Five-fold stratified CV selects parameters by average precision (AP), with seed
6227. Preprocessing is fitted within each fold; test data are evaluated after
selection. Predictions use a fixed 0.5 threshold.

| Model | Test AP | F1 | Recall |
|---|---:|---:|---:|
| Logistic regression | 0.7025 | 0.6286 | 0.5713 |
| Random forest | 0.7087 | 0.6243 | 0.5600 |

RF has slightly higher AP, but the paired 95% bootstrap interval for the
difference includes zero. LR has higher recall and F1.

The dataset is not included. Predictions and fold assignments are generated
locally. Dependency versions and input hashes are recorded in the results.
