# ECG Anomaly Detection with an MLP Autoencoder

## Problem
Detect abnormal heartbeats in ECG recordings without needing to know every type of anomaly in advance.
An autoencoder is trained **only on normal beats**. At test time, beats it reconstructs poorly are flagged as anomalies.

- **Input:** one ECG heartbeat, 140 time steps.
- **Output:** a reconstruction error (the anomaly score) and a label, `normal` or `anomaly`.

## Dataset
ECG5000 (UCR Time Series Archive), from the BIDMC Congestive Heart Failure database. The version used here is the one hosted by TensorFlow:
`http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv`

- 4998 beats × 140 values. The last column is the label: `1` = normal, `0` = anomaly. The script recodes this so that anomaly = 1.
- 2919 normal and 2079 anomalous beats.
- The script downloads the data automatically, so you don't need to do anything manually.

## Model
A symmetric MLP autoencoder with layers `140 → 64 → 32 → 8 → 32 → 64 → 140`. It uses ReLU activations and a sigmoid output.
- Loss: L1 (MAE). Optimizer: Adam (lr = 1e-3). Batch size: 64. Epochs: 100. The weights with the lowest validation loss on normal beats are kept.
- Preprocessing: global min-max scaling, fitted on the training normals only.

## Evaluation procedure
1. Stratified split with seed 42: 70% train, 15% validation, 15% test. Only the **normal** beats from the training split are used for training.
2. Anomaly score: the mean absolute reconstruction error of each beat.
3. Threshold: the value that maximises F1 on the **validation** set.
4. Final metrics on the held-out **test** set: AUROC, AUPRC, accuracy, precision, recall, specificity, F1 and the confusion matrix.

## Results (test set, 750 beats, 312 anomalies)
| AUROC | AUPRC | Accuracy | Precision | Recall | Specificity | F1 |
|-------|-------|----------|-----------|--------|-------------|----|
| 0.980 | 0.954 | 0.961 | 0.944 | 0.965 | 0.959 | 0.954 |

Confusion matrix: TN = 420, FP = 18, FN = 11, TP = 301. Threshold = 0.0347.

## Installation
Tested with Python 3.14, running on CPU.
```bash
pip install torch==2.14.1 numpy==2.4.6 pandas==3.0.3 scikit-learn==1.9.0 matplotlib==3.11.0
```

## Usage
Train the model and evaluate it. This takes about 10 s on CPU.
```bash
python script.py train
```
Run inference on the supplied example inputs, which are 5 normal and 5 anomalous test beats:
```bash
python script.py predict
```
Run inference on new data. The file should be a CSV with no header, where each row is one beat of 140 raw values:
```bash
python script.py predict --input my_beats.csv
```

## Outputs (`outputs/`, figures as vector PDF)
| File | Content |
|------|---------|
| `model.pt` | Trained weights, scaling stats and threshold |
| `metrics.json` | Test metrics |
| `example_inputs.csv` | 10 test beats for running a prediction |
| `loss_curve.pdf` | Training and validation loss |
| `error_histogram.pdf` | Error distributions and the threshold |
| `roc_curve.pdf` | ROC curve |
| `confusion_matrix.pdf` | Confusion matrix on the test set |
| `reconstructions.pdf` | Input vs. reconstruction for a normal and an anomalous beat |

## Reproducibility
- Seed 42 is used for Python, NumPy, PyTorch and the data split.
- The model weights are not supplied. Running `python script.py train` produces them.
- Results may vary slightly on different hardware or library versions.

## Limitations
- ECG5000 contains pre-segmented, length-normalised beats from a single patient, so the results may not carry over to raw multi-patient ECG.
- The threshold is tuned with labelled validation anomalies. A purely unsupervised threshold, such as a percentile of the normal-beat error, would be more realistic.

## References
- Dau et al., *The UCR Time Series Classification Archive*, 2018.
- Goldberger et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation, 2000.
- TensorFlow tutorial "Intro to Autoencoders" (data hosting).

## Use of AI
Generative AI (Claude) was used to help format the charts and english corrections.
