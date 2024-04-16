# PeptideBERT
Transformer Based Language Model for Peptide Property Prediction.
<br>
[Corresponding paper](https://arxiv.org/abs/2309.03099).
<img src="https://github.com/ChakradharG/PeptideBERT/assets/47364794/deba6f6d-8fdc-4262-a288-74b15f0543c4" alt="PeptideBERT" align="right" width="30%">

<br>

## Getting Started
* Clone this repository
* `cd PeptideBERT`
* Install the required packages (`pip install -r requirements.txt`)
* Download the datasets by running `python data/download_data.py`
* Run `python data/split_augment.py` to convert the data into the required format

<br>

## How to Use
* Update the `config.py` file with the desired parameters
* Optionally, to augment the data, use `data/split_augment.py` (uncomment the line that calls `augment_data`)
* Run `python train.py` to train the model

Note: For a detailed walkthrough of the codebase (including how to run inference using a trained model), please refer to `tutorial.ipynb`.
