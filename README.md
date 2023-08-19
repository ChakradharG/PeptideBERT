# PeptideBERT
Transformer Based Language Model for Peptide Property Prediction

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
* Run `python main.py` to train the model
