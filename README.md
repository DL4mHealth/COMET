
# COMET

## Requirements

The recommended requirements are specified as follows:
* Python 3.10
* Jupyter Notebook
* scikit-learn==1.2.1  
* torch==1.13.1+cu116  
* matplotlib==3.7.1  
* numpy==1.23.5  
* scipy==1.10.1  
* pandas==1.5.3  
* wfdb==4.1.0  
* neurokit2==0.2.4

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

To get processed dataset, run notebooks in folder `data_processing/` for each raw dataset. The folder for processed datasets have two directories: `Feature/` and `Label/`. The folder `Feature/` contains file named in format `feature_ID.npy` files for all the patients. Each`feature_ID.npy` file contains trials belongs to the same patient and stacked into a 3-D array with shape [N, T, D], where N denotes # of trials, T denotes the timestamps for a trial, and C denotes the feature dimensions. 
The folder `Label/` has a file with name `label.npy`. This label file is a 2-D array with shape [N, 2], where N also denotes the # of patients. The first column is the label, and the second column is the patient ID, ranging from 1 to N. Finally, you can put processed dataset into `datasets/` folder in the following way:

* [AD dataset](https://osf.io/jbysn/) comprises EEG recordings from 12 patients with Alzheimer’s disease and 13 healthy controls. The processed data should be put into `datasets/AD/` so that each patient file can be located by `datasets/AD/Feature/feature_ID.npy`, and the label file can be located by `datasets/AD/Label/label.npy`.
* [PTB dataset](https://physionet.org/content/ptbdb/1.0.0/) consists of ECG recordings from 290 patients, with 15 channels sampled at 1000 Hz. The processed data should be put into `datasets/PTB/` so that each patient file can be located by `datasets/PTB/Feature/feature_ID.npy`, and the label file can be located by `datasets/PTB/Label/label.npy`.
* [TDBrain dataset](https://brainclinics.com/resources/) is a large dataset that monitors the brain signals of 1274 patients with 33 channels(500 Hz) during EC (Eye closed) and EO (Eye open) tasks. The processed data should be put into `datasets/TDBrain/` so that each patient file can be located by `datasets/TDBrain/Feature/feature_ID.npy`, and the label file can be located by `datasets/TDBrain/Label/label.npy`.


## Usage

We provide jupyter notebook example for each dataset. To train and evaluate COMET on a dataset, simply run DatasetName_example.ipynb, such as AD_example.ipynb. 
All the setups including pre-training, partial fine-tuning, full fine-tuning, and semi fine-tuning are running step by step with command in the notebook.

After training and evaluation, the pre-training model and fine-tuning model can be found in`test_run/DatasetName/`; and the logging file for validation and test results can be found in  `logging/DatasetName/`. 
You could also modify the working and logging directory in jupyter notebook.


## Overview
![Medical Time Series](https://i.ibb.co/Hgw0Kww/patient-data-structure-v2.png)  
**Structure of medical time series.** Medical time series commonly have four granularities (coarse to fine): patient, trial, sample, and observation. An observation is a single value in univariate time series and a vector in multivariate time series.

![COMET](https://i.ibb.co/Cn0CgWh/comet-framework-v6.png)
**Overview of COMET approach.** Capturing data consistency is crucial in the development of a contrastive learning framework . 
Data consistency refers to the shared commonalities preserved within the data, which provide a supervisory signal to guide model optimization. 
Contrastive learning captures data consistency by contrasting positive and negative data pairs, where positive pairs share commonalities and negative pairs do not. 
We propose data consistency across four levels of granularity, spanning from fine grained to coarse-grained. 
These levels encompass observation, sample, trial, and patient-level granularities. 
