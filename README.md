# Pseudoscience Detection Using a Pre-Trained Transformer Model with Intelligent ReLabeling

This repo contains the code used in the paper *Pseudoscience Detection Using a Pre-Trained Transformer Model with Intelligent ReLabeling*.

Authors:

- Thilina C. Rajapakse
- Ruwan D. Nawarathna

## Usage

### Setup

```python
conda create -n pseudo python pandas tqdm
conda activate pseudo
conda install pytorch cudatoolkit=10.1 -c pytorch
pip install simpletransformers
```

[Install Apex](https://github.com/NVIDIA/apex) if you are using fp16 training.

### Data Preparation

1. Create the dataset by following the instructions [here](https://github.com/ThilinaRajapakse/Pseudoscience-Dataset).
2. Prepare the data for Intelligent ReLabelling (IRL).
   1. Run `IRL_data_prep/create_data_dfs_by_source.py`.
   2. Run `IRL_data_prep/create_irl_datasets.py`.

### Training

1. Run `train_irl_models.sh`
2. Run `make_irl_predictions.sh`
3. Run `train_full_model.py`


_Please contact Thilina C. Rajapakse to receive the trained model for academic purposes._
