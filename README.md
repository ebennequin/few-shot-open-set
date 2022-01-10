# Open Query Set experiments
![Python Versions](https://img.shields.io/badge/python-3.8-%23EBBD68.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Research code for experiments on open query set

# Get started

## Installation

1. Create a virtualenv with Python 3.8
2. `pip install -r dev_requirements.txt`

## Get outputs of DVC pipeline
Trained models and feature vectors computed with these models have been computed with the DVC
pipeline `dvc.yaml`.
They can be retrieved from the DVC remote without running the pipeline yourself.

1. Install AWS CLI
2. Ask Etienne for credentials.
3. Put your credentials in `~/.aws/credentials`
    ```
    [default]
    aws_access_key_id = {YOUR_ID}
    aws_secret_access_key = {YOUR_SECRET_KEY}
   ```
4. Run `dvc pull`

# Experiment
The main experiment file at this time is `notebooks/experiments_open_query_detection_on_features.py`.

It's made to infer outlier detection methods on feature vectors. Feature vectors are stored in pickle files
and follow this structure:

```python
{
  "label0": np.array((n_instances_for_this_label, feature_dimension)),
  "label1": np.array((n_instances_for_this_label, feature_dimension)),
  ...
}
```

## Normalization experiments (Malik)

Scripts in scripts/test_normalizations.sh

## Plot clusters with streamlit
To look at clusters run 
```bash
PYTHONPATH=. streamlit run notebooks/st_plot_clusters.py
```
