# Open Query Set experiments
![Python Versions](https://img.shields.io/badge/python-3.8-%23EBBD68.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Research code for experiments on open query set

# Get started

## Installation

1. Create a virtualenv with Python 3.8
2. `pip install -r dev_requirements.txt`

## Get outputs of DVC pipeline
Trained models and feature vectors are versioned with DVC. 
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

# Experiment flows

## Feature extraction

The feature extraction pipeline is located at `pipelines/compute_features`. 
To change the feature extraction process (and therefore update the features versioned with DVC
and located at `data/features`):

1. Create a new branch
2. Experiment
3. Commit your clean changes
4. Run the feature extraction pipeline: `dvc repro pipelines/compute_features/dvc.yaml`
5. Make a pull request on `master`

## Run an experimental inference

The inference pipeline is located at `pipelines/inference`. It contains 3 stages:

- `detect_outliers` run an outlier detection inference 
and dumps the model's predictions at `data/predictions/{dataset}/outliers.csv`
- `classify_queries` run a classification inference 
and dumps the model's predictions at `data/predictions/{dataset}/classifications.csv`
- `compute_metrics` compute detection and classification metrics from the previous stages'
predictions.

This pipeline depends on the feature vectors versioned with DVC and located 
at `data/features/{dataset}.pickle`.

### Running a simple inference with parameter or code changes

1. Create a new branch
2. Do my changes
3. `dvc exp run pipelines/inference/dvc.yaml`
   
   (You can add `--set-param pipelines/inference/params.yaml:<param_name>=<param_value>` 
if you don't want to change the params directly in the `params.yaml` file.)
4. Once your experiment is complete, you can:
   1. display metrics: `dvc exp show`
   2. apply the experiment on your current branch with `dvc exp apply <exp-name>`
5. Make a pull request on `master`

### Adding a new method
Whether it's a detector, a classifier or a feature transformer, you can add a new method by:

1. Creating a new class inheriting the corresponding abstract class
2. (Optional) Adding new parameters to `pipelines/inference/params.yaml`
3. Experiment your new method by changing the corresponding parameter in `pipelines/inference.params.yaml`

Ex: 
```yaml
detector: MyNewDetector
detector_args:
  # Those params were already there
  n_neighbors: 3
  method: mean
  # I'm adding this
  param_for_my_new_method: 42
```

### Running a grid of experiments

You can create several experiments and run them from the same commit to easily compare
several sets of parameters.

1. Create a `grid.yaml` file defining the grid:
   
   ```yaml
   grid:
      - detector: KNNOutlierDetector
        prepool_transformers:
          - BaseSetCentering
          - TransductiveBatchNorm
      - detector: KNNOutlierDetector
        prepool_transformers:
          - BaseSetCentering
      - detector: LOFOutlierDetector
        prepool_transformers:
          - BaseSetCentering
          - TransductiveBatchNorm
   ```
   
2. Queue a list of experiments from that grid using our custom script:

   `python -m dvc_scripts.queue_experiments`
3. Run all experiments

   `dvc exp run --run-all --jobs 4` for 4 concurrent runs
4. When all experiments are done, compare their metrics with

   `dvc exp show` (see [here](https://dvc.org/doc/command-reference/exp/show) for more doc)
   
   You can add a `--csv` flag to print in CSV format for easy export. 
   Check out [the doc](https://dvc.org/doc/command-reference/exp/show) for more cool things.

5. Share your experiments on the remote with `python -m dvc_scripts.push_exps`

6. If you'd like one experiment to become the default setting (our new SOTA):
   
   `dvc exp apply <exp-name>`

## Troubleshooting

Stages' dependencies are declarative and we didn't add all `src` to the dependency, 
so if you change a `utils` file and re-run the pipeline, there is a change that DVC will
not run it and just recover cache because it will consider that everything is the same.
Use `--force` to force reexecution.

# Streamlit
To look at clusters run 
```bash
PYTHONPATH=. streamlit run notebooks/st_plot_clusters.py
```
