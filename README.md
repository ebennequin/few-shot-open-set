# Open Query Set experiments
![Python Versions](https://img.shields.io/badge/python-3.8-%23EBBD68.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Research code for experiments on open query set

# Get started

## Download models

Download pre-trained ResNet-12 from [here](https://drive.google.com/drive/folders/14Jn1t9JxH-CxjfWy4JmVpCxkC9cDqqfE) and place the models under `data/models/standard/{arch}_{dataset}_{origin}.pth`. As examples,
 you should have the models `data/models/standard/resnet12_mini_imagenet_feat.pth` and `data/models/standard/wrn2810_mini_imagenet_feat.pth`.

## Data

### Mini-ImageNet

We still the same old version that you sent me. For TieredImageNet, to be continued...

### Aircraft


## Extract features

`make extract_standard` to get the feature on various backbones.


## Running OOD detection

### Adding a method to the code 

To add a new PyOD detector:

1. Modify `src/detectors/feature/__init__.py` and `src/detectors/feature/pyod_wrapper.py` following the template

2. Add its parameters to `configs/detectors.yaml` following the template:

```
ABOD:
  default: {<shot>: {'n_neighbors': 5, 'method': 'fast'},
              }
  tuning: 
    hparams2tune: ['n_neighbors', 'method'] # name of the argument to tune
    hparam_values: {<shot>: [[3, 5], ['fast', 'default']], # for each argument, define the list of values over which to iterate. 
                    }
```

Note that `<shot>` key allows you to potentially define different sets of hyper-parameters according to the number of shots available. If you only write arguments for shot=1, it will just use these arguments, regardless of the shot.

### Running a PyOD detector

Inspect the `run_pyod_detectors` recipe. To run with default parameters, run `make run_pyod_detectors`. To run a grid-search over parameters defined in `configs/detectors.yaml`, run `make TUNE=feature_detector run_pyod_detectors`


### Inspect results

Numeric results along with essential configuration parameters are stored under `results/{experiment_name}/{src}->{tgt}/{arch}/{shot}/out.csv`.



