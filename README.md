# Open Query Set experiments
![Python Versions](https://img.shields.io/badge/python-3.8-%23EBBD68.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Research code for experiments on open query set

## Installation

1. Create a virtualenv with Python 3.8
2. `pip install -r dev_requirements.txt`

### Paths to datasets with EasySet
Paths to images are defined in specification files such as [this one](data/tiered_imagenet/specs/train.json).
All images are expected to be found in `data/{dataset_name}/images`. For instance,
for tieredImageNet we expect a structure like this one:
```
data
|
|----tiered_imagenet
|    |
|    |----images
|    |    |
|    |    |----n04542943
|    |    |----n04554684
|    |    |----...
```
If you can't host the data there for any reason, you can create a symlink:
```bash
ln -s path/to/where/your/data/really/is data/tiered_imagenet/images
```

### Retrieving trained models

Run `dvc pull` to retrieve experiment outputs from the DVC remote.
