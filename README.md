# Open Query Set experiments
![Python Versions](https://img.shields.io/badge/python-3.8-%23EBBD68.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Code for the paper "Model-Agnostic Few-Shot Open-Set Recognition", under review.

# Get started

### Download models

This code uses several models, but most will be automatically downloaded. The only models you will need to download manually are:

1) [mini-ImageNet FEAT ResNet-12](https://drive.google.com/file/d/1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ/view) and [tiered-ImageNet FEAT ResNet-12](https://drive.google.com/file/d/1M93jdOjAn8IihICPKJg8Mb4B-eYDSZfE/view). Place them under under `data/models/feat/resnet_{dataset}_feat.pth`, with `dataset in [mini, tiered]`.

2) [mini-ImageNet pretrained ResNet-12 & WRN 28-10](https://drive.google.com/drive/folders/19TdjthkqMKLKSVHrbT5pVEmKvu-6-6iM) and [tiered-ImageNet pretrained ResNet-12 & WRN 28-10](https://drive.google.com/drive/folders/1y23iU6vW9ySsCn94XmlRs2z2FjSLMGPN). Place them under under `data/models/standard/{arch}_{dataset}_feat.pth`, with `arch in [resnet, wrn2810]` and `dataset in [mini, tiered]`.


### Download data

#### Mini-ImageNet

We still the same old version that you sent me. For TieredImageNet, to be continued...

#### Aircraft

Execute `make aircraft`

#### Fungi

Execute `make fungi`

#### CUB

Download from [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view). Unfortunately, the link now seems to be down. 

### Extract features

Because most methods work directly on top of extracted features, we factorize this part and extract features from each model once and for all, before runnning methods. In order to extract features for all splits, all datasets and all architectures, please run:
```
make extract_all
```
This operation make take quite some time.


### Reproduce results

All commands to reproduce any results in the paper is contained as a recipe in the `Makefile`. We detail in what follows every command:


#### Hyperparameter tuning (optional)

By default, the best hyperparameters found are already written down in the config files. If you wish to re-do the tuning pipeline from scratch, please following the following guidelines. First, run `make tuning` to run hyperparameter tuning on all methods on mini-ImageNet. To know which hyperparameters are being searched, refer to `configs/detectors.yaml`. Here is an example:

```
ABOD:
  default: {<shot>: {'n_neighbors': 5, 'method': 'fast'},
              }
  tuning: 
    hparams2tune: ['n_neighbors', 'method'] # name of the argument to tune
    hparam_values: {<shot>: [[3, 5], ['fast', 'default']], # for each argument, define the list of values over which to iterate. 
                    }
```
In the previous example, the tuning pipeline will perform a grid search over hyperparameters 'n_neighbors' and 'method', over the values [3, 5] and [fast, default] respectively. Note that `<shot>` key allows you to potentially define different sets of hyper-parameters according to the number of shots available. If you only write arguments for shot=1, it will just use these arguments, regardless of the shot.

Once the tuning is over, results will be logged to  `results/tuning/*`. You can directly log the best configs found for each methods (computed as the best trade-off between accuracy and ROCAUC) by executing `log_best_configs`.


#### Benchmark 

To reproduce the standard benchmarking Table (2 and 4) in the paper, please execute `make benchmark`.  Results will be saved to `results/benchmarks`. You can also directly log the results in a Markdown format by running `make log_benchmark`.


#### Cross-domain experiments (spider charts)

Run `make spider_charts`. Once this is done, run `make plot_spider_charts`, and go to `plots` to see the plots.


#### Model-agnostic (horizontal barplots)

Run `make model_agnosticity`. Once this is done, run `make plot_model_agnosticity`, and go to `plots` to see the plots.


### Additional features


#### Deploying code/data/models

The code support easy deployement to a remote server through `rsync`. If this can be helpful for you, please change the `SERVER_IP` variable and `SERVER_PATH` variables that respectively should contain the ip address of the remote server, and the path where the repo should be deployed. We let the reader refer to the Deployment / Imports part of the Makefile for all useful recipes.

#### Archiving / Restoring results

Once a set of results have been performed, you can archive them by running `make archive` and follow the subsequent instructions to reduce chances of losing them. This will store them in the `archive/` folder. To restore them back to `results/` at any time, run `make restore` and follow the instructions.





