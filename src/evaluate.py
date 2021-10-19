from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers.testbed_sampler import TestbedSampler
from easyfsl.methods import PrototypicalNetworks
from src.utils import build_model, create_dataloader


@click.option(
    "--specs-dir",
    help="Where to find the dataset specs files",
    type=Path,
    required=True,
)
@click.option(
    "--testbed",
    help="Path to the CSV defining the testbed",
    type=Path,
    required=True,
)
@click.option(
    "--trained-model",
    help="Path to an archive containing trained model weights",
    type=Path,
    required=True,
)
@click.option(
    "--output-dir", help="Where to dump evaluation results", type=Path, required=True
)
@click.option(
    "--device",
    help="What device to train the model on",
    type=str,
    default="cuda",
)
@click.command()
def main(
    specs_dir: Path, testbed: Path, trained_model: Path, output_dir: Path, device: str
):
    n_workers = 8

    logger.info("Fetching test data...")
    test_set = EasySet(specs_file=specs_dir / "test.json", training=False)
    test_sampler = TestbedSampler(
        test_set,
        testbed,
    )
    test_loader = create_dataloader(test_set, test_sampler, n_workers)

    logger.info("Retrieving model...")
    model = build_model(device=device, pretrained_weights=trained_model)

    logger.info("Starting evaluation...")
    results = model.evaluate(test_loader)

    output_file = output_dir / "raw_results.csv"
    results.to_csv(output_file)
    logger.info(f"Raw results dumped at {output_file}")


if __name__ == "__main__":
    main()
