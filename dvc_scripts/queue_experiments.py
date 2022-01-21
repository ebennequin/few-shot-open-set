import subprocess
from pathlib import Path
from typing import List, Dict

import click
import yaml


@click.option(
    "-f",
    "--file",
    help="YAML file containing grid search params.",
    type=Path,
    default=Path("grid.yaml"),
)
@click.option(
    "-p",
    "--pipeline",
    help="Root directory of the pipeline we want to launch.",
    type=Path,
    default=Path("pipelines") / "inference",
)
@click.command()
def main(file, pipeline):
    """
    Queue DVC experiments parameterized with a YAML file using this template:

        untracked_files:
          - untracked/file/a
          - untracked/file/b

        grid:
          - param1: value1a
            param2: value2a
          - param1: value1b
            param3: value3a

    Untracked files are files that are not tracked by Git nor DVC.
    They will be staged before queuing the experiments in order to be
    accessible during the experiment.
    The untracked_files key is optional.
    """
    with open(file, "r") as stream:
        grid_params = yaml.safe_load(stream)

    for experiment in grid_params["grid"]:
        if "untracked_files" in grid_params:
            index_files(grid_params["untracked_files"])
        queue(experiment, pipeline)


def index_files(untracked_files: List[str]):
    """
    Index a list of files with Git. Use force.
    Args:
        untracked_files: list of untracked files
    """
    subprocess.run(
        ["git", "add", "-f"] + [filepath for filepath in untracked_files]
    )


def queue(experiment: Dict, pipeline: Path):
    """
    Queue an experiment with DVC.
    Args:
        experiment: dictionary where the keys are the parameters to be updated
            and the values are the value with which to update the parameter.
        pipeline: root directory of the pipeline we want to launch
    """
    command_line = ["dvc", "exp", "run", str(pipeline / "dvc.yaml"), "--queue"]
    for param, value in experiment.items():
        command_line += [
            "--set-param",
            f"{pipeline}/params.yaml:{param}={value}",
        ]

    subprocess.run(command_line)


if __name__ == "__main__":
    main()
