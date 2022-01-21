import click
from dvc.repo import Repo
from loguru import logger


@click.option(
    "-r",
    "--remote",
    type=str,
    default="origin",
)
@click.command()
def main(remote):
    current_repo = Repo(".")
    experiments = current_repo.experiments.ls()
    for exp_list in experiments.values():
        for exp_name in exp_list:
            current_repo.experiments.push(remote, exp_name, push_cache=True)
            logger.info(f"Pushed experiment {exp_name} to {remote}.")


if __name__ == "__main__":
    main()
