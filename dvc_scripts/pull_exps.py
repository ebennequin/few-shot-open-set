import click
from dvc.repo import Repo
from loguru import logger


@click.option(
    "-r",
    "--remote",
    type=str,
    default="origin",
)
@click.option(
    "--list-all",
    type=bool,
    default=False,
)
@click.command()
def main(remote, list_all):
    current_repo = Repo(".")
    experiments = current_repo.experiments.ls(git_remote=remote, all_=list_all)
    for exp_list in experiments.values():
        for exp_name in exp_list:
            current_repo.experiments.pull(remote, exp_name, pull_cache=True)
            logger.info(f"Pulled experiment {exp_name} from {remote}.")


if __name__ == "__main__":
    main()
