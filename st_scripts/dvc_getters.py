import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List

import dvc.api
import pandas as pd
import streamlit as st
import yaml
from dvc.repo.get import get
from tensorboard import program

from st_constants import (
    DVC_REPO,
    PARAMS_FILE,
    METRICS_FILE,
    TENSORBOARD_LOGS_DIR,
    TENSORBOARD_CACHE_DIR,
)
from st_utils import get_hash_from_list


@st.cache
def read_params(rev: str) -> Dict:
    with dvc.api.open(PARAMS_FILE, rev=rev) as file:
        params = yaml.safe_load(file)

    return params


@st.cache
def read_metrics(rev: str) -> Dict:
    with dvc.api.open(METRICS_FILE, rev=rev) as file:
        evaluation_metrics = json.load(file)

    return evaluation_metrics


def get_params(
    exp_list: List[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.json_normalize(read_params(exp), sep=".").assign(exp_name=exp)
            for exp in exp_list
        ]
    ).set_index("exp_name")


def get_metrics(exp_list: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"exp_name": exp, **read_metrics(exp)} for exp in exp_list]
    ).set_index("exp_name")


@st.cache
def get_all_exps() -> pd.DataFrame:
    all_dvc_exps_dict = DVC_REPO.experiments.ls(all_=True)
    return pd.DataFrame(
        [
            (exp_name, parent_commit, DVC_REPO.experiments.scm.resolve_rev(exp_name))
            for parent_commit in all_dvc_exps_dict
            for exp_name in all_dvc_exps_dict[parent_commit]
        ],
        columns=["exp_name", "parent_hash", "commit_hash"],
    ).set_index("exp_name")


def download_dir(path, git_rev, out):
    get(
        url=".",
        path=path,
        out=out,
        rev=git_rev,
    )


@st.cache
def download_tensorboards(exps: Dict[str, str]) -> str:
    # TODO ici ça va chercher la donnée sur le remote à chaque fois. Ca fait beaucoup de download de donnée, et beaucoup de duplication car chaque combinaison de commits donne un nouveau cache custom.
    # TODO je pense qu'il faut faire en 2 étapes (si on considère qu'on peut que aller les chercher sur le remote et pas dans le cache local) :
    # TODO 1) download toutes les expériences manquantes dans streamlit_cache/tensorboard/all_exps
    # TODO 2) créer un dossier streamlit_cache/tensorboard/hash(selected_exps) avec dedans des symlinks vers les exps qu'on veut
    cache_dir = TENSORBOARD_CACHE_DIR / get_hash_from_list(list(exps.keys()))
    if not cache_dir.exists():
        for exp, git_rev in exps.items():
            download_dir(
                path=str(TENSORBOARD_LOGS_DIR),
                git_rev=git_rev,
                out=str(cache_dir / exp),
            )

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(cache_dir)])
    return tb.launch()


@st.cache
def read_csv(path: Path, exp: str) -> pd.DataFrame:
    with dvc.api.open(path, rev=exp, mode="r") as file:
        df = pd.read_csv(file, index_col=0)
    return df


@st.cache
def get_image(path: Path, exp: str):
    with dvc.api.open(path, rev=exp, mode="rb") as file:
        image = plt.imread(file)
    return image
