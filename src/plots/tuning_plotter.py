# %%
from pathlib import Path

import pandas as pd
import numpy as np

import typer as typer
from matplotlib import pyplot as plt
from matplotlib import collections


def main(
    exp_name: str,
    n_shot: int = 1,
):
    tuned_params = ["inference_steps", "lambda_s", "lambda_z"]

    results = pd.read_csv(
        Path("results")
        / exp_name
        / "mini_imagenet-->mini_imagenet(val)/resnet12/feat"
        / str(n_shot)
        / "out.csv"
    )

    clean_results = results.join(
        pd.DataFrame(
            results.feature_detector.str.findall(
                r"([^\r\n\t\f\v\=\(\)\,]+)\=([^\r\n\t\f\v\=\(\)\,]+)"
            )
            .apply(lambda x: dict(x))
            .to_list()
        ).astype(float)
    )[
        [
            *tuned_params,
            "mean_acc",
            "mean_rocauc",
        ]
    ].loc[
        lambda df: df.inference_steps == 2
    ][
        ["lambda_s", "lambda_z", "mean_acc", "mean_rocauc"]
    ]

    acc_df = clean_results.pivot(
        index="lambda_s", columns="lambda_z", values="mean_acc"
    )
    rocauc_df = clean_results.pivot(
        index="lambda_s", columns="lambda_z", values="mean_rocauc"
    )

    min_value = min(acc_df.min().min(), rocauc_df.min().min()) - 0.05
    max_value = max(acc_df.max().max(), rocauc_df.max().max())

    plt.rcParams.update(
        {
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "font.size": 20,
        }
    )

    fig, ax = plt.subplots()
    im1 = ax.imshow(acc_df.values, cmap="Blues", vmin=min_value, vmax=max_value)
    im2 = triamatrix(
        rocauc_df.values, ax, rot=90, cmap=im1.cmap, vmin=min_value, vmax=max_value
    )

    # Fill values in heatmap
    for i in range(4):
        for j in range(4):
            ax.text(
                j - 0.15,
                i - 0.25,
                f"{100 * acc_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=18,
                color="w",
            )
            ax.text(
                j + 0.15,
                i + 0.35,
                f"{100 * rocauc_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=18,
                color="w",
            )

    ax.set_xticks([0, 1, 2, 3], [0.01, 0.05, 0.1, 0.5])
    ax.set_yticks([0, 1, 2, 3], [0.01, 0.05, 0.1, 0.5])
    ax.set_xlabel(r"$\lambda_z$", fontsize=25)
    ax.set_ylabel(r"$\lambda_\xi$", rotation=0, fontsize=25)
    ax.yaxis.set_label_coords(-0.1, 0.47)
    ax.xaxis.set_label_coords(0.5, -0.05)
    fig.set_size_inches(6, 6)
    plt.subplots_adjust(left=0.16, top=0.95)

    plt.show()


def triamatrix(a, ax, rot=0, cmap=plt.cm.viridis, vmin=0, vmax=1, **kwargs):
    """
    From https://stackoverflow.com/questions/44291155/plotting-two-distance-matrices-together-on-same-plot
    """
    segs = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            segs.append(triatpos((j, i), rot=rot))
    col = collections.PolyCollection(segs, cmap=cmap, clim=(vmin, vmax), **kwargs)
    col.set_array(a.flatten())
    ax.add_collection(col)
    return col


def triatpos(pos=(0, 0), rot=0):
    r = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]]) * 0.5
    rm = [
        [np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))],
        [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))],
    ]
    r = np.dot(rm, r.T).T
    r[:, 0] += pos[0]
    r[:, 1] += pos[1]
    return r


if __name__ == "__main__":
    typer.run(main)
