import functools
from statistics import mean, stdev

import matplotlib.pyplot as plt
import inspect

import seaborn as sns
import streamlit as st
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from stqdm import stqdm
import tqdm

from src.feature_transforms import (
    ALL_FEATURE_TRANSFORMERS,
    SequentialFeatureTransformer,
)

tqdm.tqdm = functools.partial(stqdm)
from tqdm import tqdm

from src.utils.plots_and_metrics import confidence_interval
from src.utils.utils import (
    set_random_seed,
)
from src.utils.outlier_detectors import (
    detect_outliers,
)
from src.utils.data_fetchers import (
    get_test_features,
    get_features_data_loader,
)

from src.few_shot_methods import ALL_FEW_SHOT_CLASSIFIERS
from src.outlier_detection_methods import ALL_OUTLIER_DETECTORS

st.set_page_config(
    page_title="Outlier detection", layout="wide", initial_sidebar_state="expanded"
)

st.sidebar.header("Define tasks")
n_way = int(st.sidebar.number_input("n_way", min_value=1, value=5))
n_shot = int(st.sidebar.number_input("n_shot", min_value=1, value=5))
n_query = int(st.sidebar.number_input("n_query", min_value=1, value=10))
n_tasks = int(st.sidebar.number_input("n_tasks", min_value=1, value=500, step=100))
random_seed = 0
n_workers = 12
set_random_seed(random_seed)


st.sidebar.header("Choose features")

dataset = st.sidebar.selectbox(
    "Dataset",
    ["mini_imagenet", "cifar"],
)
backbone = st.sidebar.selectbox(
    "Backbone",
    ["resnet18", "resnet12"],
)
training_method = st.sidebar.selectbox(
    "Backbone",
    ["classic", "episodic"],
)
center_features = st.sidebar.checkbox(
    "Center features on train set average", value=True
)

features, train_features, average_train_features = get_test_features(
    backbone=backbone, dataset=dataset, training_method=training_method
)

data_loader = get_features_data_loader(
    features,
    average_train_features if center_features else None,
    n_way,
    n_shot,
    n_query,
    n_tasks,
    n_workers,
)


def select_class(class_list, name):
    class_dict = {x.__name__: x for x in class_list}
    class_str = st.selectbox(
        name,
        class_dict.keys(),
    )
    return class_dict[class_str]


def get_transformers(key=0):
    transformers_dict = {x.__name__: x for x in ALL_FEATURE_TRANSFORMERS}
    selected_transformers = st.multiselect(
        "Feature transformers", transformers_dict.keys(), key=key
    )
    instantiated_transformers = [
        transformers_dict[transformer](**get_args(transformers_dict[transformer]))
        for transformer in selected_transformers
    ]
    return SequentialFeatureTransformer(instantiated_transformers)


def get_args(class_, extra=None):
    signature = inspect.signature(class_.__init__)
    args = {}
    for parameter_name, parameter in signature.parameters.items():
        if parameter.annotation in [float]:
            args[parameter_name] = parameter.annotation(
                st.number_input(
                    parameter_name,
                    value=parameter.default,
                    format="%1.4f" if parameter.default < 0.1 else "%0.2f",
                )
            )
        if parameter.annotation in [int]:
            args[parameter_name] = parameter.annotation(
                st.number_input(parameter_name, value=parameter.default)
            )
        elif parameter.annotation in [str]:
            args[parameter_name] = parameter.annotation(
                st.text_input(parameter_name, value=parameter.default)
            )
        elif parameter.annotation in [bool]:
            args[parameter_name] = parameter.annotation(
                st.checkbox(parameter_name, value=parameter.default)
            )
        elif extra is not None:
            if parameter_name in extra:
                args[parameter_name] = None
    return args


def evaluate_classifier(classifier):
    accuracy_list = []
    query_top_1_score_list = []
    support_top_1_score_list = []
    for support_features, support_labels, query_features, query_labels, _ in tqdm(
        data_loader, desc="Evaluating the classifier..."
    ):
        support_predictions, query_predictions = classifier(
            support_features=support_features,
            query_features=query_features,
            support_labels=support_labels,
        )
        hard_query_predictions = query_predictions.argmax(-1)
        query_top_1_score_list += query_predictions.max(-1)[0].tolist()
        support_top_1_score_list += support_predictions.max(-1)[0].tolist()

        accuracy_list.append(
            (
                hard_query_predictions[: n_way * n_query].detach().data
                == query_labels[: n_way * n_query]
            )
            .sum()
            .item()
            / (n_way * n_query)
        )
    st.write(
        f"Average classifier accuracy: {(100 * mean(accuracy_list)):.2f}"
        f" +- {(100 * confidence_interval(stdev(accuracy_list), len(accuracy_list))):.2f} %"
    )
    st.write(
        f"Average Top-1 score for queries: {(100 * mean(query_top_1_score_list)):.2f}"
        f" +- {(100 * confidence_interval(stdev(query_top_1_score_list), len(query_top_1_score_list))):.2f} %"
    )

    st.write(
        f"Average Top-1 score for support examples: {(100 * mean(support_top_1_score_list)):.2f}"
        f" +- {(100 * confidence_interval(stdev(support_top_1_score_list), len(support_top_1_score_list))):.2f} %"
    )


def get_detector():
    st.header("Outlier detector")
    outlier_detector_class = select_class(ALL_OUTLIER_DETECTORS, "Outlier detector")
    detector_args = get_args(
        outlier_detector_class, extra=["few_shot_classifier", "base_features"]
    )
    detector_args["prepool_feature_transformer"] = get_transformers(key=0)
    detector_args["postpool_feature_transformer"] = get_transformers(key=1)
    if "few_shot_classifier" in detector_args.keys():
        st.header("Few-Shot Classifier")
        few_shot_classifier_class = select_class(
            ALL_FEW_SHOT_CLASSIFIERS, "Few-Shot Classifier"
        )

        classifier_args = get_args(few_shot_classifier_class)

        few_shot_classifier = few_shot_classifier_class(**classifier_args)
        evaluate_classifier(few_shot_classifier)
        detector_args["few_shot_classifier"] = few_shot_classifier

    if "base_features" in detector_args.keys():
        detector_args["base_features"] = train_features

    return outlier_detector_class(**detector_args)


outlier_detector = get_detector()

st.title("Results")

outliers_df = detect_outliers(outlier_detector, data_loader, n_way, n_query)

fp_rate, tp_rate, _ = roc_curve(outliers_df.outlier, -outliers_df.outlier_score)

cols = st.columns([1, 2, 2])

with cols[0]:
    roc_auc = auc(fp_rate, tp_rate)
    objective = st.slider("Objective", min_value=0.0, max_value=1.0, value=0.9)

    precisions, recalls, _ = precision_recall_curve(
        outliers_df.outlier, -outliers_df.outlier_score
    )
    precision_at_recall_objective = precisions[
        next(i for i, value in enumerate(recalls) if value < objective)
    ]
    recall_at_precision_objective = recalls[
        next(i for i, value in enumerate(precisions) if value > objective)
    ]
    st.write(f"ROC AUC: {100 * roc_auc:.2f}%")
    st.write(
        f"Precision for objective recall: {100 * precision_at_recall_objective:.2f}%"
    )
    st.write(
        f"Recall for objective precision: {100 * recall_at_precision_objective:.2f}%"
    )

with cols[1]:
    fig, ax = plt.subplots(1)
    ax.plot(fp_rate, tp_rate)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("ROC")
    st.pyplot(fig, clear_figure=True)

with cols[2]:
    fig, ax = plt.subplots(1)
    sns.histplot(ax=ax, data=outliers_df, x="outlier_score", hue="outlier")
    ax.set_title("Twin hist")
    st.pyplot(fig, clear_figure=True)
