import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import inspect
from src.models import BACKBONE_CONFIGS
from loguru import logger
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn
import torch.nn.functional as F
from easyfsl.utils import compute_prototypes
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch import Tensor
from mpl_toolkits.axes_grid1 import ImageGrid
from loguru import logger
import torch.nn as nn
import math

plt.style.use("ggplot")


class SSLMethod:
    def __init__(
        self,
        args,
        confidence_threshold,
        lambda_cc,
        lambda_em,
        lr,
        n_iter,
        depth_to_finetune,
        cons_loss,
        temperature,
        xi=10.0,
        eps=1.0,
        ip=1,
    ):
        self.device = "cuda"
        self.lambda_cc = lambda_cc
        self.lambda_em = lambda_em
        self.cache_size = 1
        self.temperature = temperature
        self.cons_loss = cons_loss
        self.depth_to_finetune = depth_to_finetune
        self.args = args
        self.lr = lr
        self.n_iter = n_iter

        # FixMatch / PseudoLabel
        self.confidence_threshold = confidence_threshold

        # VAT
        self.xi = xi
        self.eps = eps
        self.ip = ip

        image_size = BACKBONE_CONFIGS[args.backbone]["input_size"][-1]
        mean = BACKBONE_CONFIGS[args.backbone]["mean"]
        std = BACKBONE_CONFIGS[args.backbone]["std"]
        NORMALIZE = transforms.Normalize(mean, std)
        self.weak_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(int(image_size)),
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )

    def ssl_loss(
        self, interm_feat_wq, logits_s, support_labels, logits_cls_wq, logits_cls_sq
    ):
        """
        Enforces consistency regularization.
        """
        # assert logits_cls_wq.size() == logits_cls_sq.size()
        loss = nn.CrossEntropyLoss()

        Ls = loss(logits_s, support_labels)

        # ==== Find confident predictions and PL ====

        probs_weak_q = logits_cls_wq.softmax(-1)
        with torch.no_grad():
            mask = probs_weak_q.max(-1).values > self.confidence_threshold
            pseudo_labels = probs_weak_q.argmax(-1)

        # ==== Consistency regularization ====
        if self.cons_loss == "fixmatch":
            if mask.sum().item() > 0.0:
                Lu = loss(logits_cls_sq[mask], pseudo_labels[mask])
            else:
                Lu = torch.tensor([255]).to(self.device)
        elif self.cons_loss == "vat":
            with torch.no_grad():
                pred = logits_cls_wq.softmax(-1)

            # prepare random unit tensor
            d = torch.rand(interm_feat_wq.shape).sub(0.5).to(self.device)
            d = _l2_normalize(d)

            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = self.classification_head(
                    self.mid2end(interm_feat_wq + self.xi * d)
                )
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                adv_distance.backward()
                d = _l2_normalize(d.grad)  # Update direction

                self.optimizer.zero_grad()

            # calc LDS
            r_adv = d.detach() * self.eps
            pred_hat = self.classification_head(self.mid2end(interm_feat_wq + r_adv))
            logp_hat = F.log_softmax(pred_hat, dim=1)
            Lu = F.kl_div(logp_hat, pred, reduction="batchmean")

        # ==== Entropy Regularization (or pseudo label)
        # Lem = - (probs_weak_q * torch.log(probs_weak_q)).sum(-1).mean(0)
        if mask.sum().item() > 0.0:
            Lem = loss(logits_cls_wq[mask], pseudo_labels[mask])
        else:
            Lem = torch.tensor([255]).to(self.device)

        return Ls, Lu, Lem, Ls + self.lambda_cc * Lu + self.lambda_em * Lem, mask

    def classification_head(self, samples):
        return (
            self.temperature
            * F.normalize(samples, dim=1)
            @ F.normalize(self.prototypes, dim=1).T
        )

    def compute_auc(self, end_feat_wq, **kwargs):
        outlier_scores = self.get_outlier_scores(end_feat_wq, **kwargs).detach()
        fp_rate, tp_rate, thresholds = roc_curve(
            kwargs["outliers"].numpy(), outlier_scores.cpu().numpy()
        )
        return auc_fn(fp_rate, tp_rate)

    def start2mid(self, img_list, augmentation="weak"):
        aug_fn = eval(f"self.{augmentation}_transform")
        input_ = torch.stack([aug_fn(img).to(self.device) for img in img_list], 0)
        feats = self.frozen_part([input_])[-1]  # [N, c, h, w]
        return feats

    def start2end(self, img_list, augmentation="weak"):
        return self.mid2end(self.start2mid(img_list, augmentation))

    def mid2end(self, feats):
        end_feats = self.to_finetune([feats])[-1]
        pooled_feats = end_feats.mean((-2, -1))
        return pooled_feats

    def clear(self):
        delattr(self, "optimizer")
        delattr(self, "frozen_part")
        delattr(self, "to_finetune")

    def init_classifier(self, end_feat_ws, support_labels):
        self.prototypes = compute_prototypes(end_feat_ws, support_labels)
        self.prototypes.requires_grad_(True)

    def partition_model(self, feature_extractor):

        feature_extractor.eval()
        cutoff = len(feature_extractor.blocks) - self.depth_to_finetune
        if cutoff == 0:
            self.frozen_part = TrivialModule()
            self.to_finetune = nn.Sequential(*feature_extractor.blocks[:cutoff])
        elif cutoff == len(feature_extractor.blocks):
            self.frozen_part = nn.Sequential(*feature_extractor.blocks[:cutoff])
            self.to_finetune = TrivialModule()
        else:
            self.frozen_part = nn.Sequential(*feature_extractor.blocks[:cutoff])
            self.to_finetune = nn.Sequential(*feature_extractor.blocks[cutoff:])

        self.to_finetune.requires_grad_(True)
        self.frozen_part.requires_grad_(False)

    def __call__(self, support_images, support_labels, query_images, **kwargs):
        """
        query_images [Ns, d]
        """
        feature_extractor = kwargs["feature_extractor"]
        self.partition_model(feature_extractor)

        support_labels = support_labels.to(self.device)
        query_labels = kwargs["query_labels"].to(self.device)
        true_outliers = kwargs["outliers"].to(self.device)
        num_classes = support_labels.unique().size(0)

        with torch.no_grad():
            end_feat_ws = self.start2end(support_images, "weak")
            end_feat_wq = self.start2end(query_images, "weak")

        # logger.info(f"Caching {self.cache_size} augmented views")
        with torch.no_grad():
            pseudo_inliers = self.select_inliers(end_feat_wq=end_feat_wq, **kwargs)
            pseudo_inliers_indexes = torch.where(pseudo_inliers)[0]
            interm_feat_sq = []
            for i in range(self.cache_size):
                interm_feat_sq.append(
                    self.start2mid(
                        [query_images[i] for i in pseudo_inliers_indexes], "strong"
                    )
                )
            interm_feat_sq = torch.stack(interm_feat_sq)

            if not hasattr(
                self, "optimizer"
            ):  # Sometimes, methods would define it already
                self.init_classifier(end_feat_ws, support_labels)
                self.optimizer = torch.optim.SGD(
                    [self.prototypes] + list(self.to_finetune.parameters()), lr=self.lr
                )
        l_sup = []
        l_cons = []
        l_ems = []
        acc = []
        conf_acc = []
        accuracy_ood = []
        conf_samples_entropy = []

        # ===== Visualize episode ====
        if self.args.visu_episode:
            self.visualize_episode(
                support_images, query_images, support_labels, query_labels
            )

        interm_feat_ws = self.start2mid(support_images, "weak")
        interm_feat_wq = self.start2mid(query_images, "weak")

        # Get potential inliners and filter out

        for iter_ in range(self.n_iter):

            # Compute logits
            end_feat_ws = self.mid2end(interm_feat_ws)  # only part being optimized
            end_feat_wq = self.mid2end(interm_feat_wq)  # only part being optimized
            end_feat_sq = self.mid2end(interm_feat_sq[iter_ % self.cache_size])

            logits_cls_ws = self.classification_head(end_feat_ws)
            logits_cls_wq = self.classification_head(end_feat_wq)
            logits_cls_sq = self.classification_head(end_feat_sq)

            # Compute loss
            ls, lcons, lem, full_loss, conf_mask = self.ssl_loss(
                interm_feat_wq=interm_feat_wq[pseudo_inliers],
                logits_s=logits_cls_ws,
                support_labels=support_labels,
                logits_cls_wq=logits_cls_wq[pseudo_inliers],
                logits_cls_sq=logits_cls_sq,
            )
            detector_loss, *_ = self.update_detector(
                support_images=support_images,
                query_images=query_images,
                feature_extractor=feature_extractor,
                end_feat_ws=end_feat_ws,
                end_feat_wq=end_feat_wq,
                end_feat_sq=end_feat_sq,
                logits_cls_ws=logits_cls_ws,
                logits_cls_wq=logits_cls_wq,
                logits_cls_sq=logits_cls_sq,
                support_labels=support_labels,
            )
            if detector_loss is not None:
                full_loss += detector_loss

            loss = full_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Do some logging
            with torch.no_grad():
                inliers = ~kwargs["outliers"].bool().to(self.device)
                acc.append(
                    (logits_cls_wq.argmax(-1) == query_labels)[inliers]
                    .float()
                    .mean()
                    .item()
                )
                l_sup.append(ls.item())
                l_cons.append(lcons.item())
                l_ems.append(lem.item())
                accuracy_ood.append(
                    ((~pseudo_inliers).float() == true_outliers).float().mean().item()
                )

                if pseudo_inliers.sum() > 0 and conf_mask.sum() > 0:
                    pseudo_label_marg_dist = (
                        F.one_hot(
                            kwargs["query_labels"][pseudo_inliers][conf_mask],
                            num_classes,
                        )
                        .float()
                        .mean(0)
                    )
                    conf_samples_entropy.append(
                        -(
                            pseudo_label_marg_dist
                            * torch.log(pseudo_label_marg_dist + 1e-12)
                        )
                        .sum()
                        .item()
                        / math.log(num_classes)
                    )
                    conf_acc.append(
                        (logits_cls_wq.argmax(-1) == query_labels)[pseudo_inliers][
                            conf_mask
                        ]
                        .float()
                        .mean()
                        .item()
                    )
                else:
                    conf_samples_entropy.append(255)
                    conf_acc.append(255)

        soft_preds_s, soft_preds_q, outlier_scores = self.validate(
            support_images=support_images,
            query_images=query_images,
            **kwargs,
        )
        kwargs["intra_task_metrics"]["main_losses"]["l_sup"].append(l_sup)
        kwargs["intra_task_metrics"]["main_losses"]["l_cons"].append(l_cons)
        kwargs["intra_task_metrics"]["main_losses"]["l_em"].append(l_ems)
        kwargs["intra_task_metrics"]["main_metrics"]["acc"].append(acc)
        kwargs["intra_task_metrics"]["secondary_metrics"]["ood_accuracy"].append(
            accuracy_ood
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["conf_acc"].append(conf_acc)
        kwargs["intra_task_metrics"]["secondary_metrics"][
            "conf_samples_entropy"
        ].append(conf_samples_entropy)

        return soft_preds_s.cpu(), soft_preds_q.cpu(), outlier_scores.cpu()

    def update_detector(self, *args, **kwargs):
        raise NotImplementedError

    def select_inliers(self, *args, **kwargs):
        """
        Returns a mask, shape [Nq] that says whether samples are believed to be inliers or outliers.
        """
        raise NotImplementedError

    def validate(self, support_images, query_images, **kwargs):

        # switch to evaluate mode
        self.frozen_part.eval()
        self.to_finetune.eval()

        with torch.no_grad():

            # compute output
            val_feat_s = self.start2end(support_images, "val")
            logits_cls_vs = self.classification_head(val_feat_s)
            val_feat_q = self.start2end(query_images, "val")
            logits_cls_vq = self.classification_head(val_feat_q)
            outlier_scores = self.get_outlier_scores(
                end_feat_wq=val_feat_q, logits_cls_wq=logits_cls_vq, **kwargs
            )

        return logits_cls_vs.softmax(-1), logits_cls_vq.softmax(-1), outlier_scores

    def __str__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if "args" in arg_names:
            arg_names.remove("args")
        if "kwargs" in arg_names:
            arg_names.remove("kwargs")
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def __repr__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if "args" in arg_names:
            arg_names.remove("args")
        if "kwargs" in arg_names:
            arg_names.remove("kwargs")
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def visualize_episode(
        self, support_images, query_images, support_labels, query_labels
    ):
        num_classes = query_labels.unique().size(0)
        support = torch.stack(
            [self.weak_transform(img).to(self.device) for img in support_images], 0
        )
        query = torch.stack(
            [self.strong_transform(img).to(self.device) for img in query_images], 0
        )
        path = Path("results") / self.args.exp_name / "visu.png"
        make_episode_visualization(
            self.args,
            support.cpu().numpy(),
            query.cpu().numpy(),
            support_labels.cpu().numpy(),
            query_labels.cpu().numpy(),
            F.one_hot(query_labels, num_classes).cpu().numpy(),
            path,
        )


def make_episode_visualization(
    args,
    img_s: np.ndarray,
    img_q: np.ndarray,
    gt_s: np.ndarray,
    gt_q: np.ndarray,
    preds: np.ndarray,
    save_path: str,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    max_support = 1
    max_query = 5
    max_classes = 5

    # 0) Preliminary checks
    assert (
        len(img_s.shape) == 4
    ), f"Support shape expected : Ks x 3 x H x W or Ks x H x W x 3. Currently: {img_s.shape}"
    assert (
        len(img_q.shape) == 4
    ), f"Query shape expected : Kq x 3 x H x W or Kq x H x W x 3. Currently: {img_q.shape}"
    assert (
        len(preds.shape) == 2
    ), f"Predictions shape expected : Kq x num_classes. Currently: {preds.shape}"
    assert (
        len(gt_s.shape) == 1
    ), f"Support GT shape expected : Ks. Currently: {gt_s.shape}"
    assert (
        len(gt_q.shape) == 1
    ), f"Query GT shape expected : Kq. Currently: {gt_q.shape}"

    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[1] == 3:
        img_q = np.transpose(img_q, (0, 2, 3, 1))

    assert (
        img_s.shape[-3:-1] == img_q.shape[-3:-1]
    ), f"Support's resolution is {img_s.shape[-3:-1]} \
                                                      Query's resolution is {img_q.shape[-3:-1]}"

    if img_s.min() < 0:
        logger.info(
            f"Support images between {img_s.min()} and {img_s.max()} -> Renormalizing"
        )
        img_s *= std
        img_s += mean
        logger.info(f"Post normalization : {img_s.min()} and {img_s.max()}")

    if img_q.min() < 0:
        logger.info(
            f"Query images between {img_q.min()} and {img_q.max()} -> Renormalizing"
        )
        img_q *= std
        img_q += mean
        logger.info(f"Post normalization : {img_q.min()} and {img_q.max()}")

    Kq, num_classes = preds.shape

    # Group samples by class
    samples_s = {}
    samples_q = {}
    preds_q = {}

    for class_ in np.unique(gt_s):
        samples_s[class_] = img_s[gt_s == class_]
        samples_q[class_] = img_q[gt_q == class_]
        preds_q[class_] = preds[gt_q == class_]

    # Create Grid
    max_s = min(max_support, np.max([v.shape[0] for v in samples_s.values()]))
    max_q = min(max_query, np.max([v.shape[0] for v in samples_q.values()]))
    n_rows = max_s + max_q
    n_columns = min(num_classes, max_classes)
    assert n_columns > 0

    fig = plt.figure(figsize=(4 * n_columns, 4 * n_rows), dpi=100)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(n_rows, n_columns),
        axes_pad=(0.4, 0.4),
        direction="row",
    )

    # 1) visualize the support set
    handles = []
    labels = []
    for i in range(max_s):
        for j in range(n_columns):
            ax = grid[n_columns * i + j]
            if i < len(samples_s[j]):
                img = samples_s[j][i]
                # logger.info(img.min(), img.max())
                # assert img.min() >= 0. and img.max() <= 1.0, (img.min(), img.max())
                make_plot(ax, img)

            ax.axis("off")
            if i == 0:
                ax.set_title(f"Class {j+1}", size=20)

            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]

    # 1) visualize the query set
    for i in range(max_s, max_s + max_q):
        for j in range(n_columns):
            ax = grid[n_columns * i + j]
            if i - max_s < len(samples_q[j]):
                img = samples_q[j][i - max_s]
                # logger.info(img.min(), img.max())
                # assert img.min() >= 0. and img.max() <= 1.0, (img.min(), img.max())

                make_plot(ax, img, preds_q[j][i - max_s], j, n_columns)

            ax.axis("off")
            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]

    acc = (np.argmax(preds, axis=1) == gt_q).mean()
    fig.suptitle(
        f"Method={args.feature_detector}   /    Episode Accuracy={acc:.2f}",
        size=32,
        weight="bold",
        y=0.97,
    )
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 0.05),
        loc="center",
        ncol=3,
        prop={"size": 30},
    )

    fig.savefig(save_path)
    fig.clf()
    logger.info(f"Figure saved at {save_path}")


def frame_image(img: np.ndarray, color: list, frame_width: int = 3) -> np.ndarray:
    b = frame_width  # border size in pixel
    ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y

    framed_img = color * np.ones((b + ny + b, b + nx + b, img.shape[2]))
    framed_img[b:-b, b:-b] = img

    return framed_img


class TrivialModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def make_plot(
    ax, img: np.ndarray, preds: np.ndarray = None, label: int = None, n_columns: int = 0
) -> None:

    if preds is not None:
        assert label is not None
        assert n_columns > 0

        titles = ["{:.2f}".format(p) for p in preds]

        pred_class: int = int(np.argmax(preds))
        titles[pred_class] = r"$\mathbf{{{}}}$".format(titles[pred_class])
        titles = titles[:n_columns]

        title: str = "/".join(titles)
        # ax.set_title(title, size=12)

        well_classified: bool = int(np.argmax(preds)) == label
        color = [0, 0.8, 0] if well_classified else [0.9, 0, 0]
        img = frame_image(img, color)
        ax.plot(
            0,
            0,
            "-",
            c=color,
            label="{} Queries".format(
                "Well classified" if well_classified else "Misclassified"
            ),
            linewidth=4,
        )
    else:  # Support images
        color = [0.0, 0.0, 0.0]
        img = frame_image(img, color)
        ax.plot(0, 0, "-", c=color, label="Support", linewidth=4)

    ax.imshow(img)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d
