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
plt.style.use('ggplot')


class SSLMethod:
    def __init__(self, args, confidence_threshold,
                 lambda_cc, lambda_em, lr, n_iter, depth_to_finetune,
                 temperature):
        self.device = 'cuda'
        self.lambda_cc = lambda_cc
        self.lambda_em = lambda_em
        self.temperature = temperature
        self.depth_to_finetune = depth_to_finetune
        self.args = args
        self.lr = lr
        self.n_iter = n_iter
        self.confidence_threshold = confidence_threshold
        image_size = BACKBONE_CONFIGS[args.backbone]['input_size'][-1]
        mean = BACKBONE_CONFIGS[args.backbone]['mean']
        std = BACKBONE_CONFIGS[args.backbone]['std']
        NORMALIZE = transforms.Normalize(mean, std)
        self.weak_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                NORMALIZE,
            ])
        self.strong_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(int(image_size)),
                transforms.ToTensor(),
                NORMALIZE,
            ])
        self.val_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                NORMALIZE,
            ])

    def ssl_loss(self, logits_s, support_labels, logits_cls_wq, logits_cls_sq):
        """
        Enforces consistency regularization.
        """
        assert logits_cls_wq.size() == logits_cls_sq.size()
        loss = nn.CrossEntropyLoss()

        Ls = loss(logits_s, support_labels)

        # ==== Find confident predictions and PL ====

        probs_weak_q = logits_cls_wq.softmax(-1)
        mask = (probs_weak_q.max(-1).values > self.confidence_threshold)
        pseudo_labels = probs_weak_q.argmax(-1)

        # ==== FixMatch regularization ====
        Lu = loss(logits_cls_sq[mask], pseudo_labels[mask])

        # ==== Standard entropy_regularization
        # Lem = - (probs_weak_q * torch.log(probs_weak_q)).sum(-1).mean(0)
        # marginal = probs_weak_q.mean(0)
        # Lem += (marginal * torch.log(marginal)).sum(-1)
        Lem = loss(logits_cls_wq[mask], pseudo_labels[mask])

        return Ls, Lu, Lem, Ls + self.lambda_cc * Lu + self.lambda_em * Lem, mask

    def classification_head(self, samples):
        return (
            self.temperature
            * F.normalize(samples, dim=1)
            @ F.normalize(self.prototypes, dim=1).T
        )

    def compute_auc(self, end_feat_wq, **kwargs):
        outlier_scores = self.get_outlier_scores(end_feat_wq, **kwargs).detach()
        fp_rate, tp_rate, thresholds = roc_curve(kwargs['outliers'].numpy(), outlier_scores.cpu().numpy())
        return auc_fn(fp_rate, tp_rate)

    def extract_weak(self, feature_extractor, img_list):
        x = torch.stack([self.weak_transform(img).to(self.device) for img in img_list], 0)
        return feature_extractor([x])[-1].squeeze()

    def extract_strong(self, feature_extractor, img_list):
        x = torch.stack([self.strong_transform(img).to(self.device) for img in img_list], 0)
        return feature_extractor([x])[-1].squeeze()

    def extract_val(self, feature_extractor, img_list):
        x = torch.stack([self.val_transform(img).to(self.device) for img in img_list], 0)
        return feature_extractor([x])[-1].squeeze()

    def clear(self):
        delattr(self, 'optimizer')

    def init_classifier(self, end_feat_ws, support_labels):
        self.prototypes = compute_prototypes(end_feat_ws, support_labels)
        self.prototypes.requires_grad_(True)
        # classifier = nn.Linear(end_feat_ws.size(-1), num_classes, bias=False)
        # with torch.no_grad():
        #     classifier.weight = nn.Parameter(weights)

    def prepare_model(self, feature_extractor):

        feature_extractor.eval()
        cutoff = len(feature_extractor.blocks) - self.depth_to_finetune
        if cutoff == 0:
            frozen_part = TrivialModule()
            to_finetune = nn.Sequential(*feature_extractor.blocks[:cutoff])
        elif cutoff == len(feature_extractor.blocks):
            frozen_part = nn.Sequential(*feature_extractor.blocks[:cutoff])
            to_finetune = TrivialModule()
        else:
            frozen_part = nn.Sequential(*feature_extractor.blocks[:cutoff])
            to_finetune = nn.Sequential(*feature_extractor.blocks[cutoff:])

        to_finetune.requires_grad_(True)
        frozen_part.requires_grad_(False)

        return frozen_part, to_finetune

    def process_end(self, feats):
        pooled_feats = feats.mean((-2, -1))
        return pooled_feats

    def __call__(self, support_images, support_labels, query_images, **kwargs):
        """
        query_images [Ns, d]
        """
        feature_extractor = kwargs['feature_extractor']
        frozen_part, to_finetune = self.prepare_model(feature_extractor)

        support_labels = support_labels.to(self.device)
        query_labels = kwargs['query_labels'].to(self.device)
        true_outliers = kwargs['outliers'].to(self.device)

        start2end = lambda x, y: self.process_end(to_finetune([x(frozen_part, y)])[-1])
        mid2end = lambda x: self.process_end(to_finetune([x])[-1])

        with torch.no_grad():
            end_feat_ws = start2end(self.extract_weak, support_images)
            end_feat_wq = start2end(self.extract_weak, query_images)

            if not hasattr(self, 'optimizer'):  # Sometimes, methods would define it already
                self.init_classifier(end_feat_ws, support_labels)
                self.optimizer = torch.optim.Adam([self.prototypes] + \
                                                  list(to_finetune.parameters()) ,
                                                  lr=self.lr)
        l_sup = []
        l_cons = []
        l_ems = []
        acc = []
        conf_acc = []
        accuracy_ood = []

        # ===== Visualize episode ====
        if self.args.visu_episode:
            self.visualize_episode(support_images, query_images, support_labels, query_labels)

        interm_feat_ws = self.extract_weak(frozen_part, support_images)
        interm_feat_wq = self.extract_weak(frozen_part, query_images)

        for iter_ in range(self.n_iter):

            end_feat_ws = mid2end(interm_feat_ws)
            end_feat_wq = mid2end(interm_feat_wq)

            # Get potential inliners and filter out
            pseudo_inliers = self.select_inliers(end_feat_wq=end_feat_wq, **kwargs)
            pseudo_inliers_indexes = torch.where(pseudo_inliers)[0]
            end_feat_sq = start2end(self.extract_strong, [query_images[i] for i in pseudo_inliers_indexes])

            # Compute logits and loss
            all_logits = self.classification_head(torch.cat([end_feat_ws, end_feat_wq, end_feat_sq]))
            logits_cls_ws = all_logits[:end_feat_ws.size(0)]
            logits_cls_wq = all_logits[end_feat_ws.size(0):end_feat_ws.size(0)+end_feat_wq.size(0)]
            logits_cls_sq = all_logits[-end_feat_sq.size(0):]

            # Update classification feature_extractor

            ls, lu, lem, full_loss, conf_mask = self.ssl_loss(logits_cls_ws, support_labels,
                                                              logits_cls_wq[pseudo_inliers], logits_cls_sq)
            detector_loss, *_ = self.update_detector(support_images=support_images, query_images=query_images,
                                                     feature_extractor=feature_extractor, end_feat_ws=end_feat_ws,
                                                     end_feat_wq=end_feat_wq, end_feat_sq=end_feat_sq,
                                                     logits_cls_ws=logits_cls_ws, logits_cls_wq=logits_cls_wq,
                                                     logits_cls_sq=logits_cls_sq, support_labels=support_labels)
            if detector_loss is not None:
                full_loss += detector_loss

            loss = full_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Do some logging
            with torch.no_grad():
                inliers = ~ kwargs['outliers'].bool().to(self.device)
                acc.append((logits_cls_wq.argmax(-1) == query_labels)[inliers].float().mean().item())
                l_sup.append(ls.item())
                l_cons.append(lu.item())
                l_ems.append(lem.item())
                accuracy_ood.append(((~pseudo_inliers).float() == true_outliers).float().mean().item())
                conf_acc.append((logits_cls_wq.argmax(-1) == query_labels)[pseudo_inliers][conf_mask].float().mean().item())

        soft_preds_s, soft_preds_q, outlier_scores = self.validate(support_images=support_images,
                                                                   query_images=query_images,
                                                                   frozen_part=frozen_part,
                                                                   to_finetune=to_finetune,
                                                                   **kwargs,
                                                                   )
        kwargs['intra_task_metrics']['secondary_metrics']['ood_accuracy'].append(accuracy_ood)
        kwargs['intra_task_metrics']['main_losses']['l_sup'].append(l_sup)
        kwargs['intra_task_metrics']['main_losses']['l_cons'].append(l_cons)
        kwargs['intra_task_metrics']['main_losses']['l_em'].append(l_ems)
        kwargs['intra_task_metrics']['main_metrics']['acc'].append(acc)
        kwargs['intra_task_metrics']['main_metrics']['conf_acc'].append(conf_acc)

        return soft_preds_s.cpu(), soft_preds_q.cpu(), outlier_scores.cpu()

    def update_detector(self, *args, **kwargs):
        raise NotImplementedError

    def select_inliers(self, *args, **kwargs):
        """
        Returns a mask, shape [Nq] that says whether samples are believed to be inliers or outliers.
        """
        raise NotImplementedError

    def validate(self, support_images, query_images, frozen_part, to_finetune, **kwargs):

        # switch to evaluate mode
        frozen_part.eval()
        to_finetune.eval()

        start2end = lambda x, y: self.process_end(to_finetune([x(frozen_part, y)])[-1])

        with torch.no_grad():

            # compute output
            val_feat_s = start2end(self.extract_val, support_images)
            logits_cls_vs = self.classification_head(val_feat_s)
            val_feat_q = start2end(self.extract_val, query_images)
            logits_cls_vq = self.classification_head(val_feat_q)
            outlier_scores = self.get_outlier_scores(end_feat_wq=val_feat_q, logits_cls_wq=logits_cls_vq, **kwargs)

        return logits_cls_vs.softmax(-1), logits_cls_vq.softmax(-1), outlier_scores

    def __str__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if 'args' in arg_names: arg_names.remove('args')
        if 'kwargs' in arg_names: arg_names.remove('kwargs')
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def __repr__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if 'args' in arg_names: arg_names.remove('args')
        if 'kwargs' in arg_names: arg_names.remove('kwargs')
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def visualize_episode(self, support_images, query_images, support_labels, query_labels):
        num_classes = query_labels.unique().size(0)
        support = torch.stack([self.weak_transform(img).to(self.device) for img in support_images], 0)
        query = torch.stack([self.strong_transform(img).to(self.device) for img in query_images], 0)
        path = Path('results') / self.args.exp_name / 'visu.png'
        make_episode_visualization(self.args,
                                   support.cpu().numpy(),
                                   query.cpu().numpy(),
                                   support_labels.cpu().numpy(),
                                   query_labels.cpu().numpy(),
                                   F.one_hot(query_labels, num_classes).cpu().numpy(),
                                   path)


def make_episode_visualization(args,
                               img_s: np.ndarray,
                               img_q: np.ndarray,
                               gt_s: np.ndarray,
                               gt_q: np.ndarray,
                               preds: np.ndarray,
                               save_path: str,
                               mean = [0.485, 0.456, 0.406],
                               std = [0.229, 0.224, 0.225]):
    max_support = 1
    max_query = 5
    max_classes = 5

    # 0) Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : Ks x 3 x H x W or Ks x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 4, f"Query shape expected : Kq x 3 x H x W or Kq x H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 2, f"Predictions shape expected : Kq x num_classes. Currently: {preds.shape}"
    assert len(gt_s.shape) == 1, f"Support GT shape expected : Ks. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 1, f"Query GT shape expected : Kq. Currently: {gt_q.shape}"

    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[1] == 3:
        img_q = np.transpose(img_q, (0, 2, 3, 1))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1], f"Support's resolution is {img_s.shape[-3:-1]} \
                                                      Query's resolution is {img_q.shape[-3:-1]}"

    if img_s.min() < 0:
        logger.info(f"Support images between {img_s.min()} and {img_s.max()} -> Renormalizing")
        img_s *= std
        img_s += mean
        logger.info(f"Post normalization : {img_s.min()} and {img_s.max()}")

    if img_q.min() < 0:
        logger.info(f"Query images between {img_q.min()} and {img_q.max()} -> Renormalizing")
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
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_rows, n_columns),
                     axes_pad=(0.4, 0.4),
                     direction='row',
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

            ax.axis('off')
            if i == 0:
                ax.set_title(f'Class {j+1}', size=20)

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

            ax.axis('off')
            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]

    acc = (np.argmax(preds, axis=1) == gt_q).mean()
    fig.suptitle(f'Method={args.feature_detector}   /    Episode Accuracy={acc:.2f}',
                 size=32,
                 weight='bold',
                 y=0.97)
    by_label = dict(zip(labels, handles))

    fig.legend(by_label.values(),
               by_label.keys(),
               bbox_to_anchor=(0.5, 0.05),
               loc='center',
               ncol=3,
               prop={'size': 30})

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


def make_plot(ax,
              img: np.ndarray,
              preds: np.ndarray = None,
              label: int = None,
              n_columns: int = 0) -> None:

    if preds is not None:
        assert label is not None
        assert n_columns > 0

        titles = ['{:.2f}'.format(p) for p in preds]

        pred_class: int = int(np.argmax(preds))
        titles[pred_class] = r'$\mathbf{{{}}}$'.format(titles[pred_class])
        titles = titles[:n_columns]

        title: str = '/'.join(titles)
        # ax.set_title(title, size=12)

        well_classified: bool = int(np.argmax(preds)) == label
        color = [0, 0.8, 0] if well_classified else [0.9, 0, 0]
        img = frame_image(img, color)
        ax.plot(0,
                0,
                "-",
                c=color,
                label='{} Queries'.format('Well classified' if well_classified
                                          else 'Misclassified'),
                linewidth=4)
    else:  # Support images
        color = [0., 0., 0.]
        img = frame_image(img, color)
        ax.plot(0, 0, "-", c=color, label='Support', linewidth=4)

    ax.imshow(img)
