import torch
import torch.nn.functional as F
import torch.nn as nn
from .abstract import SSLMethod
from loguru import logger
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn


class T2T(SSLMethod):

    def __init__(self, args, pretrain_iter, **kwargs):
        print(kwargs)
        super().__init__(args, **kwargs)
        self.pretrain_iter = pretrain_iter

    def select_inliers(self, weak_feat_q, **kwargs):
        with torch.no_grad():
            logits_cls_wq = self.classification_head(weak_feat_q)
            outlier_scores = self.get_outlier_scores(weak_feat_q, logits_cls_wq)
            thresh = threshold_otsu(outlier_scores.cpu().numpy())
            mask = outlier_scores < thresh
        return mask

    def compute_auc(self, weak_feat_q, logits_cls_wq, **kwargs):
        outlier_scores = self.get_outlier_scores(weak_feat_q, logits_cls_wq).detach()
        fp_rate, tp_rate, thresholds = roc_curve(kwargs['outliers'].numpy(), outlier_scores.cpu().numpy())
        return auc_fn(fp_rate, tp_rate)

    def get_outlier_scores(self, weak_feat_q, logits_cls_wq, **kwargs):

        y_onehot = torch.zeros_like(logits_cls_wq)
        y_pred = torch.argmax(logits_cls_wq, dim=-1, keepdim=True)
        y_onehot.scatter_(1, y_pred, 1)

        matching_score = torch.sigmoid(self.cmm_head(weak_feat_q, y_onehot))
        return 1 - matching_score

    def update_detector(self, weak_feat_s, weak_feat_q, logits_cls_ws, logits_cls_wq,
                        support_labels, feature_extractor, query_images,
                        include_unballed_xm=True, **kwargs):

        num_classes = support_labels.unique().size(0)

        # Cross Modal Matching Training: 1 positve pair + 2 negative pair for each labeled data
        # [--pos--, --hard_neg--, --easy_neg--]
        batch_size = weak_feat_s.size(0)
        matching_gt = torch.zeros(3 * batch_size).to(self.device)
        matching_gt[:batch_size] = 1
        y_onehot = torch.zeros((3 * batch_size, num_classes)).float().to(self.device)
        y = torch.zeros(3 * batch_size).long().to(self.device)
        y[:batch_size] = support_labels
        with torch.no_grad():
            prob_sorted_index = torch.argsort(logits_cls_ws, descending=True)
            for i in range(batch_size):
                if prob_sorted_index[i, 0] == support_labels[i]:
                    y[1 * batch_size + i] = prob_sorted_index[i, 1]
                    y[2 * batch_size + i] = int(np.random.choice(prob_sorted_index[i, 2:].cpu(), 1))
                else:
                    y[1 * batch_size + i] = prob_sorted_index[i, 0]
                    choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                    while choice == support_labels[i]:
                        choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                    y[2 * batch_size + i] = choice
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        matching_score_x = self.cmm_head(weak_feat_s.repeat(3, 1), y_onehot)
        Lmx = F.binary_cross_entropy_with_logits(matching_score_x.view(-1), matching_gt)

        # Cross Entropy Loss for Rotation Recognition
        feats_r = self.extract_rot(feature_extractor, query_images)
        targets_r = torch.cat(
            [torch.empty(weak_feat_q.size(0)).fill_(i).long() for i in range(4)], dim=0).to(self.device)
        Lr = F.cross_entropy(self.rotnet_head(feats_r), targets_r, reduction='mean')

        loss = Lmx + Lr

        if include_unballed_xm:
            batch_size = weak_feat_q.size(0)
            y_onehot = torch.zeros((2 * batch_size, num_classes)).float().to(self.device)
            y = torch.zeros(2 * batch_size).long().to(self.device)
            # select the most confident class and randomly choose one from rest classes
            with torch.no_grad():
                prob_sorted_index = torch.argsort(logits_cls_wq, descending=True, dim=-1)
                y[:batch_size] = prob_sorted_index[:, 0]
                for i in range(batch_size):
                    y[batch_size + i] = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            matching_score_u = self.cmm_head(weak_feat_q.repeat(2, 1), y_onehot)
            Lmu = F.binary_cross_entropy_with_logits(matching_score_u, torch.sigmoid(matching_score_u))

            loss += Lmu
        else:
            Lmu = torch.Tensor([0.])

        return loss, Lmx, Lr, Lmu

    def extract_rot(self, feature_extractor, img_list):
        """
        img_list [N] images

        returns:
            [4N] extracted features
        """
        tensor_images = torch.stack([self.val_transform(img).to(self.device) for img in img_list], 0)
        rotated_images = torch.cat(
                [torch.rot90(tensor_images, i, [2, 3]) for i in range(4)], dim=0)
        return feature_extractor(rotated_images, [self.layer])[self.layer].squeeze()

    def __call__(self, support_images, support_labels, query_images, **kwargs):
        """
        query_images [Ns, d]
        """
        intra_task_metrics = kwargs['intra_task_metrics']

        # ====== Initialize modules =====
        support_labels = support_labels.to(self.device)

        feature_extractor = kwargs['feature_extractor']
        feature_extractor.eval()
        feature_extractor.requires_grad_(False)
        num_classes = support_labels.unique().size(0)

        self.classification_head = nn.Linear(feature_extractor.layer_dims[-1], num_classes).to(self.device)
        self.rotnet_head = torch.nn.Linear(feature_extractor.layer_dims[-1], 4).to(self.device)
        self.cmm_head = CrossModalMatchingHead(num_classes, feature_extractor.layer_dims[-1]).to(self.device)
        grouped_parameters = list(self.classification_head.parameters()) + \
            list(self.cmm_head.parameters()) + \
            list(self.rotnet_head.parameters())
        self.optimizer = torch.optim.SGD(grouped_parameters, lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # ===== Pre-train =====
        l_sups = []
        l_rots = []
        l_mxs = []
        detector_losses = []
        accs = []
        aucs = []

        for i in range(self.pretrain_iter):
            weak_feat_s = self.extract_weak(feature_extractor, support_images)
            weak_feat_q = self.extract_weak(feature_extractor, query_images)

            logits_cls_ws = self.classification_head(weak_feat_s).squeeze()
            logits_cls_wq = self.classification_head(weak_feat_q).squeeze()

            # Loss
            l_sup = F.cross_entropy(logits_cls_ws, support_labels, reduction='mean')
            detector_loss, Lmx, Lr, _ = self.update_detector(support_images=support_images, query_images=query_images,
                                                             feature_extractor=feature_extractor, weak_feat_s=weak_feat_s,
                                                             weak_feat_q=weak_feat_q, logits_cls_ws=logits_cls_ws,
                                                             logits_cls_wq=logits_cls_wq, support_labels=support_labels,
                                                             include_unballed_xm=False)

            loss = l_sup + detector_loss

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                l_sups.append(l_sup.item())
                accs.append((logits_cls_wq.argmax(-1) == kwargs['query_labels'].to(self.device)).float().mean().item())
                l_mxs.append(Lmx.item())
                l_rots.append(Lr.item())
                aucs.append(self.compute_auc(weak_feat_q, logits_cls_wq, **kwargs))

        # ===== Perform standard SSL inference =====
        kwargs['intra_task_metrics']['main_losses']['sup'].append(l_sups)
        kwargs['intra_task_metrics']['main_losses']['cross_matching'].append(l_mxs)
        kwargs['intra_task_metrics']['main_losses']['rotation'].append(l_rots)
        kwargs['intra_task_metrics']['metrics']['acc'].append(accs)
        kwargs['intra_task_metrics']['metrics']['auc'].append(aucs)

        return super().__call__(support_images, support_labels, query_images, **kwargs)


class CrossModalMatchingHead(nn.Module):
    def __init__(self, num_classes, feats_dim):
        super(CrossModalMatchingHead, self).__init__()
        self.label_embedding = nn.Linear(num_classes, 128)
        self.mlp = nn.Sequential(
            nn.Linear(feats_dim + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        # y is onehot vectors
        y_embbeding = self.label_embedding(y)
        return self.mlp(torch.cat([x, y_embbeding], dim=1))