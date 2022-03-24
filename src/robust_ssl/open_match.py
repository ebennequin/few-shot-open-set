import torch
import torch.nn.functional as F
import torch.nn as nn
from .abstract import SSLMethod
from loguru import logger


class OpenMatch(SSLMethod):

    def __init__(self, args, pretrain_iter, lambda_oem, lambda_socr, **kwargs):
        print(kwargs)
        super().__init__(args, **kwargs)
        self.pretrain_iter = pretrain_iter
        self.lambda_socr = lambda_socr
        self.lambda_oem = lambda_oem

    def select_inliers(self, weak_feat_q, **kwargs):

        outlier_scores = self.get_outlier_scores(weak_feat_q)
        potential_inliers = outlier_scores < 0.5
        return potential_inliers

    def get_outlier_scores(self, weak_feat_q, **kwargs):

        logits_cls = self.classification_head(weak_feat_q)
        cls_preds = logits_cls.argmax(-1)

        logits_open = self.outlier_head(weak_feat_q)
        out_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
        tmp_range = torch.arange(out_open.size(0)).long().cuda()
        outlier_scores = out_open[tmp_range, 0, cls_preds]
        return outlier_scores

    def update_detector(self, weak_feat_s, weak_feat_q, strong_feat_q, support_labels, **kwargs):

        logits_ood_ws = self.outlier_head(weak_feat_s)
        logits_ood_wq = self.outlier_head(weak_feat_q)
        logits_ood_sq = self.outlier_head(strong_feat_q)

        l_ova = ova_loss(logits_ood_ws, support_labels)

        # Open-set entropy minimization

        l_oem = ova_ent(logits_ood_wq) / 2.
        l_oem += ova_ent(logits_ood_sq) / 2.

        # Soft consistenty regularization

        logits_open_u1 = logits_ood_wq.view(logits_ood_wq.size(0), 2, -1)
        logits_open_u2 = logits_ood_sq.view(logits_ood_sq.size(0), 2, -1)
        logits_open_u1 = F.softmax(logits_open_u1, 1)
        logits_open_u2 = F.softmax(logits_open_u2, 1)
        # logger.warning((logits_open_u1.size(), logits_open_u2.size()))
        l_socr = torch.mean(torch.sum(torch.sum(torch.abs(
            logits_open_u1 - logits_open_u2)**2, 1), 1))

        detector_loss = l_ova + self.lambda_oem * l_oem + self.lambda_socr * l_socr

        return detector_loss

    def __call__(self, support_images, support_labels, query_images, **kwargs):
        """
        query_images [Ns, d]
        """
        # ====== Initialize modules =====
        support_labels = support_labels.cuda()

        feature_extractor = kwargs['feature_extractor']
        feature_extractor.eval()
        feature_extractor.requires_grad_(False)
        num_classes = support_labels.unique().size(0)

        self.classification_head = nn.Linear(feature_extractor.layer_dims[-1], num_classes).cuda()
        self.outlier_head = nn.Linear(feature_extractor.layer_dims[-1], 2 * num_classes, bias=False).cuda()
        self.optimizer = torch.optim.Adam(list(self.classification_head.parameters()) + \
                                          list(self.outlier_head.parameters()),
                                          lr=self.lr)

        # ===== Pre-train =====

        for i in range(self.pretrain_iter):
            weak_feat_s = self.extract_weak(feature_extractor, support_images)
            weak_feat_q = self.extract_weak(feature_extractor, query_images)
            strong_feat_q = self.extract_strong(feature_extractor, query_images)

            logits_cls_ws = self.classification_head(weak_feat_s).squeeze()

            # Loss for classifier

            l_sup = F.cross_entropy(logits_cls_ws, support_labels)

            # Open-set entropy minimization

            detector_loss = self.update_detector(weak_feat_s, weak_feat_q, strong_feat_q, support_labels)
            loss = l_sup + detector_loss

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ===== Perform standard SSL inference =====

        return super().__call__(support_images, support_labels, query_images, **kwargs)


def ova_loss(logits_open, label):
    """
    Using labelled data to increase margin on OVA.

    logits_open : shape [N, 2 * K]
    label : shape [N,]
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)  # [N, 2, K]
    logits_open = F.softmax(logits_open, 1)  # [N, 2, k]
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)  # [N, K]
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                               torch.log(logits_open + 1e-8), 1), 1))
    return Le
