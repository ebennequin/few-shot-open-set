import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import inspect
from src.models import BACKBONE_CONFIGS
from loguru import logger

class SSLMethod:
    def __init__(self, args, confidence_threshold,
                 lambda_, lr, n_iter):

        self.lambda_ = lambda_
        self.lr = lr
        self.n_iter = n_iter
        self.confidence_threshold = confidence_threshold
        image_size = BACKBONE_CONFIGS[args.backbone]['input_size'][-1]
        mean = BACKBONE_CONFIGS[args.backbone]['mean']
        std = BACKBONE_CONFIGS[args.backbone]['std']
        self.layer = args.layers[0]
        NORMALIZE = transforms.Normalize(mean, std)
        self.weak_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                NORMALIZE,
            ])
        self.strong_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(int(image_size * 256 / 224)),
                transforms.ToTensor(),
                NORMALIZE,
            ])
        self.val_transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                NORMALIZE,
            ])

    def ssl_loss(self, logits_s, support_labels, logits_weak_q, logits_strong_q):
        """
        Enforces consistency regularization.
        """
        loss = nn.CrossEntropyLoss()

        Ls = loss(logits_s, support_labels)

        # ==== Find confident predictions and PL ====

        probs_weak_q = logits_weak_q.softmax(-1)
        mask = (probs_weak_q.max(-1).values > self.confidence_threshold)
        pseudo_labels = probs_weak_q.argmax(-1)

        # ==== FixMatch regularization ====

        Lu = loss(logits_strong_q[mask], pseudo_labels[mask])

        return Ls, Lu, Ls + self.lambda_ * Lu

    def prepare_loaders(self, support_images, support_labels, query_images):

        support_trainloader = data.DataLoader(data.TensorDataset(support_images, support_labels), batch_size=64, shuffle=True)
        query_trainloader = data.DataLoader(data.TensorDataset(query_images), batch_size=64, shuffle=True)

        support_valloader = data.DataLoader(data.TensorDataset(support_images, support_labels), batch_size=64, shuffle=False)
        query_valloader = data.DataLoader(data.TensorDataset(query_images, support_labels), batch_size=64, shuffle=False)

        return support_trainloader, support_valloader, query_trainloader, query_valloader

    def extract_weak(self, feature_extractor, img_list):
        # logger.warning((feature_extractor, img_list))
        return feature_extractor(torch.stack([self.weak_transform(img).cuda() for img in img_list], 0), [self.layer])[self.layer].squeeze()

    def extract_strong(self, feature_extractor, img_list):
        return feature_extractor(torch.stack([self.strong_transform(img).cuda() for img in img_list], 0), [self.layer])[self.layer].squeeze()

    def extract_val(self, feature_extractor, img_list):
        return feature_extractor(torch.stack([self.val_transform(img).cuda() for img in img_list], 0), [self.layer])[self.layer].squeeze()

    def __call__(self, support_images, support_labels, query_images, **kwargs):
        """
        query_images [Ns, d]
        """
        feature_extractor = kwargs['feature_extractor']
        support_labels = support_labels.cuda()
        if not hasattr(self, 'optimizer'):  # Sometimes, methods would define it already
            num_classes = support_labels.unique().size(0)
            self.classification_head = nn.Linear(feature_extractor.layer_dims[-1], num_classes)
            self.optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + \
                                              list(self.classification_head.parameters()),
                                              lr=self.lr)

        # ====== Prepare data =========

        # support_trainloader, support_valloader, query_trainloader, query_valloader = self.prepare_loaders(
        #     support_images, support_labels, query_images)

        # ===== Train classification part =====

        for iter_ in range(self.n_iter):

            # try:
            #     support_images, tgt = next(support_iterloader)
            # except:
            #     support_iterloader = iter(support_trainloader)
            #     support_images, tgt = next(support_iterloader)
            # try:
            #     query_images, tgt = next(query_iterloader)
            # except:
            #     query_iterloader = iter(query_trainloader)
            #     query_images = next(query_iterloader)

            # Get potential inliners
            weak_feat_q = self.extract_weak(feature_extractor, query_images)

            candidate_inliers = torch.where(self.select_inliers(features_q=weak_feat_q))[0]

            # Filter out OOD sampes

            strong_feat_q = self.extract_strong(feature_extractor, query_images)

            # Compute logits and loss
            logits_weak_s = self.classification_head(self.extract_weak(feature_extractor, support_images))
            logits_weak_q = self.classification_head(weak_feat_q)
            logits_strong_q = self.classification_head(strong_feat_q)

            # Update classification feature_extractor

            ls, lu, full_loss = self.ssl_loss(logits_weak_s, support_labels,
                                              logits_weak_q[candidate_inliers], logits_strong_q[candidate_inliers])
            detector_loss = self.update_detector(weak_feat_q)
            if detector_loss is not None:
                full_loss += detector_loss

            self.optimizer.zero_grad()
            full_loss.backward()
            self.optimizer.step()

        soft_preds_s, soft_preds_q, outlier_scores = self.validate(support_images=support_images,
                                                                   query_images=query_images,
                                                                   feature_extractor=feature_extractor,
                                                                   )

        return soft_preds_s.cpu(), soft_preds_q.cpu(), outlier_scores.cpu()

    def update_detector(self, *args, **kwargs):
        raise NotImplementedError

    def select_inliers(self, *args, **kwargs):
        """
        Returns a mask, shape [Nq] that says whether samples are believed to be inliers or outliers.
        """
        raise NotImplementedError

    def pre_train(self, *args, **kwargs):
        """
        In case the outlier detector requires pre-training
        """
        pass

    def validate(self, support_images, query_images, feature_extractor):

        # switch to evaluate mode
        feature_extractor.eval()

        with torch.no_grad():

            # compute output
            soft_preds_s = self.classification_head(self.extract_val(feature_extractor, support_images))
            val_feat_q = self.extract_val(feature_extractor, query_images)
            soft_preds_q = self.classification_head(val_feat_q)
            outlier_scores = self.get_outlier_scores()

        return soft_preds_s, soft_preds_q, outlier_scores

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