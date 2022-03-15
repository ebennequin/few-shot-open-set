from typing import Tuple, List
import torch
import torch.nn.functional as F
from loguru import logger
from .abstract import FewShotMethod
import math

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize


class ICI(FewShotMethod):

    def __init__(self, classifier: str, step: int, max_iter: str, reduce: str, d: int, norm_name: str):
        super().__init__()
        self.step = step
        self.max_iter = max_iter
        self.reduce = reduce
        self.d = d
        self.norm_name = norm_name
        self.initial_embed(reduce, d)
        self.initial_norm(norm_name)
        self.initial_classifier(classifier)
        self.elasticnet = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                                     normalize=True, warm_start=True, selection='cyclic')

    def forward(self, support_features, query_features, support_labels, **kwargs):
        support_X, support_y = self.norm(support_features.numpy()), support_labels.numpy()
        way, num_support = support_labels.unique().size(0), len(support_X)
        query_X = self.norm(query_features.numpy())
        unlabel_X = query_X
        num_unlabel = unlabel_X.shape[0]

        embeddings = np.concatenate([support_X, unlabel_X])
        X = self.embed(embeddings)
        H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
        X_hat = np.eye(H.shape[0]) - H
        if self.max_iter == 'auto':
            # set a big number
            self.max_iter = num_support + num_unlabel
        elif self.max_iter == 'fix':
            self.max_iter = math.ceil(num_unlabel / self.step)
        else:
            assert float(self.max_iter).is_integer()

        support_set = np.arange(num_support).tolist()

        # Train classifier
        self.classifier.fit(support_X, support_y)
        
        for _ in range(self.max_iter):

            # Get pseudo labels
            pseudo_y = self.classifier.predict(unlabel_X)
            y = np.concatenate([support_y, pseudo_y])
            Y = self.label2onehot(y, way)
            y_hat = np.dot(X_hat, Y)

            # Expand based on credibility of pseudo labels
            support_set = self.expand(support_set, X_hat, y_hat, way, num_support, pseudo_y,
                                      embeddings, y)
            y = np.argmax(Y, axis=1)

            # Re-train classifier
            self.classifier.fit(embeddings[support_set], y[support_set])
            if len(support_set) == len(embeddings):
                break
        preds_q = torch.from_numpy(self.classifier.predict(query_X))
        preds_s = torch.from_numpy(self.classifier.predict(support_X))

        return F.one_hot(preds_s, way).float(), F.one_hot(preds_q, way).float()

    def expand(self, support_set, X_hat, y_hat, way, num_support, pseudo_y, embeddings, targets):

        # Get the path (i.e the evolution of |gamma_i| as a function of lambda increasing)
        _, coefs, _ = self.elasticnet.path(X_hat, y_hat, l1_ratio=1.0)
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, num_support:, :]), axis=2)
        selected = np.zeros(way)
        for gamma in coefs:
            for i, g in enumerate(gamma):
                if g == 0.0 and \
                    (i+num_support not in support_set) and \
                        (selected[pseudo_y[i]] < self.step):
                    support_set.append(i+num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= self.step) == way:
                break
        return support_set

    def initial_embed(self, reduce, d):
        reduce = reduce.lower()
        assert reduce in ['isomap', 'ltsa', 'mds', 'lle', 'se', 'pca', 'none']
        if reduce == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=d)
        elif reduce == 'ltsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d,
                                           n_neighbors=5, method='ltsa')
        elif reduce == 'mds':
            from sklearn.manifold import MDS
            embed = MDS(n_components=d, metric=False)
        elif reduce == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense')
        elif reduce == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=d)
        elif reduce == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x

    def initial_classifier(self, classifier):
        assert classifier in ['lr', 'svm']
        if classifier == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(C=10, gamma='auto', kernel='linear',probability=True)
        elif classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result 