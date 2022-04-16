import torch.nn.functional as F
from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
import math
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from .abstract import FewShotMethod
from .bd_cspn import BDCSPN
from torch import Tensor
from easyfsl.utils import compute_prototypes


class LaplacianShot(FewShotMethod):
    def __init__(self, inference_steps, knn, lambda_, softmax_temperature, proto_rect):
        super().__init__()
        self.proto_rect = proto_rect
        self.knn = knn
        self.inference_steps = inference_steps
        self.lambda_ = lambda_
        self.softmax_temperature = softmax_temperature

    def create_affinity(self, X):
        N, D = X.shape

        nbrs = NearestNeighbors(n_neighbors=self.knn).fit(X)
        dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), self.knn - 1)
        col = knnind[:, 1:].flatten()
        data = np.ones(X.shape[0] * (self.knn - 1))
        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
        return W

    def normalize(self, Y_in):
        maxcol = np.max(Y_in, axis=1)
        Y_in = Y_in - maxcol[:, np.newaxis]
        N = Y_in.shape[0]
        size_limit = 150000
        if N > size_limit:
            batch_size = 1280
            Y_out = []
            num_batch = int(math.ceil(1.0 * N / batch_size))
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, N)
                tmp = np.exp(Y_in[start:end, :])
                tmp = tmp / (np.sum(tmp, axis=1)[:, None])
                Y_out.append(tmp)
            del Y_in
            Y_out = np.vstack(Y_out)
        else:
            Y_out = np.exp(Y_in)
            Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

        return Y_out

    def entropy_energy(self, Y, unary, kernel, bound_lambda, batch=False):
        tot_size = Y.shape[0]
        pairwise = kernel.dot(Y)
        if batch == False:
            temp = (unary * Y) + (-bound_lambda * pairwise * Y)
            E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
        else:
            batch_size = 1024
            num_batch = int(math.ceil(1.0 * tot_size / batch_size))
            E = 0
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, tot_size)
                temp = (unary[start:end] * Y[start:end]) + (-bound_lambda * pairwise[start:end] * Y[start:end])
                E = E + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

        return E

    def bound_update(self, unary, kernel, batch=False):
        oldE = float('inf')
        Y = self.normalize(-unary)
        for i in range(self.inference_steps):
            additive = -unary
            mul_kernel = kernel.dot(Y)
            Y = -self.lambda_ * mul_kernel
            additive = additive - Y
            Y = self.normalize(additive)
            E = self.entropy_energy(Y, unary, kernel, self.lambda_, batch)

            if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
                break

            else:
                oldE = E.copy()
        return Y

    def forward(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ):

        # Perform normalizations required

        if self.proto_rect:
            rectifier = BDCSPN(self.softmax_temperature)
            rectifier.prototypes = compute_prototypes(support_features, support_labels)
            support = rectifier.rectify_prototypes(
                        support_features=support_features,
                        query_features=query_features,
                        support_labels=support_labels)
        else:
            support = compute_prototypes(support_features, support_labels)

        unary = torch.cdist(query_features, support) ** 2
        W = self.create_affinity(query_features.numpy())
        probs_q = self.bound_update(unary=unary.numpy(), kernel=W)
        probs_q = torch.from_numpy(probs_q)

        return None, probs_q