import torch
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
from src.models import __dict__ as BACKBONES
from src.utils.utils import strip_prefix
from loguru import logger
import numpy as np
import torch.nn.functional as F
import dataset.transforms as transforms
import torch.nn as nn
from copy import deepcopy
import torchvision
import torch.utils.data as data
from skimage.filters import threshold_otsu


class MTC(AllInOne):
    """
    """
    def __init__(self, args, T: float, alpha: float, lambda_u: float):

        super().__init__()
        self.T = T
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.RandomFlip(),
            transforms.ToTensor(),
            ])
        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def semi_loss(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        def linear_rampup(current, rampup_length=16):
            if rampup_length == 0:
                return 1.0
            else:
                current = np.clip(current / rampup_length, 0.0, 1.0)
                return float(current)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch)

    def __call__(self, support_features, support_labels, query_features, **kwargs):
        """
        query_features [Ns, d]
        """

        model = kwargs['feature_extractor']
        num_classes = support_labels.unique().size(0)
        outlier_head = nn.Linear(model.layers_dims[-1])
        classification_head = nn.Linear(model.layers_dims[-1], num_classes)

        train_criterion = SemiLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([model.parameters(), outlier_head, classification_head], lr=self.lr)

        # ====== Prepare data =========

        domain_dataset = DomainDataset(support_features, support_labels, query_features)
        unlabelled_dataset = UnlabelledDataset(query_features, transform=TransformTwice(self.transform_train))
        val_dataset_s = UnlabelledDataset(support_features, transform=self.transform_val)
        val_dataset_q = UnlabelledDataset(query_features, transform=self.transform_val)
        labeled_dataset = LabelledDataset(support_features, support_labels, transform=self.transform_train)

        domain_trainloader = data.DataLoader(domain_dataset, batch_size=64, shuffle=True)
        labeled_trainloader = data.DataLoader(labeled_dataset, batch_size=64, shuffle=True)
        unlabeled_trainloader = data.DataLoader(unlabelled_dataset, batch_size=64, shuffle=True)
        val_loader_q = data.DataLoader(val_dataset_q, batch_size=64, shuffle=False)
        val_loader_s = data.DataLoader(val_dataset_s, batch_size=64, shuffle=False)

        # ====== Pre-train outlier detector =========

        ema_model = deepcopy(model).detach()
        ema_optimizer = WeightEMA(model, ema_model, alpha=self.ema_decay)

        for epoch in range(self.pre_epochs):

            train_loss, train_loss_x, train_loss_u = domain_train(domain_trainloader,
                                                                  model,
                                                                  outlier_head,
                                                                  optimizer,
                                                                  ema_optimizer,
                                                                  train_criterion,
                                                                  epoch)

        # ===== Train classification part =====

        for epoch in range(self.epochs):

            train_loss, train_loss_x, train_loss_u, prec, recall = self.train(domain_trainloader,
                                                                              labeled_trainloader,
                                                                              unlabeled_trainloader,
                                                                              model,
                                                                              outlier_head,
                                                                              classification_head,
                                                                              optimizer,
                                                                              ema_optimizer,
                                                                              train_criterion,
                                                                              epoch,
                                                                              )

        soft_preds_s, soft_preds_q, outlier_scores = self.validate(val_loader_s,
                                                                   val_loader_q,
                                                                   model,
                                                                   outlier_head,
                                                                   classification_head)

        return soft_preds_s, soft_preds_q, outlier_scores

    def train(self, domain_trainloader, labeled_trainloader, unlabeled_trainloader,
              model, outlier_head, classification_head, optimizer,
              ema_optimizer, criterion, epoch):

        labeled_train_iter = iter(labeled_trainloader)

        train_iter = iter(domain_trainloader)

        results = np.zeros((len(domain_trainloader.dataset)), dtype=np.float32)

        # Get OOD scores of unlabeled samples
        n_labeled = len(labeled_trainloader.dataset)
        weights = domain_trainloader.dataset.soft_labels[n_labeled:].copy()

        # Calculate threshold by otsu
        th = threshold_otsu(weights.reshape(-1, 1))

        # Select samples having small OOD scores as ID data
        '''
        Attention:
        Weights is the (1 - OOD score) in this implement, which is different from the paper.
        So a larger weight means the data is more likely to be ID data.
        '''
        subset_indexs = np.arange(len(unlabeled_trainloader.dataset))[weights >= th]

        unlabeled_trainloader = data.DataLoader(data.Subset(unlabeled_trainloader.dataset, subset_indexs),
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                drop_last=True)

        unlabeled_train_iter = iter(unlabeled_trainloader)

        model.train()
        for batch_idx in range(self.val_iteration):
            try:
                inputs, domain_labels, indexs = train_iter.next()
            except:
                train_iter = iter(domain_trainloader)
                inputs, domain_labels, indexs = train_iter.next()

            try:
                inputs_x, targets_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()

            try:
                (inputs_u, inputs_u2) = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2) = unlabeled_train_iter.next()

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1), 1)

            inputs = inputs.cuda()
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            domain_labels = domain_labels.cuda()

            model.apply(set_bn_eval)
            features = model(inputs, ['last'])['last']
            model.apply(set_bn_train)

            outlier_logits = outlier_head(features)
            probs = torch.sigmoid(outlier_logits).view(-1)
            Ld = F.binary_cross_entropy_with_logits(outlier_logits, domain_labels.view(-1,1))

            results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u, _ = classification_head(model(inputs_u, ['last'])['last'])
                outputs_u2, _ = classification_head(model(inputs_u2, ['last'])['last'])
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p**(1/self.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l = np.random.beta(self.alpha, self.alpha)

            l = max(l, 1-l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = self.interleave(mixed_input, batch_size)

            logits = [model(mixed_input[0])[0]]
            for input in mixed_input[1:]:
                logits.append(model(input)[0])

            # put interleaved samples back
            logits = self.interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u,
                                  mixed_target[batch_size:],
                                  epoch+batch_idx / self.val_iteration)

            loss = Ld + Lx + w * Lu

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

        domain_trainloader.dataset.label_update(results)
        ema_optimizer.step(bn=True)

    def validate(self, val_loader_s, val_loader_q, model, outlier_head, classification_head):

        # switch to evaluate mode
        model.eval()

        outlier_scores = []
        soft_preds_s = []
        soft_preds_q = []

        with torch.no_grad():
            for inputs in val_loader_s:

                inputs = inputs.cuda()
                
                # compute output
                features = model(inputs, ['last'])['last']
                soft_preds_s.append(classification_head(features))

            for inputs in val_loader_q:

                inputs = inputs.cuda()
                
                # compute output
                features = model(inputs, ['last'])['last']
                outlier_scores.append(outlier_head(features))
                soft_preds_q.append(classification_head(features))

        return soft_preds_s, soft_preds_q, 1 - outlier_scores


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = deepcopy(model)
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)


class SemiLoss(object):

    def __init__(self, lambda_u):
        self.lambda_u = lambda_u

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean(torch.mean((probs_u - targets_u)**2, dim=1))

        return Lx, Lu, self.lambda_u * linear_rampup(epoch)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def domain_train(domain_trainloader, model, outlier_head, optimizer, ema_optimizer, criterion, epoch):

    results = np.zeros((len(domain_trainloader.dataset)), dtype=np.float32)
    model.train()
    for batch_idx, (inputs, _, domain_labels, indexs) in enumerate(domain_trainloader):
        # measure data loading time

        inputs, domain_labels = inputs.cuda(), domain_labels.cuda(non_blocking=True)

        features = model(inputs, ['last'])['last']
        logits = outlier_head(features)

        probs = torch.sigmoid(logits).view(-1)
        Ld = F.binary_cross_entropy_with_logits(logits, domain_labels.view(-1, 1))

        results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

        loss = Ld

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    domain_trainloader.dataset.label_update(results)


class DomainDataset(torch.utils.data.Dataset):

    def __init__(self, support_images, support_labels, query_images, start_label=0, transform=None):
        self.transform = transform
        self.support_images = support_images
        self.targets_x = support_labels

        self.query_images = query_images

        self.soft_labels = np.zeros((len(self.support_images)+len(self.query_images)), dtype=np.float32)
        for idx in range(len(self.support_images)+len(self.query_images)):
            if idx < len(self.support_images):
                self.soft_labels[idx] = 1.0
            else:
                self.soft_labels[idx] = start_label
        self.prediction = np.zeros((len(self.support_images)+len(self.query_images), 10), dtype=np.float32)
        self.prediction[:len(self.support_images), :] = 1.0
        self.count = 0

    def __len__(self):
        return len(self.support_images) + len(self.query_images)

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[len(self.support_images):, idx] = results[len(self.support_images):]

        if self.count >= 10:
            self.soft_labels = self.prediction.mean(axis=1)

    def __getitem__(self, index):
        if index < len(self.support_images):
            img = self.support_images[index], self.targets_x[index]

        elif index < len(self.support_images) + len(self.query_images):
            img = self.query_images[index-len(self.support_images)]

        if self.transform is not None:
            img = self.transform(img)

        return img, self.soft_labels[index], index


class UnlabelledDataset(torch.utils.data.Dataset):

    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        return img


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = False


def set_bn_train(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = True


class LabelledDataset(torch.utils.data.Dataset):

    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.data[index]
        tgt = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, tgt