import torch
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
from src.models import __dict__ as BACKBONES
from src.utils.utils import strip_prefix
from loguru import logger
import numpy as np
import torch.nn.functional as F
import dataset.transforms as transforms


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
        prototypes = compute_prototypes(support_features, support_labels).to(self.device).unsqueeze(0)  # [Nk, d]

        support_features = self.transform_train(query_features)
        query_features = self.transform_train(query_features)
        query_features_2 = self.transform_train(query_features)
    
        inputs_x, targets_x = support_features.cuda(), support_labels.cuda(non_blocking=True)
        inputs_u = query_features.cuda()
        inputs_u2 = query_features_2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
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

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = self.interleave(logits, batch_size)

        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = self.semi_loss(logits_x, mixed_target[:batch_size],
                                   logits_u, mixed_target[batch_size:],
                                   epoch+batch_idx/args.val_iteration)

        loss = Lx + w * Lu
        return logits_s.softmax(-1), logits_q.softmax(-1), outlier_scores