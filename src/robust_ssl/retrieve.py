from .abstract import AllInOne
import torch.nn as nn
import torch.utils.data as data
from cords.selectionstrategies.helpers.ssl_lib.consistency.builder import gen_consistency
from cords.selectionstrategies.helpers.ssl_lib.algs.builder import gen_ssl_alg
import torch.utils.data as data
from cords.utils.data.dataloader.SSL.adaptive import RETRIEVEDataLoader
from dotmap import DotMap
import logging
import numpy, random, time, json, copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cords.utils.models import utils as model_utils
from cords.selectionstrategies.helpers.ssl_lib.param_scheduler import scheduler
import time
from copy import deepcopy

class Retrieve(AllInOne):
    """
    """
    def __init__(self, args, T: float, alpha: float, lambda_u: float):

        super().__init__()
        
    def __call__(self, support_features, support_labels, query_features, **kwargs):
        """
        query_features [Ns, d]
        """

        # ===== Prepare models =======

        model = kwargs['feature_extractor']
        average_model = deepcopy(model)
        num_classes = support_labels.unique().size(0)
        classification_head = nn.Linear(model.layers_dims[-1], num_classes)

        # ==== Get consistency loss function =====

        consistency = gen_consistency(cfg.ssl_args.consis, cfg)
        consistency_nored = gen_consistency(cfg.ssl_args.consis + '_red', cfg)

        # ==== Define SSL algorithm =====

        ssl_alg = gen_ssl_alg(cfg.ssl_args.alg, cfg)

        # ==== Create data loaders ====
        ult_data = data.TensorDataset(query_features, kwargs['query_labels'])
        ult_seq_loader = data.DataLoader(ult_data,
                                         batch_size=cfg.dataloader.ul_batch_size,
                                         shuffle=False, pin_memory=True)

        lt_data = data.TensorDataset(support_features, support_labels)
        lt_seq_loader = data.DataLoader(lt_data,
                                        batch_size=self.batch_size,
                                        shuffle=False, pin_memory=True)

        test_loader = data.DataLoader(
            data.TensorDataset(query_features),
            1,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        cfg.dss_args.model = model
        cfg.dss_args.tea_model = None
        cfg.dss_args.ssl_alg = ssl_alg
        cfg.dss_args.loss = consistency_nored
        cfg.dss_args.num_classes = num_classes
        cfg.dss_args.num_iters = max_iteration
        cfg.dss_args.eta = cfg.optimizer.lr
        cfg.dss_args.device = self.device

        ult_loader = RETRIEVEDataLoader(ult_seq_loader,
                                        lt_seq_loader,
                                        cfg.dss_args,
                                        batch_size=cfg.dataloader.ul_batch_size,
                                        pin_memory=cfg.dataloader.pin_memory,
                                        num_workers=0)

        # === Get optimizer ===

        if cfg.optimizer.type == "sgd":
            optimizer = optim.SGD(
                        model.parameters(), cfg.optimizer.lr, cfg.optimizer.momentum, 
                        weight_decay=cfg.optimizer.weight_decay, nesterov=cfg.optimizer.nesterov)
        elif cfg.optimizer.type == "adam":
            optimizer = optim.Adam(
                model.parameters(), cfg.optimizer.lr, (cfg.optimizer.momentum, 0.999), 
                weight_decay=cfg.optimizer.weight_decay)
        else:
            raise NotImplementedError

        # === Get scheduler ===

        # set lr scheduler
        if cfg.scheduler.lr_decay == "cos":
            if cfg.dss_args.type == 'Full':
                lr_scheduler = scheduler.CosineAnnealingLR(optimizer, max_iteration)
            else:
                lr_scheduler = scheduler.CosineAnnealingLR(optimizer,
                                                           cfg.train_args.iteration * cfg.dss_args.fraction)
        elif cfg.scheduler.lr_decay == "step":
            # TODO: fixed milestones
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400000, ], cfg.scheduler.lr_decay_rate)
        else:
            raise NotImplementedError

        # ==== Beginning training ====

        model.train()
        iter_count = 1

        # Start training until maximum number of iterations are reached
        while iter_count <= self.max_iteration:

            lt_loader = data.DataLoader(
                lt_data,
                self.batch_size,
                sampler=dataset_utils.InfiniteSampler(len(lt_data), len(list(
                        ult_loader.batch_sampler)) * self.batch_size),
                num_workers=0
            )

            # Enumerate on batches of labeled and unlabeled data. 
            # Note that the ult_loader enumerates only on subsets of unlabeled data selected by RETRIEVE
            for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_loader)):
                if iter_count > self.max_iteration:
                    break
                l_aug, labels = l_data
                ul_w_aug, ul_s_aug, _, weights = ul_data
                params = self.param_update(
                        iter_count, model, optimizer, ssl_alg,
                        consistency, l_aug.to(self.train_args.device), ul_w_aug.to(self.device),
                        ul_s_aug.to(self.device), labels.to(self.device),
                        average_model, weights=weights.to(self.device))
                
                lr_scheduler.step()
                iter_count += 1

    def param_update(self,
                     cfg,
                     cur_iteration,
                     model,
                     classification_head,
                     optimizer,
                     ssl_alg,
                     consistency,
                     labeled_data,
                     ul_weak_data,
                     ul_strong_data,
                     labels,
                     average_model,
                     weights=None
                     ):
        # Concatenate labeled data, weakly augmented, and strongly augmented unlabeled data
        all_data = torch.cat([labeled_data, ul_weak_data, ul_strong_data], 0)
        forward_func = lambda x: classification_head(model(x, ['last'])['last'])
        stu_logits = forward_func(all_data)
        labeled_preds = stu_logits[:labeled_data.shape[0]]

        # Separate weak unlabeled logits, and strong unlabeled logits
        stu_unlabeled_weak_logits, stu_unlabeled_strong_logits = torch.chunk(stu_logits[labels.shape[0]:], 2, dim=0)
        
        # Use training signal
        L_supervised = F.cross_entropy(labeled_preds, labels)

        # IF SSL coefficient is greater than zero, calculate the consistency loss
        if cfg.ssl_args.coef > 0:
            # get target values
            t_forward_func = forward_func
            tea_unlabeled_weak_logits = stu_unlabeled_weak_logits

            # calculate consistency loss
            model.update_batch_stats(False)
            y, targets, mask = ssl_alg(
                stu_preds=stu_unlabeled_strong_logits,
                tea_logits=tea_unlabeled_weak_logits.detach(),
                w_data=ul_strong_data,
                subset=False,
                stu_forward=forward_func,
                tea_forward=t_forward_func
            )
            model.update_batch_stats(True)

            # calculate weighted consistency loss
            if weights is None:
                L_consistency = consistency(y, targets, mask,
                                            weak_prediction=tea_unlabeled_weak_logits.softmax(1))
            else:
                L_consistency = consistency(y, targets, mask * weights,
                                            weak_prediction=tea_unlabeled_weak_logits.softmax(1))
        else:
            L_consistency = torch.zeros_like(L_supervised)
            mask = None

        # calculate total loss (consistency with entropy minimization)
        coef = scheduler.exp_warmup(cfg.ssl_args.coef, int(cfg.scheduler.warmup_iter), cur_iteration + 1)
        loss = L_supervised + coef * L_consistency
        if cfg.ssl_args.em > 0:
            loss -= cfg.ssl_args.em * \
                    (stu_unlabeled_weak_logits.softmax(1) * F.log_softmax(stu_unlabeled_weak_logits, 1)).sum(1).mean()

        # update parameters
        cur_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss.backward()
        if cfg.optimizer.weight_decay > 0:
            decay_coeff = cfg.optimizer.weight_decay * cur_lr
            model_utils.apply_weight_decay(model.modules(), decay_coeff)
        optimizer.step()
        
        # update evaluation model's parameters by exponential moving average
        if cfg.ssl_eval_args.weight_average:
            model_utils.ema_update(
                average_model, model, cfg.ssl_eval_args.wa_ema_factor,
                cfg.optimizer.weight_decay * cur_lr if cfg.ssl_eval_args.wa_apply_wd else None)