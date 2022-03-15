from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes
import numpy as np

class FEAT(FewShotMethod):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        else:
            raise ValueError('')
        
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)          
        
    def forward(self, support_features, query_features, support_labels, **kwargs):

        emb_dim = support_features.size(-1)

        # get mean of the support
        proto = compute_prototypes(support_features, support_labels)  # NK x d

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        proto = self.slf_attn(proto, proto, proto)        
        if self.args.use_euclidean:
            query = query_features  # (Nbatch*Nq*Nw, d)
            # proto = proto.unsqueeze(0).expand(num_query, num_proto, emb_dim).contiguous()
            # proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            # logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
            logits_s = - (torch.cdist(proto, support_features) ** 2 / self.temperature)  # [Nq, K]
            logits_q = - (torch.cdist(proto, query_features) ** 2 / self.temperature)  # [Nq, K]
        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            logits_s = torch.bmm(query, proto.t()) / self.args.temperature  # [Nq, K]
            logits_q = torch.bmm(query, proto.t()) / self.args.temperature  # [Nq, K]

        return logits_s.softmax(-1).cpu(), logits_q.softmax(-1).cpu()