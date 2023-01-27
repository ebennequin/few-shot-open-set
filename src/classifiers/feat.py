from typing import Tuple, List
import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from easyfsl.utils import compute_prototypes
import numpy as np
from pathlib import Path

from .abstract import FewShotMethod
from src.models import __dict__ as BACKBONES
from src.utils.utils import strip_prefix


class FEAT(FewShotMethod):
    def __init__(self, args, use_euclidean: bool, temperature: float):
        super().__init__(args)
        self.use_euclidean = use_euclidean
        self.temperature = temperature

        # Load attention module
        if args.backbone == "resnet12":
            hdim = 640
        elif args.backbone == "resnet18":
            hdim = 512
        elif args.backbone == "wrn2810":
            hdim = 640
        else:
            raise ValueError("")
        self.device = args.device
        self.attn_model = BACKBONES["MultiHeadAttention"](
            args, 1, hdim, hdim, hdim, dropout=0.5
        )
        weights = (
            Path(args.data_dir)
            / "models"
            / args.training
            / f"{args.backbone}_{args.src_dataset}_{args.model_source}.pth"
        )
        state_dict = torch.load(weights)["params"]
        state_dict = strip_prefix(state_dict, "module.")
        state_dict = strip_prefix(state_dict, "slf_attn.")
        missing_keys, unexpected = self.attn_model.load_state_dict(
            state_dict, strict=False
        )
        logger.info(
            f"Loaded Snatcher attention module. \n Missing keys: {missing_keys} \n Unexpected keys: {unexpected}"
        )

        self.attn_model.eval()
        self.attn_model = self.attn_model.to(self.device)

    def forward(self, support_features, query_features, support_labels, **kwargs):
        support_features, query_features = (
            support_features.cuda(),
            query_features.cuda(),
        )
        support_labels, query_labels = (
            support_labels.cuda(),
            kwargs["query_labels"].cuda(),
        )

        # get mean of the support
        proto = compute_prototypes(support_features, support_labels).unsqueeze(
            0
        )  # NK x d

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        with torch.no_grad():
            proto = self.attn_model(proto, proto, proto)[0][0]

            if self.use_euclidean:
                logits_s = -(
                    torch.cdist(support_features, proto) ** 2 / self.temperature
                )  # [Nq, K]
                logits_q = -(
                    torch.cdist(query_features, proto) ** 2 / self.temperature
                )  # [Nq, K]
            else:
                proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
                logits_s = (
                    torch.bmm(support_features, proto.t()) / self.temperature
                )  # [Nq, K]
                logits_q = (
                    torch.bmm(query_features, proto.t()) / self.temperature
                )  # [Nq, K]

        return logits_s.softmax(-1).cpu(), logits_q.softmax(-1).cpu()
