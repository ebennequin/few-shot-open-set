import torch
from pathlib import Path
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
from src.models import __dict__ as BACKBONES
from src.utils.utils import strip_prefix
from loguru import logger


class SnatcherF(AllInOne):
    """ """

    def __init__(self, args, temperature: float):
        self.temperature = 64.0
        self.device = args.device
        self.works_on_features = True
        self.args = args

        # Load attention module
        if args.backbone == "resnet12":
            hdim = 640
        elif args.backbone == "resnet18":
            hdim = 512
        elif args.backbone == "wrn2810":
            hdim = 640
        else:
            raise ValueError("")
        weights = (
            Path(args.data_dir)
            / "models"
            / args.training
            / f"{args.backbone}_{args.src_dataset}_{args.model_source}.pth"
        )
        state_dict = torch.load(weights)["params"]
        state_dict = strip_prefix(state_dict, "module.")
        state_dict = strip_prefix(state_dict, "slf_attn.")

        self.attn_model = BACKBONES["MultiHeadAttention"](
            self.args, 1, hdim, hdim, hdim, dropout=0.5
        )
        missing_keys, unexpected = self.attn_model.load_state_dict(
            state_dict, strict=False
        )
        self.attn_model = self.attn_model.to(self.device)
        self.attn_model.eval()
        logger.info(
            f"Loaded Snatcher attention module. \n Missing keys: {missing_keys} \n Unexpected keys: {unexpected}"
        )

    def __call__(self, support_features, support_labels, query_features, **kwargs):
        """
        query_features [Ns, d]
        """

        support_features = support_features.to(self.device)
        query_features = query_features.to(self.device)

        prototypes = (
            compute_prototypes(support_features, support_labels)
            .to(self.device)
            .unsqueeze(0)
        )  # [Nk, d]

        proto = self.attn_model(prototypes, prototypes, prototypes)[0][0]  # [K, d]

        logits_s = (
            -torch.cdist(support_features, proto) ** 2 / self.temperature
        )  # [Nq, K]
        logits_q = (
            -torch.cdist(query_features, proto) ** 2 / self.temperature
        )  # [Nq, K]

        """ Snatcher """
        outlier_scores = torch.zeros(logits_q.size(0))
        with torch.no_grad():
            for j in range(logits_q.size(0)):
                pproto = prototypes.clone().detach()  # [K, d]
                """ Algorithm 1 Line 1 """
                c = logits_q[j].argmax(0)
                """ Algorithm 1 Line 2 """
                pproto[0, c] = query_features[j]
                """ Algorithm 1 Line 3 """
                pproto = self.attn_model(pproto, pproto, pproto)[0]
                pdiff = (pproto - proto).pow(2).sum(-1).sum() / self.temperature
                """ pdiff: d_SnaTCHer in Algorithm 1 """
                outlier_scores[j] = pdiff
        return (
            logits_s.softmax(-1).cpu(),
            logits_q.softmax(-1).cpu(),
            outlier_scores.cpu(),
        )
