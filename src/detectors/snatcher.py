import torch
from .abstract_detector import AbstractDetector
from easyfsl.utils import compute_prototypes
from src.constants import MISC_MODULES, TRAINED_MODELS_DIR
from src.utils.utils import strip_prefix
from loguru import logger


class SNATCHERF(AbstractDetector):
    """
    """
    def __init__(self, args, temperature):

        self.temperature = 64.
        self.device = args.device

        # Load attention module
        self.attn_model = MISC_MODULES['snatcher_f'](args)
        weights = TRAINED_MODELS_DIR / args.training / f"{args.backbone}_{args.dataset}_{args.model_source}.pth"
        state_dict = torch.load(weights)['params']
        state_dict = strip_prefix(state_dict, "slf_attn.")
        missing_keys, unexpected = self.attn_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded Snatcher attention module. \n Missing keys: {missing_keys} \n Unexpected keys: {unexpected}")

        self.attn_model.eval()
        self.attn_model = self.attn_model.to(self.device)

    def fit(self, support_features, support_labels):
        self.prototypes = compute_prototypes(support_features, support_labels).to(self.device).unsqueeze(0)  # [Nk, d]

    def decision_function(self, query_features):
        """
        query_features [Ns, d]
        """
        query_features = query_features.to(self.device)

        proto = self.attn_model(self.prototypes, self.prototypes, self.prototypes)[0][0]  # [K, d]

        logits = - torch.cdist(query_features, proto) ** 2 / self.temperature  # [Nq, K]

        """ Snatcher """
        outlier_scores = torch.zeros(logits.size(0))
        with torch.no_grad():
            for j in range(logits.size(0)):
                pproto = self.prototypes.clone().detach()  # [K, d]
                """ Algorithm 1 Line 1 """
                c = logits[j].argmax(0)
                """ Algorithm 1 Line 2 """
                pproto[0, c] = query_features[j]
                """ Algorithm 1 Line 3 """
                pproto = self.attn_model(pproto, pproto, pproto)[0]
                pdiff = (pproto - proto).pow(2).sum(-1).sum() / self.temperature
                """ pdiff: d_SnaTCHer in Algorithm 1 """
                outlier_scores[j] = pdiff
        return outlier_scores