"""
See original implementation at
https://github.com/facebookresearch/low-shot-shrink-hallucinate
"""

import torch
import torch.nn as nn

from easyfsl.methods import AbstractMetaLearner


class MatchingNetworks(AbstractMetaLearner):
    """
    Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
    "Matching networks for one shot learning." (2016)
    https://arxiv.org/pdf/1606.04080.pdf

    Matching networks extract feature vectors for both support and query images. Then they refine
    these feature by using the context of the whole support set, using LSTMs. Finally they compute
    query labels using their cosine similarity to support images.
    """

    def __init__(self, **kwargs):
        """
        Build Matching Networks by calling the constructor of AbstractMetaLearner.

        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(**kwargs)

        if len(self.backbone_output_shape) != 1:
            raise ValueError(
                "Illegal backbone for Matching Networks. "
                "Expected output for an image is a 1-dim tensor."
            )

        # The model outputs log-probabilities, so we use the negative log-likelihood loss
        self.loss_function = nn.NLLLoss()

        # These modules refine support and query feature vectors
        # using information from the whole support set
        self.support_features_encoder = nn.LSTM(
            input_size=self.feature_dimension,
            hidden_size=self.feature_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.query_features_encoding_cell = nn.LSTMCell(
            self.feature_dimension * 2, self.feature_dimension
        )

        self.softmax = nn.Softmax(dim=1)

        # Here we create the fields so that the model can store
        # the computed information from one support set
        self.contextualized_support_features = None
        self.one_hot_support_labels = None

    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Overrides process_support_set of AbstractMetaLearner.
        Extract features from the support set with full context embedding.
        Store contextualized feature vectors, as well as support labels in the one hot format.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        support_features = self.backbone(support_images)
        self.contextualized_support_features = self.encode_support_features(
            support_features
        )

        self.one_hot_support_labels = nn.functional.one_hot(support_labels).float()

    def forward(self, query_images: torch.Tensor) -> torch.Tensor:
        """
        Overrides method forward in AbstractMetaLearner.
        Predict query labels based on their cosine similarity to support set features.
        Classification scores are log-probabilities.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """

        # Refine query features using the context of the whole support set
        contextualized_query_features = self.encode_query_features(
            self.backbone(query_images)
        )

        # Compute the matrix of cosine similarities between all query images
        # and normalized support images
        # Following the original implementation, we don't normalize query features to keep
        # "sharp" vectors after softmax (if normalized, all values tend to be the same)
        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                nn.functional.normalize(self.contextualized_support_features).T
            )
        )

        # Compute query log probabilities based on cosine similarity to support instances
        # and support labels
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return log_probabilities

    def encode_support_features(
        self,
        support_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine support set features by putting them in the context of the whole support set,
        using a bidirectional LSTM.
        Args:
            support_features: output of the backbone

        Returns:
            contextualised support features, with the same shape as input features
        """

        # Since the LSTM is bidirectional, hidden_state is of the shape
        # [number_of_support_images, 2 * feature_dimension]
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[
            0
        ].squeeze(0)

        # Following the paper, contextualized features are computed by adding original features, and
        # hidden state of both directions of the bidirectional LSTM.
        contextualized_support_features = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )

        return contextualized_support_features

    def encode_query_features(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Refine query set features by putting them in the context of the whole support set,
        using attention over support set features.
        Args:
            query_features: output of the backbone

        Returns:
            contextualized query features, with the same shape as input features
        """

        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        # We do as many iterations through the LSTM cell as there are query instances
        # Check out the paper for more details about this!
        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1)

            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features

        return hidden_state
