from abc import abstractmethod
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from easyfsl.utils import (
    compute_backbone_output_shape,
    get_accuracies,
    get_task_perf,
    sliding_average,
)


class AbstractMetaLearner(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(
        self,
        backbone: nn.Module,
        tensorboard_writer: SummaryWriter = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.backbone = backbone
        self.backbone_output_shape = compute_backbone_output_shape(backbone)
        self.feature_dimension = self.backbone_output_shape[0]
        self.loss_function = nn.CrossEntropyLoss()

        self.best_validation_accuracy = 0.0
        self.best_model_state = None

        self.training_tasks_record = []
        self.training_confusion_matrix = None

        self.tensorboard_writer = tensorboard_writer
        if not tensorboard_writer:
            logger.warning(
                "No tensorboard writer specified. Training curves won't be logged."
            )

        self.device = torch.device(device=device)

    # pylint: disable=all
    @abstractmethod
    def forward(
        self,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict classification labels.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    @abstractmethod
    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using
        a forward call

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a process_support_set method."
        )

    # pylint: enable=all

    def evaluate_on_one_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> [int, int]:
        """
        Returns the number of correct predictions of query labels, and the total number of
        predictions.
        """
        self.process_support_set(
            support_images.to(self.device), support_labels.to(self.device)
        )
        return (
            torch.max(
                self(query_images.to(self.device)).detach().data,
                1,
            )[1]
            == query_labels.to(self.device)
        ).sum().item(), len(query_labels)

    def infer_on_one_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the labels of query images given a few labelled support examples.
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
            query_images: images of the query set
        Returns:
            classification scores of shape (number_of_query_images, n_way)
        """

        self.process_support_set(
            support_images.to(self.device), support_labels.to(self.device)
        )

        return self(query_images.to(self.device)).detach()

    def evaluate(self, data_loader: DataLoader) -> pd.DataFrame:
        """
        Evaluate the model on few-shot classification tasks
        Args:
            data_loader: loads data in the shape of few-shot classification tasks
        Returns:
            average classification accuracy
        """
        list_of_task_perfs = []

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph
        self.eval()
        with torch.no_grad():
            with tqdm(
                enumerate(data_loader), total=len(data_loader), desc="Evaluation"
            ) as tqdm_eval:
                for task_id, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    true_class_ids,
                ) in tqdm_eval:
                    predicted_scores = self.infer_on_one_task(
                        support_images, support_labels, query_images
                    )
                    list_of_task_perfs.append(
                        get_task_perf(
                            task_id, predicted_scores, query_labels, true_class_ids
                        ).assign(
                            task_loss=self.compute_loss(
                                predicted_scores, query_labels.to(self.device)
                            ).item()
                        )
                    )

        return pd.concat(list_of_task_perfs, ignore_index=True)

    def compute_loss(
        self, classification_scores: torch.Tensor, query_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the method's criterion to compute the loss between predicted classification scores,
        and query labels.
        We do this in a separate function because some few-shot learning algorithms don't apply
        the loss function directly to classification scores and query labels. For instance, Relation
        Networks use Mean Square Error, so query labels need to be put in the one hot encoding.
        Args:
            classification_scores: predicted classification scores of shape (n_query, n_classes)
            query_labels: ground truth labels. 1-dim tensor of length n_query

        Returns:
            loss
        """
        return self.loss_function(classification_scores, query_labels)

    def fit_on_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        true_class_ids: List[int],
        optimizer: optim.Optimizer,
    ) -> Tuple[float, float]:
        """
        Predict query set labels and updates model's parameters using classification loss
        Args:
            support_images: images of the support set
            support_labels: labels of support set images (used in the forward pass)
            query_images: query set images
            query_labels: labels of query set images (only used for loss computation)
            true_class_ids: the ids (in the dataset) of the classes present in the task
            optimizer: optimizer to train the model

        Returns:
            the value of the classification loss, and accuracy (for reporting purposes)
        """
        optimizer.zero_grad()
        self.process_support_set(
            support_images.to(self.device), support_labels.to(self.device)
        )
        classification_scores = self(query_images.to(self.device))

        task_confusion_matrix = self.compute_task_confusion_matrix(
            query_labels, classification_scores
        )
        self.update_training_tasks_record(true_class_ids, task_confusion_matrix)
        self.update_training_confusion(true_class_ids, task_confusion_matrix)

        loss = self.compute_loss(classification_scores, query_labels.to(self.device))
        loss.backward()
        optimizer.step()

        accuracy = metrics.accuracy_score(
            query_labels.cpu(), classification_scores.argmax(dim=1).cpu()
        )

        return loss.item(), accuracy

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        val_loader: DataLoader = None,
        validation_frequency: int = 1000,
        tqdm_description: str = "Meta-Training",
        epoch: Optional[int] = None,
    ):
        """
        Train the model on few-shot classification tasks.
        Args:
            train_loader: loads training data in the shape of few-shot classification tasks
            optimizer: optimizer to train the model
            val_loader: loads data from the validation set in the shape of few-shot classification
                tasks
            validation_frequency: number of training episodes between two validations
            tqdm_description: description of the progress bar following training tasks
            epoch: possibility to specify an epoch (for tensorboard writer)
        """
        log_update_frequency = 10

        all_loss = []
        all_accuracy = []
        self.initialize_training_confusion(train_loader)

        self.train()
        with tqdm(
            enumerate(train_loader), total=len(train_loader), desc=tqdm_description
        ) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                true_class_ids,
            ) in tqdm_train:
                loss_value, train_accuracy = self.fit_on_task(
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    true_class_ids,
                    optimizer,
                )
                all_loss.append(loss_value)
                all_accuracy.append(train_accuracy)

                # Log training loss in real time
                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(
                        loss=sliding_average(all_loss, log_update_frequency)
                    )

        # Validation
        validation_accuracy, validation_loss = self.validate(val_loader)

        if self.tensorboard_writer:
            if epoch is None:
                logger.warning(f"No epoch specified. Tensorboard writing could fail.")
            self.tensorboard_writer.add_scalar("Train/loss", mean(all_loss), epoch)
            self.tensorboard_writer.add_scalar("Train/acc", mean(all_accuracy), epoch)
            self.tensorboard_writer.add_scalar("Val/acc", validation_accuracy, epoch)
            self.tensorboard_writer.add_scalar("Val/loss", validation_loss, epoch)

    def fit_multiple_epochs(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        n_epochs: int,
        val_loader: DataLoader,
    ) -> List[Dict]:
        """
        Fit on the dataloader for a given number of epochs. This function can be used to update the
        dataloaders or optimizers between epochs. It is recommended to use a smaller length for the
        train than when using fit() directly.
        Using this function, validation will be performed at the end of each epoch.
        Args:
            train_loader: loads training data in the shape of few-shot classification tasks
            optimizer: optimizer to train the model
            n_epochs: number of training epochs
            val_loader: loads data from the validation set in the shape of few-shot classification
                tasks
        Returns:
            list of information about training tasks. For each training task:
                the list of true class ids
                the task-level confusion matrix (torch.Tensor)
        """
        for epoch in range(n_epochs):
            self.fit(
                train_loader=train_loader,
                optimizer=optimizer,
                val_loader=val_loader,
                validation_frequency=len(train_loader),
                tqdm_description=f"Epoch {epoch}",
                epoch=epoch,
            )

            train_loader.batch_sampler.update(
                confusion_matrix=self.training_confusion_matrix
            )

        return self.training_tasks_record

    def validate(self, val_loader: DataLoader) -> [float, float]:
        """
        Validate the model on the validation set.
        Args:
            val_loader: loads data from the validation set in the shape of few-shot classification
                tasks
        Returns:
            average classification accuracy and loss on the validation set
        """
        validation_results = self.evaluate(val_loader)
        validation_accuracy = get_accuracies(validation_results).mean()
        validation_loss = validation_results.task_loss.mean()
        print(f"Validation accuracy: {(100 * validation_accuracy):.2f}%")
        # If this was the best validation performance, we save the model state
        if validation_accuracy > self.best_validation_accuracy:
            print("Best validation accuracy so far!")
            self.best_model_state = self.state_dict()

        return validation_accuracy, validation_loss

    def _check_that_best_state_is_defined(self):
        """
        Raises:
            AttributeError: if self.best_model_state is None, i.e. if no best sate has been
                defined yet.
        """
        if not self.best_model_state:
            raise AttributeError(
                "There is not best state defined for this model. "
                "You need to train the model using validation to define a best state."
            )

    def restore_best_state(self):
        """
        Retrieves the state (i.e. a dictionary of model parameters) of the model at the time it
        obtained its best performance on the validation set.
        """
        self._check_that_best_state_is_defined()
        self.load_state_dict(self.best_model_state)

    def dump_best_state(self, output_path: Union[Path, str]):
        """
        Retrieves the state (i.e. a dictionary of model parameters) of the model at the time it
        obtained its best performance on the validation set.
        Args:
            output_path: path to the output file. Common practice in PyTorch is to save models
                using either a .pt or .pth file extension.
        """
        self._check_that_best_state_is_defined()
        torch.save(self.best_model_state, output_path)

    @staticmethod
    def compute_task_confusion_matrix(ground_truth_labels, classification_scores):
        """
        Compute the task-level confusion matrix, of shape (n_way, n_way), from the model's
        predictions in one task.
        Args:
            ground_truth_labels: ground truth labels
            classification_scores: model's prediction

        Returns:
            task-level confusion matrix
        """
        return torch.tensor(
            metrics.confusion_matrix(
                ground_truth_labels.cpu(), classification_scores.argmax(dim=1).cpu()
            )
        )

    def initialize_training_confusion(self, train_loader: DataLoader):
        """
        Initialize the training confusion matrix as a square 2-dim tensor. Its size is the total
        number of classes in the dataset delivered by the train loader.
        Args:
            train_loader: training data loader
        """
        total_number_of_classes = np.max(train_loader.dataset.labels) + 1
        self.training_confusion_matrix = torch.zeros(
            (total_number_of_classes, total_number_of_classes)
        )

    def update_training_tasks_record(
        self,
        true_class_ids: List[int],
        task_confusion_matrix: torch.Tensor,
    ):
        """
        Record the true class ids and the task-level confusion matrix of a given task.
        Args:
            true_class_ids: the ids (in the dataset) of the classes present in the task
            task_confusion_matrix: task-level confusion matrix of shape (n_way, n_way)
        """

        self.training_tasks_record.append(
            {
                "true_class_ids": true_class_ids,
                "task_confusion_matrix": task_confusion_matrix,
            }
        )

    def update_training_confusion(
        self,
        true_class_ids: List[int],
        task_confusion_matrix: torch.Tensor,
    ):
        """
        Update the global training confusion matrix from a task-level confusion matrix
        Args:
            true_class_ids: the ids (in the dataset) of the classes present in the task
            task_confusion_matrix: task-level confusion matrix of shape (n_way, n_way)
        """

        for (local_label1, true_label1) in enumerate(true_class_ids):
            for (local_label2, true_label2) in enumerate(true_class_ids):
                self.training_confusion_matrix[
                    true_label1, true_label2
                ] += task_confusion_matrix[local_label1, local_label2]
