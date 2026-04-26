"""Weighted sampler for class-balanced training."""

import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional
import numpy as np


class WeightedSamplerBuilder:
    """Builder for WeightedRandomSampler from dataset labels.

    Creates a sampler that oversamples minority classes and undersamples
    majority classes to achieve balanced training.
    """

    def from_dataset(self, dataset) -> Optional[Sampler]:
        """Build a WeightedRandomSampler from a dataset.

        Args:
            dataset: Dataset with __getitem__ returning (image, label, ...).

        Returns:
            WeightedRandomSampler instance, or None if dataset is empty.
        """
        # Extract labels
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i][:2]
            labels.append(label)

        if len(labels) == 0:
            return None

        labels = np.array(labels)
        return self.from_labels(labels)

    def from_labels(self, labels: np.ndarray) -> Sampler:
        """Build a WeightedRandomSampler from label array.

        Args:
            labels: Array of class labels.

        Returns:
            WeightedRandomSampler instance.
        """
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
        sample_weights = class_weights[labels]
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True,
        )

        return sampler
