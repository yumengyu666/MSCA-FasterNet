"""PlantVillage Dataset Loader.

PlantVillage: An Open Access Dataset for Plant Disease Identification
- 38 classes (including healthy leaves)
- ~54,000 images total
- No official train/val/test split; we use 8:1:1 random split
- Source: https://www.kaggle.com/datasets/emmarex/plantdisease

Expected directory structure:
    data/PlantVillage/
    ├── PlantVillage/
    │   ├── Tomato___Bacterial_spot/
    │   │   ├── image1.JPG
    │   │   └── ...
    │   ├── Tomato___Early_blight/
    │   └── ...
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from typing import Optional, Callable, Tuple, List
import torchvision.transforms as transforms


class PlantVillageDataset(Dataset):
    """PlantVillage Plant Disease Dataset.

    Args:
        root_dir: Root directory of PlantVillage dataset.
        split: One of 'train', 'val', 'test', or 'all'.
        transform: Image transformations.
        val_ratio: Validation ratio from training set.
        test_ratio: Test ratio from training set.
        seed: Random seed for reproducible splits.
        return_path: Whether to return image path.
    """

    CLASS_NAMES = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
        "Cherry___healthy", "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
        "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
        "Peach___healthy", "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
        "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
        "Strawberry___healthy", "Tomato___Bacterial_spot",
        "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
    ]

    def __init__(
        self,
        root_dir: str = "data/PlantVillage",
        split: str = "train",
        transform: Optional[Callable] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        return_path: bool = False,
    ):
        super().__init__()
        assert split in ["train", "val", "test", "all"], f"Invalid split: {split}"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_path = return_path

        # Auto-discover class folders
        self.classes, self.class_to_idx = self._discover_classes()
        self.num_classes = len(self.classes)

        # Load all samples
        all_samples = self._load_all_samples()

        # Split dataset
        if split == "all":
            self.samples = all_samples
        else:
            self.samples = self._split_data(all_samples, val_ratio, test_ratio, seed)

        print(f"PlantVillage {self.split}: {len(self.samples)} images, {self.num_classes} classes")

    def _discover_classes(self) -> Tuple[List[str], dict]:
        """Auto-discover class folders."""
        data_dir = os.path.join(self.root_dir, "PlantVillage")
        if not os.path.exists(data_dir):
            data_dir = self.root_dir

        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _load_all_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and labels."""
        samples = []
        data_dir = os.path.join(self.root_dir, "PlantVillage")
        if not os.path.exists(data_dir):
            data_dir = self.root_dir

        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            label = self.class_to_idx[cls_name]
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    samples.append((img_path, label))

        return samples

    def _split_data(
        self, all_samples: List[Tuple[str, int]],
        val_ratio: float, test_ratio: float, seed: int
    ) -> List[Tuple[str, int]]:
        """Split data into train/val/test (8:1:1) with per-class stratification."""
        rng = np.random.RandomState(seed)

        # Group by class
        class_samples = {}
        for path, label in all_samples:
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append((path, label))

        selected = []
        for label, samples in class_samples.items():
            indices = list(range(len(samples)))
            rng.shuffle(indices)

            n_total = len(samples)
            n_test = max(1, int(n_total * test_ratio))
            n_val = max(1, int(n_total * val_ratio))
            n_train = n_total - n_test - n_val

            if self.split == "train":
                selected_idx = indices[:n_train]
            elif self.split == "val":
                selected_idx = indices[n_train:n_train + n_val]
            elif self.split == "test":
                selected_idx = indices[n_train + n_val:]
            else:
                selected_idx = indices

            selected.extend([samples[i] for i in selected_idx])

        return selected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, img_path
        return image, label

    def get_class_distribution(self) -> torch.Tensor:
        """Get the number of samples per class."""
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def get_plantvillage_transforms(split: str = "train", input_size: int = 224):
    """Get standard transforms for PlantVillage dataset."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])


def build_plantvillage_dataloader(
    root_dir: str = "data/PlantVillage",
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    input_size: int = 224,
    return_path: bool = False,
) -> DataLoader:
    """Build DataLoader for PlantVillage dataset."""
    transform = get_plantvillage_transforms(split, input_size)
    dataset = PlantVillageDataset(root_dir, split, transform, return_path=return_path)

    shuffle = True if split == "train" else False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return dataloader
