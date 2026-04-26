"""IP102 Dataset Loader.

IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition
- 102 classes of agricultural insect pests
- ~75,000 images total
- Official train/val/test split provided
- Source: CVPR 2019, https://github.com/xpwu95/IP102

Expected directory structure:
    data/IP102/
    ├── ip102_v1.1/
    │   ├── images/
    │   │   ├── 001/
    │   │   │   ├── 00101.jpg
    │   │   │   ├── 00102.jpg
    │   │   │   └── ...
    │   │   ├── 002/
    │   │   └── ...
    │   └── list/
    │       ├── train.txt
    │       ├── val.txt
    │       └── test.txt
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Callable, Tuple, List
import torchvision.transforms as transforms


class IP102Dataset(Dataset):
    """IP102 Insect Pest Recognition Dataset.

    Args:
        root_dir: Root directory of IP102 dataset.
        split: One of 'train', 'val', 'test'.
        transform: Image transformations.
        return_path: Whether to return image path (for visualization).
    """

    # Class names mapping (1-indexed in dataset, 0-indexed in code)
    CLASS_NAMES = [
        "rice_leaf_roller", "rice_striped_stem_borer", "rice_gall_midge",
        "rice_stem_fly", "brown_planthopper", "white_backed_planthopper",
        "small_brown_planthopper", "rice_leaf_caterpillar", "rice_skipper",
        "rice_seedling_chaf", "paddy_stem_maggot", "asiatic_rice_borer",
        "yellow_rice_borer", "rice_aphid", "rice_water_weevil",
        "rice_leafminer", "rice_whorl_maggot", "rice_hispa",
        "rice_caseworm", "rice_stem_nematode", "wheat_phloeothrips",
        "wheat_sawfly", "wheat_midge", "wheat_spider",
        "wheat_aphid", "wheat_stem_borer", "corn_borer",
        "corn_aphid", "corn_armyworm", "corn_flea_beetle",
        "corn_seed_maggot", "corn_leaf_blight", "corn_brown_spot",
        "corn_rust", "corn_smut", "corn_head_smu",
        "corn_sheath_blight", "cotton_bollworm", "cotton_aphid",
        "cotton_pink_bollworm", "cotton_spider_mite", "cotton_whitefly",
        "cotton_mirid_bug", "cotton_thrips", "cotton_leaf_roller",
        "cotton_virescence", "soybean_aphid", "soybean_looper",
        "soybean_bean_beetle", "soybean_whitefly", "soybean_pod_borer",
        "soybean_stem_fly", "soybean_nematode", "peanut_aphid",
        "peanut_thrips", "peanut_jassid", "peanut_mite",
        "peanut_white_grub", "peanut_wireworm", "peanut_cutworm",
        "peanut_tobacco_caterpillar", "peanut_sapling_fly",
        "peanut_leaf_spot", "peanut_bacterial_wilt", "peanut_rust",
        "peanut_web blotch", "peanut_pepper_spot", "peanut_sclerotium_blight",
        "beet_flea_beetle", "beet_webworm", "beet_armyworm",
        "beet_sugarbeet_cyst_nematode", "beet_aphid", "beet_leafminer",
        "beet_bacterial_leaf_spot", "beet_cercospora_leaf_spot", "beet_powdery_mildew",
        "beet_rhizoctonia_root_rot", "beet_pythium_root_rot", "beet_fusarium_yellows",
        "rape_flea_beetle", "rape_aphid", "rape_sclerotinia_stem_rot",
        "rape_cabbage_worm", "rape_cabbage_sawfly", "rape_pod_midge",
        "rape_white_rust", "rape_downy_mildew", "rape_virus_disease",
        "rape_black_spot", "rape_soft_rot", "rape_charcoal_rot",
        "rape_club_root", "rape_sclerotinia", "rape_alternaria",
        "rape_gray_mold", "rape_fusarium_wilt", "rape_black_leg",
        "rape_light_leaf_spot", "rape_stem_rot",
    ]

    def __init__(
        self,
        root_dir: str = "data/IP102",
        split: str = "train",
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_path = return_path

        self.images_dir = os.path.join(root_dir, "ip102_v1.1", "images")
        self.list_dir = os.path.join(root_dir, "ip102_v1.1", "list")

        # Load file list
        list_file = os.path.join(self.list_dir, f"{split}.txt")
        self.samples = self._load_list(list_file)

    def _load_list(self, list_file: str) -> List[Tuple[str, int]]:
        """Load image paths and labels from list file.

        IP102 list format: image_name class_id (tab-separated)
        Example: 00101.jpg 1
        """
        samples = []
        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = int(parts[1]) - 1  # Convert to 0-indexed

                    # Construct full path: class_folder / image_name
                    class_folder = f"{class_id + 1:03d}"
                    img_path = os.path.join(self.images_dir, class_folder, img_name)

                    if os.path.exists(img_path):
                        samples.append((img_path, class_id))

        print(f"IP102 {self.split}: loaded {len(samples)} images from {len(set(s[1] for s in samples))} classes")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, img_path
        return image, label

    def get_class_distribution(self) -> torch.Tensor:
        """Get the number of samples per class."""
        labels = [s[1] for s in self.samples]
        counts = torch.zeros(102, dtype=torch.long)
        for label in labels:
            counts[label] += 1
        return counts


def get_ip102_transforms(split: str = "train", input_size: int = 224):
    """Get standard transforms for IP102 dataset.

    Args:
        split: 'train', 'val', or 'test'.
        input_size: Input image size.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
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


def build_ip102_dataloader(
    root_dir: str = "data/IP102",
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    input_size: int = 224,
    use_weighted_sampler: bool = True,
    return_path: bool = False,
) -> DataLoader:
    """Build DataLoader for IP102 dataset.

    Args:
        root_dir: Root directory of IP102.
        split: Data split.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        input_size: Input image size.
        use_weighted_sampler: Use WeightedRandomSampler for class balance (train only).
        return_path: Return image paths (for visualization).

    Returns:
        torch.utils.data.DataLoader.
    """
    from utils.sampler import WeightedSamplerBuilder

    transform = get_ip102_transforms(split, input_size)
    dataset = IP102Dataset(root_dir, split, transform, return_path)

    sampler = None
    shuffle = True if split == "train" else False

    if split == "train" and use_weighted_sampler:
        sampler_builder = WeightedSamplerBuilder()
        sampler = sampler_builder.from_dataset(dataset)
        shuffle = False  # Cannot shuffle with custom sampler

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=(split == "train"),
    )

    return dataloader
