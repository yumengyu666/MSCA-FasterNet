"""IP102 Dataset Loader.

IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition
- 102 classes of agricultural insect pests
- ~45,000 images (by-class-name organized version)
- Source: CVPR 2019, https://github.com/xpwu95/IP102

Directory structure (by class name):
    data/IP102/
    ├── Adristyrannus/
    │   ├── img001.jpg
    │   └── ...
    ├── Aleurocanthus spiniferus/
    ├── ...
    ├── classes.txt          (optional: class index mapping)
    └── yellow rice borer/

When official train/val/test split files are not available,
a stratified random split (8:1:1) is used with a fixed seed for reproducibility.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from typing import Optional, Callable, Tuple, List, Dict
import torchvision.transforms as transforms


class IP102Dataset(Dataset):
    """IP102 Insect Pest Recognition Dataset.

    Supports two directory layouts:
    1. By class name: data/IP102/ClassName/images (current)
    2. Official: data/IP102/ip102_v1.1/images/NNN/images (legacy)

    Args:
        root_dir: Root directory of IP102 dataset.
        split: One of 'train', 'val', 'test'.
        transform: Image transformations.
        return_path: Whether to return image path (for visualization).
    """

    # Class names from classes.txt (102 classes, 0-indexed in code)
    CLASS_NAMES = [
        "rice leaf roller", "rice striped stem borer", "rice gall midge",
        "rice stem fly", "brown planthopper", "white backed planthopper",
        "small brown planthopper", "rice leaf caterpillar", "rice skipper",
        "rice seedling chaf", "paddy stem maggot", "asiatic rice borer",
        "yellow rice borer", "rice aphid", "rice water weevil",
        "rice leafminer", "rice whorl maggot", "rice hispa",
        "rice caseworm", "rice stem nematode", "wheat phloeothrips",
        "wheat sawfly", "wheat midge", "wheat spider",
        "wheat aphid", "wheat stem borer", "corn borer",
        "corn aphid", "corn armyworm", "corn flea beetle",
        "corn seed maggot", "corn leaf blight", "corn brown spot",
        "corn rust", "corn smut", "corn head smu",
        "corn sheath blight", "cotton bollworm", "cotton aphid",
        "cotton pink bollworm", "cotton spider mite", "cotton whitefly",
        "cotton mirid bug", "cotton thrips", "cotton leaf roller",
        "cotton virescence", "soybean aphid", "soybean looper",
        "soybean bean beetle", "soybean whitefly", "soybean pod borer",
        "soybean stem fly", "soybean nematode", "peanut aphid",
        "peanut thrips", "peanut jassid", "peanut mite",
        "peanut white grub", "peanut wireworm", "peanut cutworm",
        "peanut tobacco caterpillar", "peanut sapling fly",
        "peanut leaf spot", "peanut bacterial wilt", "peanut rust",
        "peanut web blotch", "peanut pepper spot", "peanut sclerotium blight",
        "beet flea beetle", "beet webworm", "beet armyworm",
        "beet sugarbeet cyst nematode", "beet aphid", "beet leafminer",
        "beet bacterial leaf spot", "beet cercospora leaf spot", "beet powdery mildew",
        "beet rhizoctonia root rot", "beet pythium root rot", "beet fusarium yellows",
        "rape flea beetle", "rape aphid", "rape sclerotinia stem rot",
        "rape cabbage worm", "rape cabbage sawfly", "rape pod midge",
        "rape white rust", "rape downy mildew", "rape virus disease",
        "rape black spot", "rape soft rot", "rape charcoal rot",
        "rape club root", "rape sclerotinia", "rape alternaria",
        "rape gray mold", "rape fusarium wilt", "rape black leg",
        "rape light leaf spot", "rape stem rot",
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
        self.num_classes = 102

        # Try official structure first, then fall back to by-class-name
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and labels.

        Tries:
        1. Official IP102 structure (ip102_v1.1/images/NNN/ + list/train.txt)
        2. By-class-name structure (ClassName/images)
        """
        # Method 1: Official structure
        official_dir = os.path.join(self.root_dir, "ip102_v1.1")
        if os.path.isdir(official_dir):
            list_file = os.path.join(official_dir, "list", f"{self.split}.txt")
            if os.path.isfile(list_file):
                return self._load_official(list_file, official_dir)

        # Method 2: By-class-name structure
        # Collect all images from class folders
        all_samples = self._load_by_class_name()

        # Stratified random split (8:1:1)
        return self._stratified_split(all_samples)

    def _load_official(self, list_file: str, official_dir: str) -> List[Tuple[str, int]]:
        """Load from official IP102 format."""
        samples = []
        images_dir = os.path.join(official_dir, "images")
        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = int(parts[1]) - 1  # Convert to 0-indexed
                    class_folder = f"{class_id + 1:03d}"
                    img_path = os.path.join(images_dir, class_folder, img_name)
                    if os.path.exists(img_path):
                        samples.append((img_path, class_id))
        print(f"IP102 {self.split} (official): {len(samples)} images, "
              f"{len(set(s[1] for s in samples))} classes")
        return samples

    def _load_by_class_name(self) -> List[Tuple[str, int]]:
        """Load all images from class-name-organized folders."""
        # Try to load class mapping from classes.txt
        class_map = self._load_class_map()

        all_samples = []
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        # List class directories
        class_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(valid_exts):
                    img_path = os.path.join(class_dir, fname)
                    # Use class_map if available, otherwise use alphabetical index
                    label = class_map.get(class_name, class_idx)
                    all_samples.append((img_path, label))

        print(f"IP102: found {len(all_samples)} images in {len(class_dirs)} class folders")
        return all_samples

    def _load_class_map(self) -> Dict[str, int]:
        """Load class name -> index mapping from classes.txt."""
        classes_file = os.path.join(self.root_dir, "classes.txt")
        class_map = {}
        if os.path.isfile(classes_file):
            with open(classes_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        idx = int(parts[0]) - 1  # Convert 1-indexed to 0-indexed
                        name = parts[1].strip()
                        class_map[name] = idx
        return class_map

    def _stratified_split(self, all_samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Split samples into train/val/test with stratified random sampling.

        Uses 8:1:1 ratio with a fixed seed (42) for reproducibility.
        """
        rng = np.random.RandomState(42)

        # Group by class
        class_samples: Dict[int, List[Tuple[str, int]]] = {}
        for sample in all_samples:
            label = sample[1]
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(sample)

        split_samples = []
        for label in sorted(class_samples.keys()):
            samples = class_samples[label]
            indices = list(range(len(samples)))
            rng.shuffle(indices)

            n_total = len(samples)
            n_train = max(1, int(n_total * 0.8))
            n_val = max(1, int(n_total * 0.1))

            if self.split == "train":
                sel_indices = indices[:n_train]
            elif self.split == "val":
                sel_indices = indices[n_train:n_train + n_val]
            else:  # test
                sel_indices = indices[n_train + n_val:]

            for idx in sel_indices:
                split_samples.append(samples[idx])

        print(f"IP102 {self.split} (stratified 8:1:1): {len(split_samples)} images, "
              f"{len(set(s[1] for s in split_samples))} classes")
        return split_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, IOError) as e:
            # Return a black image if file is corrupted
            print(f"Warning: Cannot load {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

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
            if 0 <= label < 102:
                counts[label] += 1
        return counts


def get_ip102_transforms(split: str = "train", input_size: int = 224):
    """Get standard transforms for IP102 dataset."""
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
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Simulate occlusion
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
    """Build DataLoader for IP102 dataset."""
    from utils.sampler import WeightedSamplerBuilder

    transform = get_ip102_transforms(split, input_size)
    dataset = IP102Dataset(root_dir, split, transform, return_path)

    sampler = None
    shuffle = True if split == "train" else False

    if split == "train" and use_weighted_sampler:
        sampler_builder = WeightedSamplerBuilder()
        sampler = sampler_builder.from_dataset(dataset)
        shuffle = False

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
