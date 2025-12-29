import os
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit


def _read_kv_file(path: str) -> dict:
    data = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                data[int(parts[0])] = parts[1]
    return data


def load_class_names(root: str) -> List[str]:
    classes_path = os.path.join(root, "classes.txt")
    class_names = []
    with open(classes_path, "r") as f:
        for line in f:
            cid, cname = line.strip().split(maxsplit=1)
            # Remove class ID prefix (e.g., "001.Black_footed_Albatross" -> "Black_footed_Albatross")
            class_name = cname.split(".", 1)[-1] if "." in cname else cname
            class_names.append(class_name)
    return class_names


def stratified_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int]]:

    labels = np.array([sample[1] for sample in dataset.samples])
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    indices = np.arange(len(labels))
    train_idx, val_idx = next(sss.split(indices, labels))
    
    return train_idx.tolist(), val_idx.tolist()


class CUB200Dataset(Dataset):
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[callable] = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Load dataset files
        images_txt = os.path.join(root, "images.txt")
        labels_txt = os.path.join(root, "image_class_labels.txt")
        split_txt = os.path.join(root, "train_test_split.txt")
        
        id_to_path = _read_kv_file(images_txt)
        id_to_label = _read_kv_file(labels_txt)
        id_to_split = _read_kv_file(split_txt)
        
        # Filter samples by split
        is_train_val = 1 if train else 0
        selected_ids = [
            img_id for img_id, split_flag in id_to_split.items()
            if int(split_flag) == is_train_val
        ]
        
        # Build samples list
        self.samples = []
        for img_id in selected_ids:
            rel_path = id_to_path[img_id]
            label = int(id_to_label[img_id]) - 1  # Convert from 1-indexed to 0-indexed
            full_path = os.path.join(root, "images", rel_path)
            self.samples.append((full_path, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            tuple: (image, label) where image is a PIL Image or tensor
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def build_dataloaders(
    root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.0,
    seed: int = 42,
    train_transform: Optional[callable] = None,
    test_transform: Optional[callable] = None,
) -> Tuple[DataLoader, ...]:

    from .transforms import build_transforms
    
    if train_transform is None:
        train_transform = build_transforms(image_size, train=True)
    if test_transform is None:
        test_transform = build_transforms(image_size, train=False)
    
    # Create datasets
    train_dataset = CUB200Dataset(root, train=True, transform=train_transform)
    test_dataset = CUB200Dataset(root, train=False, transform=test_transform)
    
    # Create dataloaders
    if val_ratio > 0:
        # Split training data into train and validation
        train_idx, val_idx = stratified_split(train_dataset, val_ratio, seed)
        
        train_subset = Subset(train_dataset, train_idx)
        val_dataset = CUB200Dataset(root, train=True, transform=test_transform)
        val_subset = Subset(val_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, test_loader
