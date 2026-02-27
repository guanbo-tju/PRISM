import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PRISMDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        self.samples = self._read_csv(csv_path)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def _read_csv(self, csv_path: str) -> List[Dict]:
        samples: List[Dict] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(row)
        if not samples:
            raise ValueError(f"empty dataset: {csv_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_mask(self, path: str) -> torch.Tensor:
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return torch.from_numpy(arr).long()

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image) if self.image_transform else torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0

        mask_tensor = None
        if "mask_path" in sample and sample["mask_path"]:
            mask_tensor = self._read_mask(sample["mask_path"])

        instance_tensor = None
        if "instance_path" in sample and sample["instance_path"]:
            instance_tensor = self._read_mask(sample["instance_path"])

        label = int(sample["label"]) if "label" in sample and sample["label"] != "" else -1

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "instance_mask": instance_tensor,
            "label": label,
            "id": sample.get("id", image_path.stem),
        }
