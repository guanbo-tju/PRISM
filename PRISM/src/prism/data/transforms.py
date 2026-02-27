from typing import Callable, Dict

import torchvision.transforms as T


def build_train_transforms(image_size: int) -> Callable:
    image_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
            T.ToTensor(),
        ]
    )
    return image_tf


def build_eval_transforms(image_size: int) -> Callable:
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])


def augmentation_for_consistency() -> Callable:
    return T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)


def collate_batch(batch) -> Dict:
    out = {}
    keys = batch[0].keys()
    for key in keys:
        out[key] = [item[key] for item in batch]
    return out
