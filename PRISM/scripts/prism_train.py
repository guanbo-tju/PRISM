import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from prism.configs.default import ModelConfig, TrainConfig
from prism.data.dataset import PRISMDataset
from prism.data.transforms import (
    augmentation_for_consistency,
    build_eval_transforms,
    build_train_transforms,
    collate_batch,
)
from prism.models.prism_model import PRISMModel
from prism.training.engine import train_model
from prism.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM multi-omics pathology subgroup training")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )
    set_seed(train_cfg.seed)

    train_ds = PRISMDataset(
        csv_path=args.train_csv,
        image_transform=build_train_transforms(model_cfg.image_size),
    )
    val_ds = PRISMDataset(
        csv_path=args.val_csv,
        image_transform=build_eval_transforms(model_cfg.image_size),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRISMModel(model_cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    consistency_aug = augmentation_for_consistency()

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg=train_cfg,
        device=device,
        consistency_aug=consistency_aug,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
