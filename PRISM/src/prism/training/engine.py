from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from prism.configs.default import TrainConfig
from prism.losses.subgroup_losses import focal_loss
from prism.losses.segmentation_losses import (
    dice_loss,
    kd_spatial_loss,
    kd_temporal_loss,
)


def _stack_images(images: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack(images, dim=0).to(device)


def _stack_mask_like(masks: List[torch.Tensor], out_hw: int, device: torch.device) -> torch.Tensor:
    x = torch.stack([m.float() for m in masks], dim=0).unsqueeze(1).to(device)
    x = F.interpolate(x, size=(out_hw, out_hw), mode="nearest")
    return x.squeeze(1).long()


def run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    cfg: TrainConfig,
    device: torch.device,
    train: bool,
    consistency_aug: Callable,
) -> Dict[str, float]:
    model.train(mode=train)
    losses = {"total": 0.0, "seg": 0.0, "focal": 0.0, "phys": 0.0}
    n = 0

    for batch in tqdm(loader, disable=False):
        images = _stack_images(batch["image"], device)
        masks = _stack_mask_like(batch["mask"], out_hw=model.cfg.image_size, device=device)
        labels = torch.tensor(batch["label"], dtype=torch.long, device=device)
        instance_masks = [x.to(device) for x in batch["instance_mask"]]

        out = model(images, instance_masks=instance_masks)
        seg_logits = out["seg_logits"]
        subgroup_logits = out["subgroup_logits"]

        loss_kd_t = kd_temporal_loss(out["teacher_feat"], out["student_feat"])
        loss_kd_s = kd_spatial_loss(out["teacher_feat"], out["student_feat"])
        loss_dice = dice_loss(seg_logits, masks)
        loss_ce = F.cross_entropy(seg_logits, masks)
        loss_seg = cfg.lambda_kd_t * loss_kd_t + cfg.lambda_kd_s * loss_kd_s + cfg.lambda_dice * loss_dice + cfg.lambda_ce * loss_ce

        loss_cls = focal_loss(subgroup_logits, labels, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

        with torch.no_grad():
            images_aug = consistency_aug(images)
        out_aug = model(images_aug, instance_masks=instance_masks)
        p = torch.softmax(subgroup_logits, dim=-1)
        p_aug = torch.softmax(out_aug["subgroup_logits"], dim=-1)
        obs_cons = model.physics.observation_consistency(p, p_aug)

        smooth_terms = []
        for node_prob, edge_index in zip(out["node_probs"], out["edge_refs"]):
            smooth_terms.append(model.physics.spatial_consistency(node_prob, edge_index))
        smooth_loss = torch.stack(smooth_terms).mean() if smooth_terms else torch.zeros((), device=device)
        loss_phys = cfg.lambda_smooth * smooth_loss + cfg.lambda_cons * obs_cons

        total = loss_seg + loss_cls + loss_phys

        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

        bs = images.size(0)
        losses["total"] += total.item() * bs
        losses["seg"] += loss_seg.item() * bs
        losses["focal"] += loss_cls.item() * bs
        losses["phys"] += loss_phys.item() * bs
        n += bs

    return {k: v / max(n, 1) for k, v in losses.items()}


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    cfg: TrainConfig,
    device: torch.device,
    consistency_aug: Callable,
    output_dir: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, cfg, device, True, consistency_aug)
        va = run_epoch(model, val_loader, optimizer, cfg, device, False, consistency_aug)

        if va["total"] < best:
            best = va["total"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")

        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={tr['total']:.4f} val_total={va['total']:.4f} "
            f"train(seg={tr['seg']:.4f}, focal={tr['focal']:.4f}, phys={tr['phys']:.4f})"
        )
