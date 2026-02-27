from typing import Dict

import torch
import torch.nn as nn

from .fpn_decoder import FPNDecoder
from .sam_teacher import SAMTeacherProxy
from .vit_encoder import LightweightViTEncoder


class SegmentationBranch(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        vit_embed_dim: int,
        vit_depth: int,
        vit_heads: int,
        sam_dim: int,
        fpn_dim: int,
    ) -> None:
        super().__init__()
        self.vit = LightweightViTEncoder(
            image_size=image_size,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
        )
        self.sam_teacher = SAMTeacherProxy(out_dim=sam_dim)
        self.decoder = FPNDecoder(
            in_student=vit_embed_dim,
            in_teacher=sam_dim,
            out_dim=fpn_dim,
            num_classes=num_classes,
        )
        self.image_size = image_size

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        student_feat = self.vit(images)
        teacher_feat = self.sam_teacher(images, target_hw=student_feat.shape[-2:])
        logits = self.decoder(student_feat, teacher_feat, out_size=self.image_size)
        return {
            "seg_logits": logits,
            "student_feat": student_feat,
            "teacher_feat": teacher_feat,
        }
