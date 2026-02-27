import torch
import torch.nn.functional as F


def kd_temporal_loss(teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    t = torch.softmax(teacher_feat.flatten(2), dim=-1)
    s = torch.log_softmax(student_feat.flatten(2), dim=-1)
    return F.kl_div(s, t, reduction="batchmean")


def kd_spatial_loss(teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_feat, teacher_feat)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.size(1)
    prob = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (prob * target_one_hot).sum(dim=(2, 3))
    den = (prob.pow(2) + target_one_hot.pow(2)).sum(dim=(2, 3))
    score = (2 * inter + eps) / (den + eps)
    return 1 - score.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target) + dice_loss(logits, target)
