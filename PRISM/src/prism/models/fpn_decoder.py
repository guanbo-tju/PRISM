import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    def __init__(self, in_student: int, in_teacher: int, out_dim: int, num_classes: int) -> None:
        super().__init__()
        self.student_lateral = nn.Conv2d(in_student, out_dim, kernel_size=1)
        self.teacher_lateral = nn.Conv2d(in_teacher, out_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(out_dim, num_classes, kernel_size=1)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, out_size: int) -> torch.Tensor:
        s = self.student_lateral(student_feat)
        t = self.teacher_lateral(teacher_feat)
        x = self.fuse(s + t)
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return self.head(x)
