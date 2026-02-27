from typing import Dict, List, Optional

import torch
import torch.nn as nn

from prism.configs.default import ModelConfig

from .attention_gnn import EdgeAwareAttentionGNN
from .graph_builder import GraphBuilder
from .physics_constraints import PhysicsConstraintModule
from .subgroup_head import SubgroupHead
from .segmentation_branch import SegmentationBranch


class PRISMModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.seg_branch = SegmentationBranch(
            image_size=cfg.image_size,
            num_classes=cfg.num_seg_classes,
            vit_embed_dim=cfg.vit_embed_dim,
            vit_depth=cfg.vit_depth,
            vit_heads=cfg.vit_heads,
            sam_dim=cfg.sam_feat_dim,
            fpn_dim=cfg.fpn_out_dim,
        )
        self.graph_builder = GraphBuilder(
            knn_k=cfg.knn_k,
            morph_text_dim=cfg.morph_text_dim,
            position_dim=cfg.position_dim,
        )
        node_dim = cfg.vit_embed_dim + cfg.position_dim + cfg.morph_text_dim + 2
        self.gnn = EdgeAwareAttentionGNN(
            in_dim=node_dim,
            hidden_dim=cfg.gnn_hidden_channels,
            out_dim=cfg.graph_out_dim,
            edge_dim=3,
        )
        self.cls_head = SubgroupHead(
            in_dim=cfg.graph_out_dim,
            fusion_dim=cfg.fusion_dim,
            num_classes=cfg.num_subgroup_classes,
        )
        self.physics = PhysicsConstraintModule()

    def forward(self, images: torch.Tensor, instance_masks: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        seg_out = self.seg_branch(images)
        seg_logits = seg_out["seg_logits"]
        student_feat = seg_out["student_feat"]
        bsz = images.size(0)

        if instance_masks is None:
            raise ValueError("instance_masks are required for graph construction.")
        if len(instance_masks) != bsz:
            raise ValueError("instance_masks length must match batch size.")

        graph_feats = []
        node_probs, edge_refs = [], []
        for i in range(bsz):
            graph = self.graph_builder.build(student_feat[i : i + 1], instance_masks[i].to(images.device))
            graph_feat = self.gnn(graph.x, graph.edge_index, graph.edge_attr)
            graph_feats.append(graph_feat)

            node_logit = self.cls_head(graph_feat).squeeze(0)
            node_prob = torch.softmax(node_logit, dim=-1).unsqueeze(0).repeat(graph.x.size(0), 1)
            node_probs.append(node_prob)
            edge_refs.append(graph.edge_index)

        graph_feats = torch.cat(graph_feats, dim=0)
        subgroup_logits = self.cls_head(graph_feats)

        return {
            "seg_logits": seg_logits,
            "student_feat": seg_out["student_feat"],
            "teacher_feat": seg_out["teacher_feat"],
            "subgroup_logits": subgroup_logits,
            "node_probs": node_probs,
            "edge_refs": edge_refs,
        }
