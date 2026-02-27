from dataclasses import dataclass


@dataclass
class ModelConfig:
    image_size: int = 270
    sam_input_size: int = 1024
    num_seg_classes: int = 2
    num_subgroup_classes: int = 2
    vit_embed_dim: int = 768
    vit_depth: int = 8
    vit_heads: int = 8
    sam_feat_dim: int = 256
    fpn_out_dim: int = 256
    gnn_hidden_channels: int = 256
    graph_out_dim: int = 512
    fusion_dim: int = 512
    knn_k: int = 8
    morph_text_dim: int = 128
    position_dim: int = 64


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lambda_kd_t: float = 1.0
    lambda_kd_s: float = 1.0
    lambda_dice: float = 1.0
    lambda_ce: float = 1.0
    lambda_smooth: float = 0.2
    lambda_cons: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
