# Pathology Representation for Intrinsic Subtyping and Modeling (PRISM)

## Project Structure

```text
.
├── requirements.txt
├── scripts
│   ├── prism_infer.py
│   └── prism_train.py
└── src
    └── prism
        ├── configs
        │   └── default.py
        ├── data
        │   ├── dataset.py
        │   └── transforms.py
        ├── losses
        │   ├── subgroup_losses.py
        │   └── segmentation_losses.py
        ├── models
        │   ├── attention_gnn.py
        │   ├── fpn_decoder.py
        │   ├── prism_model.py
        │   ├── graph_builder.py
        │   ├── morphology_text_encoder.py
        │   ├── physics_constraints.py
        │   ├── subgroup_head.py
        │   ├── sam_teacher.py
        │   ├── segmentation_branch.py
        │   └── vit_encoder.py
        ├── training
        │   └── engine.py
        └── utils
            └── seed.py
```

## Technical Pipeline

1. `SegmentationBranch`: ViT encoding + SAM teacher feature distillation + FPN decoding. The nuclear segmentation loss uses `KD + Dice + CE`.
2. `GraphBuilder`: Constructs the graph based on nuclear instance masks, integrating local features, positional encodings, and morphological text embeddings.
3. `EdgeAwareAttentionGNN`: Edge-aware attention graph learning designed to obtain graph-level representations.
4. `PhysicsConstraintModule`: Incorporates spatial consistency and observational consistency constraints during training, introducing no additional overhead during inference.
5. `PRISMModel`: Executes the joint forward pass, outputting both segmentation and subgroup classification results.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/prism_train.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --output-dir ./outputs
```

## Data Description

The CSV file must contain at least the following fields:

- `image_path`: Path to the pathology image.
- `mask_path`: Path to the nuclear semantic segmentation label (pixel-level classes).
- `instance_path`: Path to the nuclear instance mask (a unique instance ID for each nucleus, where 0 represents the background).
- `label`: Classification label.

## Inference

```bash
export PYTHONPATH=src
python scripts/prism_infer.py \
  --checkpoint ./outputs/best.pt \
  --image /path/to/example.png \
  --instance-mask /path/to/example_instance.png
```
