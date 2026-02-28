# 🧬 PRISM  
## Pathology Representation for Intrinsic Subtyping and Modeling  

> A Vision–Language–Graph Framework for Nuclear-aware Intrinsic Subtyping in Computational Pathology  

---

## ✨ Highlights

- 🔹 Joint nuclear segmentation and intrinsic subtype prediction  
- 🔹 SAM-based knowledge distillation for pathology adaptation  
- 🔹 Morphology-aware nucleus-level graph construction  
- 🔹 Edge-aware attention graph neural network  
- 🔹 Physics-inspired structural regularization (zero inference overhead)  

---

## 🔬 Overview

PRISM is a unified computational pathology framework for intrinsic molecular subtype modeling from histopathology images.

Instead of slide-level feature pooling, PRISM explicitly models nuclei as biological primitives and performs structured relational reasoning.

Key characteristics:

- Instance-aware segmentation backbone  
- Morphology-text aligned feature embedding  
- Graph-based structural representation learning  
- Physics-constrained optimization  
- End-to-end joint training  

---

## 🏗️ Architecture

```
Input Image
     │
     ▼
Vision Transformer Encoder
     │
     ▼
SAM Teacher Feature Distillation
     │
     ▼
Segmentation Branch (FPN Decoder)
     │
     ▼
Nuclear Instance Masks
     │
     ▼
GraphBuilder
(Appearance + Position + Morphology Text)
     │
     ▼
EdgeAwareAttentionGNN
     │
     ▼
PhysicsConstraintModule (training only)
     │
     ▼
Subgroup Classification Head
```

Outputs:
- Pixel-level nuclear segmentation
- Graph-level intrinsic subtype prediction

---

## 🧠 Core Components

### 1️⃣ SegmentationBranch

- ViT backbone
- SAM feature distillation
- FPN decoder

Segmentation loss:

\[
\mathcal{L}_{seg}
=
\lambda_{KD} L_{KD}
+
\lambda_{Dice} L_{Dice}
+
\lambda_{CE} L_{CE}
\]

Ensures biologically meaningful nuclear delineation.

---

### 2️⃣ GraphBuilder

Constructs a nucleus-level graph from instance masks.

Node features:
- Local appearance embedding
- Relative positional encoding
- Morphology-aware text embedding

Edge attributes:
- Spatial proximity
- Morphological similarity

This produces a structured biological interaction graph.

---

### 3️⃣ EdgeAwareAttentionGNN

- Edge-conditioned attention mechanism
- Adaptive relational weight learning
- Graph-level subtype embedding extraction

Captures higher-order nuclear interaction patterns.

---

### 4️⃣ PhysicsConstraintModule

Training-time structural regularization:

- Spatial consistency constraint
- Observational consistency constraint

No additional parameters or inference-time overhead.

---

### 5️⃣ PRISMModel

Unified forward pipeline:

```
Image → Segmentation → Graph Construction → GNN → Classification
```

Outputs:
- segmentation_logits
- subgroup_logits

---

## 📁 Project Structure

```text
PRISM/
├── requirements.txt
├── scripts
│   ├── prism_train.py
│   └── prism_infer.py
└── src/prism
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

---

## 🚀 Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

---

## 🏋️ Training

```bash
python scripts/prism_train.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --output-dir ./outputs
```

---

## 🧾 Dataset Format

The CSV file must contain at least:

| Field | Description |
|-------|------------|
| image_path | Path to pathology image |
| mask_path | Pixel-level semantic segmentation mask |
| instance_path | Nuclear instance mask (unique ID per nucleus, 0 = background) |
| label | Intrinsic subtype label |

---

## 🔎 Inference

```bash
export PYTHONPATH=src
python scripts/prism_infer.py \
  --checkpoint ./outputs/best.pt \
  --image /path/to/example.png \
  --instance-mask /path/to/example_instance.png
```

---

## 📊 Optimization Objective

Total training loss:

\[
\mathcal{L}
=
\mathcal{L}_{seg}
+
\lambda_{sub} \mathcal{L}_{subgroup}
+
\lambda_{phys} \mathcal{L}_{physics}
\]

Enables:

- Pixel-level supervision
- Graph-level supervision
- Structural regularization

---

## 🔬 Design Principles

PRISM is built upon three principles:

1. Nucleus as the atomic biological unit  
2. Subtype as structured morphological pattern  
3. Learning as constrained relational modeling  

---

## 🧩 Extensibility

The framework supports:

- Alternative backbone encoders
- Different graph neural network variants
- Additional structural constraints
- Extension to survival prediction or multi-task learning

---

## 📜 Citation

```bibtex
@article{prism2026,
  title   = {PRISM: Pathology Representation for Intrinsic Subtyping and Modeling},
  author  = {Your Name},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 📌 License

Specify your preferred open-source license (e.g., MIT, Apache-2.0).
