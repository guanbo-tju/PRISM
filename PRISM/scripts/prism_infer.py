import argparse

import numpy as np
import torch
from PIL import Image

from prism.configs.default import ModelConfig
from prism.data.transforms import build_eval_transforms
from prism.models.prism_model import PRISMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM subgroup inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--instance-mask", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ModelConfig()
    tf = build_eval_transforms(cfg.image_size)

    image = Image.open(args.image).convert("RGB")
    image_t = tf(image).unsqueeze(0)

    instance = np.array(Image.open(args.instance_mask))
    if instance.ndim == 3:
        instance = instance[..., 0]
    instance_t = torch.from_numpy(instance).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRISMModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    with torch.no_grad():
        out = model(image_t.to(device), instance_masks=[instance_t.to(device)])
        prob = torch.softmax(out["subgroup_logits"], dim=-1).squeeze(0)
        pred = prob.argmax().item()

    print(f"pred={pred} prob={prob.cpu().numpy().round(4).tolist()}")


if __name__ == "__main__":
    main()
