import json
from pathlib import Path

import torch

from app.models.mlp_head import build_mlp_head_from_checkpoint
from app.services.derm_backbone import DermFoundationBackbone


def load_class_names() -> dict[int, str]:
    project_root = Path(__file__).resolve().parents[2]
    class_names_path = project_root / "class_names.json"

    with open(class_names_path, "r", encoding="utf-8") as f:
        raw_class_names = json.load(f)

    return {int(index): name for index, name in raw_class_names.items()}


class TwoStageDermPredictor:
    """
    Stage 1: Derm Foundation image -> embedding.
    Stage 2: PyTorch MLP head embedding -> class probabilities.
    """

    def __init__(
        self,
        derm_model_id: str,
        head_checkpoint_path: str,
        hf_token: str | None = None,
        local_files_only: bool = False,
        image_size: int = 448,
        device_name: str = "auto",
    ) -> None:
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        self.class_names = load_class_names()

        self.backbone = DermFoundationBackbone(
            repo_id=derm_model_id,
            token=hf_token,
            local_files_only=local_files_only,
            image_size=image_size,
        )

        self.head, _ = build_mlp_head_from_checkpoint(
            checkpoint_path=head_checkpoint_path,
            device=self.device,
        )

        output_dim = self.head[-1].out_features

        if output_dim != len(self.class_names):
            raise ValueError(
                f"MLP output dimension is {output_dim}, "
                f"but class_names.json contains {len(self.class_names)} classes."
            )

    def predict(self, image_bytes: bytes) -> dict:
        embedding_np = self.backbone.image_to_embedding(image_bytes)
        embedding = torch.from_numpy(embedding_np).float().to(self.device)

        with torch.no_grad():
            logits = self.head(embedding)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

        print(self.class_names)

        probabilities = [
            {
                "index": i,
                "class_name": self.class_names[i],
                "probability": float(prob),
            }
            for i, prob in enumerate(probs.tolist())
        ]

        return {
            "predicted_index": pred_idx,
            "predicted_class": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": probabilities,
        }