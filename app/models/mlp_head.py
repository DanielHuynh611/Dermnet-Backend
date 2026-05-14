import torch
import torch.nn as nn


DROPOUT = 0.6


class DermFoundationMLPHead(nn.Sequential):
    """
    Exact MLP head used after Derm Foundation embeddings.

    Architecture:
        Linear(input_dim, 512) -> ReLU -> Dropout(0.6)
        Linear(512, 256)      -> ReLU -> Dropout(0.6)
        Linear(256, 128)      -> ReLU -> Dropout(0.6)
        Linear(128, num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, num_classes),
        )


def build_mlp_head_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """
    Load derm_foundation_mlp_head.pt.

    Expected checkpoint format:
        {
            "model_state_dict": model.state_dict(),
            ...
        }
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = checkpoint["model_state_dict"]

    input_dim = int(state_dict["0.weight"].shape[1])
    num_classes = int(state_dict["9.weight"].shape[0])

    head = DermFoundationMLPHead(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(device)

    head.load_state_dict(state_dict, strict=True)
    head.eval()

    return head, checkpoint