from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    # Hugging Face / Derm Foundation
    derm_model_id: str = os.getenv("DERM_MODEL_ID", "google/derm-foundation")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    local_files_only: bool = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"

    # Model artifacts
    head_checkpoint_path: Path = Path(os.getenv("HEAD_CHECKPOINT_PATH", "derm_foundation_mlp_head.pt"))
    class_names_path: Path = Path(os.getenv("CLASS_NAMES_PATH", "class_names.json"))

    # Inference
    image_size: int = int(os.getenv("DERM_IMAGE_SIZE", "448"))
    device: str = os.getenv("TORCH_DEVICE", "auto")

    # API
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")


settings = Settings()
