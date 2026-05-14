from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download

from app.services.preprocessing import image_bytes_to_tf_string_tensor


class DermFoundationBackbone:
    """
    Thin wrapper around the Google Derm Foundation SavedModel.
    It converts image bytes into the model's serialized tf.Example input
    and returns the 6144-d embedding.
    """

    def __init__(
        self,
        repo_id: str = "google/derm-foundation",
        token: str | None = None,
        local_files_only: bool = False,
        image_size: int = 448,
    ) -> None:
        self.repo_id = repo_id
        self.image_size: Tuple[int, int] = (image_size, image_size)

        model_path = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_files_only=local_files_only,
        )
        self.model_path = Path(model_path)
        self.model = tf.saved_model.load(str(self.model_path))
        self.infer = self.model.signatures["serving_default"]

    def image_to_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Return embedding with shape [1, embedding_dim].
        Derm Foundation normally returns key: "embedding".
        """
        tf_inputs = image_bytes_to_tf_string_tensor(image_bytes, img_size=self.image_size)

        # Your notebook used infer(inputs=tf_inputs). Keep that first.
        try:
            output = self.infer(inputs=tf_inputs)
        except TypeError:
            output = self.infer(tf_inputs)

        if "embedding" not in output:
            available = ", ".join(output.keys())
            raise KeyError(f"Expected output key 'embedding'. Available keys: {available}")

        return output["embedding"].numpy().astype("float32")
