import io
from typing import Tuple

import tensorflow as tf
from PIL import Image


DERM_FOUNDATION_INPUT_SIZE = (448, 448)


def pil_to_serialized_example(
    img: Image.Image,
    img_size: Tuple[int, int] = DERM_FOUNDATION_INPUT_SIZE,
) -> bytes:
    """
    Convert one PIL image into the serialized tf.train.Example format
    expected by Google Derm Foundation.

    Pipeline:
    RGB -> resize -> PNG bytes -> tf.train.Example with key image/encoded
    """
    img = img.convert("RGB")
    img = img.resize(img_size, resample=Image.BILINEAR)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_bytes])
                )
            }
        )
    )
    return example.SerializeToString()


def image_bytes_to_tf_string_tensor(
    image_bytes: bytes,
    img_size: Tuple[int, int] = DERM_FOUNDATION_INPUT_SIZE,
) -> tf.Tensor:
    """
    Convert uploaded image bytes into a batch of one tf.string input.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        serialized = pil_to_serialized_example(img, img_size=img_size)

    return tf.constant([serialized], dtype=tf.string)
