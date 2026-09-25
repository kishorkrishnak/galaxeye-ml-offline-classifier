"""Small RGB feature extractor used by training and inference."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

FEATURE_VERSION = "rgb-grid-v1"
IMAGE_SIZE = 32


def extract_features(image_bytes: bytes) -> np.ndarray:
    """Return one deterministic, normalized feature vector for an RGB tile.

    This deliberately compact baseline combines a low-resolution RGB image,
    per-channel color histograms, and coarse spatial color averages.  It keeps
    the demo CPU-only and makes the preprocessing easy to inspect.
    """
    with Image.open(BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    pixel_features = pixels[::2, ::2, :].reshape(-1)
    histograms = [
        np.histogram(pixels[:, :, channel], bins=16, range=(0.0, 1.0))[0]
        for channel in range(3)
    ]
    histogram_features = np.concatenate(histograms).astype(np.float32)
    histogram_features /= float(IMAGE_SIZE * IMAGE_SIZE)

    cell_means = []
    for row in range(4):
        for column in range(4):
            cell = pixels[row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8]
            cell_means.extend(cell.mean(axis=(0, 1)))

    return np.concatenate(
        [pixel_features, histogram_features, np.asarray(cell_means, dtype=np.float32)]
    ).astype(np.float32)
