"""Utility functions to handle object detection."""
from typing import Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from detector import BoundingBox


def draw_detections(
    image: Image, 
    bbs: List[BoundingBox], 
    category_dict: Optional[Dict[int, str]] = None,
    confidence: Optional[torch.Tensor] = None, 
    channel_first: bool = False
) -> torch.Tensor:
    """Add bounding boxes to image.

    Args:
        image:
            The image without bounding boxes.
        bbs:
            List of bounding boxes to display.
            Each bounding box dict has the format as specified in
            detector.Detector.decode_output.
        category_dict:
            Map from category id to string to label bounding boxes.
            No labels if None.
        channel_first:
            Whether the returned image should have the channel dimension first.

    Returns:
        The image with bounding boxes. Shape (H, W, C) if channel_first is False,
        else (C, H, W).
    """
    fig, ax = plt.subplots(1)
    plt.imshow(image)
    if confidence is not None:
        plt.imshow(
            confidence,
            interpolation="nearest",
            extent=(0, 640, 480, 0),
            alpha=0.5,
        )
    for bb in bbs:
        rect = patches.Rectangle(
            (bb["x"], bb["y"]),
            bb["width"],
            bb["height"],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        if category_dict is not None:
            plt.text(
                bb["x"],
                bb["y"],
                category_dict[bb["category"]]["name"],
            )


    # Save matplotlib figure to numpy array without any borders
    plt.axis("off")
    plt.subplots_adjust(0,0,1,1,0,0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    plt.close(fig)

    if channel_first:
        data = data.transpose((2, 0, 1))  # HWC -> CHW

    return torch.from_numpy(data).float() / 255


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save model to disk.

    Args:
        model: The model to save.
        path: The path to save the model to.
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: str) -> torch.nn.Module:
    """Load model weights from disk.

    Args:
        model: The model to load the weights into.
        path: The path from which to load the model weights.
        device: The device the model weights should be on.

    Returns:
        The loaded model (note that this is the same object as the passed model).
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
