"""Utility functions to handle object detection."""
from typing import Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from detector import BoundingBox


def add_bounding_boxes(
    ax: plt.Axes, bbs: List[BoundingBox], category_dict: Dict[int, str] = None
) -> None:
    """Add bounding boxes to specified axes.

    Args:
        ax:
            The axis to add the bounding boxes to.
        bbs:
            List of bounding boxes to display.
            Each bounding box dict has the format as specified in
            detector.Detector.decode_output.
        category_dict:
            Map from category id to string to label bounding boxes.
            No labels if None.
    """
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
