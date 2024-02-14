"""Baseline detector model.

Inspired by
You only look once: Unified, real-time object detection, Redmon, 2016.
"""
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class BoundingBox(TypedDict):
    """Bounding box dictionary.

    Attributes:
        x: Top-left corner column
        y: Top-left corner row
        width: Width of bounding box in pixel
        height: Height of bounding box in pixel
        score: Confidence score of bounding box.
        category: Category (not implemented yet!)
    """

    x: int
    y: int
    width: int
    height: int
    score: float
    category: int


class Detector(nn.Module):
    """Baseline module for object detection."""

    def __init__(self) -> None:
        """Create the module.

        Define all trainable layers.
        """
        super(Detector, self).__init__()

        self.features = models.mobilenet_v2(pretrained=True).features
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images

        self.out_channels = 5
        self.head = nn.Conv2d(in_channels=1280, out_channels=self.out_channels, kernel_size=1)
        # 1x1 Convolution to reduce channels to out_channels without changing H and W

        # 1280x15x20 -> 5x15x20, where each element 5 channel tuple corresponds to
        #   (rel_x_offset, rel_y_offset, rel_x_width, rel_y_height, confidence
        # Where rel_x_offset, rel_y_offset is relative offset from cell_center
        # Where rel_x_width, rel_y_width is relative to image size
        # Where confidence is predicted IOU * probability of object center in this cell
        self.out_cells_x = 20
        self.out_cells_y = 15
        self.img_height = 480.0
        self.img_width = 640.0

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Compute output of neural network from input.

        Args:
            inp: The input images. Shape (N, 3, H, W).

        Returns:
            The output tensor encoding the predicted bounding boxes.
            Shape (N, 5, self.out_cells_y, self.out_cells_y).
        """
        features = self.features(inp)
        out = self.head(features)  # Linear (i.e., no) activation

        return out

    def out_to_bbs(
        self, out: torch.Tensor, threshold: Optional[float] = None, topk: int = 100
    ) -> List[List[BoundingBox]]:
        """Convert output to list of bounding boxes.

        Args:
            out (torch.tensor):
                The output tensor encoding the predicted bounding boxes.
                Shape (N, 5, self.out_cells_y, self.out_cells_y).
                The 5 channels encode in order:
                    - the x offset,
                    - the y offset,
                    - the width,
                    - the height,
                    - the confidence.
            threshold:
                The confidence threshold above which a bounding box will be accepted.
                If None, the topk bounding boxes will be returned.
            topk (int):
                Number of returned bounding boxes if threshold is None.

        Returns:
            List containing N lists of detected bounding boxes in the respective images.
        """
        bbs = []
        out = out.cpu()
        # decode bounding boxes for each image
        for o in out:
            img_bbs = []

            # find cells with bounding box center
            if threshold is not None:
                bb_indices = torch.nonzero(o[4, :, :] >= threshold)
            else:
                _, flattened_indices = torch.topk(o[4, :, :].flatten(), topk)
                bb_indices = np.array(
                    np.unravel_index(flattened_indices.numpy(), o[4, :, :].shape)
                ).T

            # loop over all cells with bounding box center
            for bb_index in bb_indices:
                bb_coeffs = o[0:4, bb_index[0], bb_index[1]]

                # decode bounding box size and position
                width = self.img_width * abs(bb_coeffs[2].item())
                height = self.img_height * abs(bb_coeffs[3].item())
                y = (
                    self.img_height / self.out_cells_y * (bb_index[0] + bb_coeffs[1])
                    - height / 2.0
                ).item()
                x = (
                    self.img_width / self.out_cells_x * (bb_index[1] + bb_coeffs[0])
                    - width / 2.0
                ).item()

                img_bbs.append(
                    {
                        "width": width,
                        "height": height,
                        "x": x,
                        "y": y,
                        "score": o[4, bb_index[0], bb_index[1]].item(),
                    }
                )
            bbs.append(img_bbs)

        return bbs

    def anns_to_target_batch(self, anns: Tuple[dict]) -> Tuple[torch.Tensor]:
        """Convert annotations to batched target tensor.

        Args:
            anns:
                Tuple of bounding box annotations.
                Contains the following keys: "image_id", "boxes", "labels".

                Image ids are not used. Shape (num_bbs,).
                Boxes are in XYXY (top-left, bottom-right) format. Shape (num_bbs, 4).
                Labels are 0-indexed class ids. Shape (num_bbs,).
        Returns:
            Target tensor for the given annotations. 
            Shape (num_anns, self.out_channels, self.out_cells_y, self.out_cells_x).
        """
        # First two channels contain relativ x and y offset of bounding box center
        # Channel 3 & 4 contain relative width and height, respectively
        # Last channel is 1 for cell with bounding box center and 0 without

        # If there is no bb, the first 4 channels will not influence the loss
        # -> can be any number (will be kept at 0)
        target = torch.zeros(len(anns), self.out_channels, self.out_cells_y, self.out_cells_x)

        for i, ann in enumerate(anns):
            for box in ann["boxes"]:
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                x_center_rel = x_center / self.img_width * self.out_cells_x
                y_center_rel = y_center / self.img_height * self.out_cells_y
                x_ind = int(x_center_rel)
                y_ind = int(y_center_rel)
                x_cell_pos = x_center_rel - x_ind
                y_cell_pos = y_center_rel - y_ind
                rel_width = (box[2] - box[0]) / self.img_width
                rel_height = (box[3] - box[1]) / self.img_height

                # channels, rows (y cells), cols (x cells)
                target[i, 4, y_ind, x_ind] = 1

                # bb size
                target[i, 0, y_ind, x_ind] = x_cell_pos
                target[i, 1, y_ind, x_ind] = y_cell_pos
                target[i, 2, y_ind, x_ind] = rel_width
                target[i, 3, y_ind, x_ind] = rel_height

        return target
