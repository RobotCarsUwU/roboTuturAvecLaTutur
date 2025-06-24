##
## EPITECH PROJECT, 2025
## Post process
## File description:
## Post process
##

import cv2
import numpy as np
from skimage.morphology import skeletonize


class PostProcessor:

    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def clean_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        max_allowed_area = (mask.shape[0] * mask.shape[1]) * 0.3
        min_allowed_aspect_ratio = 0.05
        max_allowed_aspect_ratio = 20

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = width / height if height != 0 else 0

            if (
                area < 100
                or area > max_allowed_area
                or not (
                    min_allowed_aspect_ratio <= aspect_ratio <= max_allowed_aspect_ratio
                )
            ):
                mask[labels == i] = 0

        mask = skeletonize(mask > 0).astype(np.uint8) * 255
        return mask

    def process(self, mask):
        """Simple post-processing pipeline"""
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        return self.clean_mask(mask)
