#!/usr/bin/env python

"""
Detect objects in an image and estimate their poses

Usage:
    python detect_object.py <image_path>

To use the provided pixi python environment:
    pixi run python detect_object.py <image_path>

"""

import cv2
from cv2 import typing as cvt
from typing import List, Tuple
from pandas.io.parsers.base_parser import Enum
from ultralytics import YOLO
from ultralytics.engine.results import Results
import sys
import torch
import numpy as np
from numpy import typing as npt
from math import atan2, cos, sin, sqrt, pi

def dbgshow(*imgs: cvt.MatLike, msg="Debug image show"):
    for idx, img in enumerate(imgs):
        cv2.imshow(f"{msg} Image {idx+1}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

yolo = YOLO("yolo11l-seg.pt").to("mps")

def detect_objects(image: cvt.MatLike) -> Tuple[List[npt.NDArray], cvt.MatLike]:
    """
    Detect objects in an image

    Args:
        image: The image to detect objects in

    Returns:
        A list of tensors of masks for each detected object, in shape (obj_idx (list len), point_idx, x, y)
    """
    results: List[Results] = yolo(image)

    # List[(obj_idx, point_idx, x, y)]
    all_masks: List[npt.NDArray] = sum([res.masks.xy for res in results], [])

    input_img = image.copy()
    for res in results:
        input_img = res.plot(img = input_img)

    dbgshow(input_img)
    return all_masks, input_img

def detect_poses(objects: List[npt.NDArray], image: cvt.MatLike):
    """
    Detects poses in the image given object masks.

    Args:
        objects: A list of tensors of masks for each detected object, in shape (obj_idx (list len), point_idx, x, y)
        image: The image to detect objects in
    """
    for idx, mask in enumerate(objects):
        object_points = mask

        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(object_points, mean)

        cntr = (int(mean[0,0]), int(mean[0,1]))
        cv2.circle(image, cntr, 3, (255,0,255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        drawAxis(image, cntr, p1, (0, 255, 0), 1)
        drawAxis(image, cntr, p2, (255, 255, 0), 1)
        # import pdb; pdb.set_trace()
        dbgshow(image, msg=f"detected pose {idx}")

# utils, from https://docs.opencv.org/4.x/d1/dee/tutorial_introduction_to_pca.html
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_object.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    masks, annotated = detect_objects(image)
    poses = detect_poses(masks, annotated.copy())
