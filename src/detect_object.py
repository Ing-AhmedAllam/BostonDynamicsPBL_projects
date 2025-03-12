#!/usr/bin/env python

"""
Detect objects in an image using OpenCV

Usage:
    python detect_object.py <image_path>

To use the provided pixi python environment:
    pixi run python detect_object.py <image_path>

"""

import cv2
from cv2 import typing as cvt
from matplotlib import pyplot as plt
from typing import List
import sys

def dbgshow(*imgs: cvt.MatLike, msg="Debug image show"):
    # Show a grid of images
    for idx, img in enumerate(imgs):
        cv2.imshow(f"{msg} Image {idx+1}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect(img: cvt.MatLike) -> List[cvt.MatLike]:
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    img_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 20, 80)
    # dbgshow(edges, msg="edges")
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    closed_contours = []
    closed_contours_img = img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        # print(f"Area: {area}, Perimeter: {perimeter}")

        if area<300: continue

        # if circularity>0.1:
        closed_contours.append(contour)

        tmp_img = img.copy()
        # cv2.drawContours(tmp_img, [contour], -1, (0, 255, 0), 2)
        # dbgshow(tmp_img, msg=f"contour, area={area}, perimeter={perimeter}")

    # cv2.drawContours(closed_contours_img, closed_contours, -1, (0, 0, 255), 2)
    # dbgshow(closed_contours_img, msg="detected object")

    return closed_contours


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../tests/test.jpg"
    img = cv2.imread(img_path)
    results = detect(img)

    result_img = img.copy()
    cv2.drawContours(result_img, results, -1, (0, 0, 255), 2)
    dbgshow(result_img, msg="detected object")
