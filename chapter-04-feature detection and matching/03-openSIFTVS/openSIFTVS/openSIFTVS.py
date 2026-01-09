import argparse
import os
import sys

import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Unable to load image: {}".format(path))
    return img


def stack_images_vertical(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if w1 != w2:
        width = max(w1, w2)
        pad1 = width - w1
        pad2 = width - w2
        if pad1:
            img1 = cv2.copyMakeBorder(img1, 0, 0, 0, pad1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if pad2:
            img2 = cv2.copyMakeBorder(img2, 0, 0, 0, pad2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.vstack([img1, img2])


def main():
    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="OpenCV SIFT matching and visualization")
    parser.add_argument("--img1", default=os.path.join(default_dir, "sse1.bmp"))
    parser.add_argument("--img2", default=os.path.join(default_dir, "sse2.bmp"))
    parser.add_argument("--ratio", type=float, default=0.7, help="Lowe ratio threshold")
    args = parser.parse_args()

    img1 = load_image(args.img1)
    img2 = load_image(args.img2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("No descriptors found.")
        return 1

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(des1, des2, k=2)

    stacked = stack_images_vertical(img1, img2)
    good = 0

    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < args.ratio * n.distance:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
            x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
            y2 += img1.shape[0]
            cv2.line(stacked, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
            good += 1

    print("Found {} total matches".format(good))
    cv2.imshow("Matches", stacked)
    cv2.waitKey(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
