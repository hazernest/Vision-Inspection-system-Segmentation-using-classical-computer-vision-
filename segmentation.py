import cv2
import numpy as np


def qimage_to_gray_array(qimg):
    # convert QImage to grayscale numpy array
    qimg = qimg.convertToFormat(4)  # QImage::Format_ARGB32 (value 4)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((qimg.height(), qimg.width(), 4))
    # ARGB -> convert to BGR order
    bgr = arr[:, :, :3][:, :, ::-1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def segment_cell(gray, method='otsu', adapt_block=51, adapt_C=10, gaussian_blur=3, morph_kernel=3):
    # gray: numpy uint8
    img = gray.copy()
    if gaussian_blur and gaussian_blur > 0:
        k = int(gaussian_blur) if gaussian_blur % 2 == 1 else int(gaussian_blur) + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if method == 'otsu':
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        bs = max(3, adapt_block | 1)
        mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, bs, adapt_C)
    else:
        # default to Otsu
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # morphology: close small holes, open small speckle
    if morph_kernel and morph_kernel > 0:
        k = max(1, int(morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def mask_stats(mask):
    # mask: uint8 0/255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {'area': 0, 'centroid': (0, 0)}
    area = int(len(xs))
    cx = float(xs.mean())
    cy = float(ys.mean())
    return {'area': area, 'centroid': (cx, cy)}
