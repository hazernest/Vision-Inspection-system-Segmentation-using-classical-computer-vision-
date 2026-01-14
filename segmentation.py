import cv2
import numpy as np

try:
    from PyQt6.QtGui import QImage
except Exception:
    QImage = None


def qimage_to_gray_array(qimg):
    # convert QImage to grayscale numpy array
    # PyQt6 requires QImage.Format enum (PyQt5 historically allowed int values)
    if QImage is None:
        raise RuntimeError("PyQt6 is required for qimage_to_gray_array() in improved_UI")
    qimg = qimg.convertToFormat(QImage.Format.Format_ARGB32)
    ptr = qimg.bits()
    byte_count = getattr(qimg, "sizeInBytes", None)
    byte_count = byte_count() if callable(byte_count) else qimg.byteCount()
    ptr.setsize(int(byte_count))
    arr = np.frombuffer(ptr, np.uint8).reshape((qimg.height(), qimg.width(), 4))
    # ARGB -> convert to BGR order
    bgr = arr[:, :, :3][:, :, ::-1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside a binary mask.

    A "hole" is any 0-valued region fully enclosed by 255-valued foreground.
    Regions connected to the image border are considered background and are not filled.

    Args:
        mask: uint8 mask where foreground is >0 (typically 255) and background is 0.

    Returns:
        A uint8 mask (0/255) with internal holes filled.
    """
    if mask is None:
        return mask
    if mask.ndim != 2:
        raise ValueError('fill_internal_holes expects a 2D mask')

    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    if h == 0 or w == 0:
        return m

    inv = cv2.bitwise_not(m)
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Fill the exterior/background in the inverted image.
    # IMPORTANT: do NOT assume (0,0) is background because unit crops can be fully inside the mold surface.
    # Instead, flood-fill from every border pixel that is background in the original mask (255 in inv).
    def _try_seed(x: int, y: int):
        if flood[y, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, y), 0)

    # Top and bottom rows
    for x in range(w):
        _try_seed(x, 0)
        _try_seed(x, h - 1)
    # Left and right columns
    for y in range(h):
        _try_seed(0, y)
        _try_seed(w - 1, y)

    # Remaining 255s in `flood` correspond to holes.
    holes = flood
    filled = cv2.bitwise_or(m, holes)
    return filled


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

    # IMPORTANT: ensure the segmented surface is a solid region.
    # Bright/white foreign material can create internal holes that would then be excluded from defect detection.
    mask = fill_internal_holes(mask)
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
