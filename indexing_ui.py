import sys
import json
import os
import base64
import csv
from PyQt6 import QtCore, QtGui, QtWidgets

# Ensure local imports resolve when running from repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import segmentation
import cv2
import numpy as np

try:
    from qfluentwidgets import (
        FluentWindow,
        PushButton,
        PrimaryPushButton,
        TransparentPushButton,
        ToggleButton,
        SwitchButton,
        SpinBox,
        ComboBox,
        Pivot,
    )
except Exception:
    FluentWindow = QtWidgets.QMainWindow
    PushButton = QtWidgets.QPushButton
    PrimaryPushButton = QtWidgets.QPushButton
    TransparentPushButton = QtWidgets.QPushButton
    ToggleButton = QtWidgets.QPushButton
    SwitchButton = QtWidgets.QPushButton
    SpinBox = QtWidgets.QSpinBox
    ComboBox = QtWidgets.QComboBox
    Pivot = None

# Item data roles in PyQt6 are scoped; keep existing arithmetic (UserRole + N)
ROLE_BASE = int(QtCore.Qt.ItemDataRole.UserRole)


class ImageWidget(QtWidgets.QWidget):
    selectionChanged = QtCore.pyqtSignal()
    cellClicked = QtCore.pyqtSignal(int)
    exclusionDrawn = QtCore.pyqtSignal(object)
    exclusionEditUpdated = QtCore.pyqtSignal(object)
    exclusionEditCommitted = QtCore.pyqtSignal(object)
    imageFilesDropped = QtCore.pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.image = None  # QImage
        # positions are stored in image coordinates (not display coordinates)
        self.start_img_pos = None
        self.current_img_rect = None
        self.fixed_img_rect = None
        self.grid_rects = []
        self.setMinimumSize(400, 100)
        self.scale = 1.0
        self.offset = QtCore.QPoint(0, 0)
        # signal emitted when user finishes drawing a selection
        # manual zoom multiplier (1.0 = fit-to-window)
        self.manual_zoom = 1.0
        # drawing enabled (allow click-drag to set base unit)
        self.drawing_enabled = True
        # exclusion drawing mode flag
        self.exclusion_mode = False
        self.current_exclusion_rect = None
        # when drawing an exclusion we temporarily enable drawing; remember and restore the prior state
        self._drawing_enabled_before_exclusion = None

        # exclusion edit mode (resize existing exclusion)
        self.exclusion_edit_mode = False
        self.exclusion_edit_shape = None  # 'rect'|'circle'|None
        # in image coordinates
        self.exclusion_edit_rect = None   # QRect for rect edits
        self.exclusion_edit_circle = None # (cx:int, cy:int, r:int) for circle edits
        self._excl_dragging_handle = False
        self._excl_drag_anchor = None
        # selected cell index and mask pixmap (in image coordinates)
        self.selected_cell_index = None
        self.selected_mask_pixmap = None
        # optional QPainterPath outlining eroded mask (in image coordinates)
        self.erosion_path = None
        # per-cell overlays to draw on the main canvas: {grid_idx: {'seg': QPixmap|None, 'defect': QPixmap|None}}
        self.cell_overlays = {}
        # current overlay mode for full-canvas drawing
        self.overlay_mode = 'Defect'

        # inspection mode: show only X/O verdicts per unit (no overlays)
        self.inspection_mode = False
        # {grid_idx: bool} where True means defect (X), False means OK (O)
        self.inspection_results = {}

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        try:
            md = event.mimeData()
            if md is None or not md.hasUrls():
                event.ignore()
                return
            paths = []
            for u in md.urls():
                try:
                    if u.isLocalFile():
                        paths.append(str(u.toLocalFile()))
                except Exception:
                    continue
            ok = any(p.lower().endswith(('.tif', '.tiff')) for p in paths)
            if ok:
                event.acceptProposedAction()
            else:
                event.ignore()
        except Exception:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        # Same policy as dragEnterEvent
        self.dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        try:
            md = event.mimeData()
            if md is None or not md.hasUrls():
                event.ignore()
                return
            paths = []
            for u in md.urls():
                try:
                    if u.isLocalFile():
                        p = str(u.toLocalFile())
                        if p.lower().endswith(('.tif', '.tiff')):
                            paths.append(p)
                except Exception:
                    continue
            if not paths:
                event.ignore()
                return
            self.imageFilesDropped.emit(paths)
            event.acceptProposedAction()
        except Exception:
            event.ignore()

    def load_image(self, path):
        img = QtGui.QImage(path)
        if img.isNull():
            # Fallback for TIFF variants / Qt plugin limitations.
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError('Failed to load image: ' + path)

            if arr.dtype != np.uint8:
                if arr.dtype == np.uint16:
                    arr = (arr / 256).astype(np.uint8)
                else:
                    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            if arr.ndim == 2:
                h, w = arr.shape
                img = QtGui.QImage(
                    arr.data,
                    w,
                    h,
                    w,
                    QtGui.QImage.Format.Format_Grayscale8,
                ).copy()
            else:
                h, w = arr.shape[:2]
                if arr.shape[2] == 4:
                    rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
                    img = QtGui.QImage(
                        rgba.data,
                        w,
                        h,
                        w * 4,
                        QtGui.QImage.Format.Format_RGBA8888,
                    ).copy()
                else:
                    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    img = QtGui.QImage(
                        rgb.data,
                        w,
                        h,
                        w * 3,
                        QtGui.QImage.Format.Format_RGB888,
                    ).copy()

        if img.isNull():
            raise RuntimeError('Failed to load image: ' + path)

        self.image = img
        self.updateScale()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateScale()

    def updateScale(self):
        if not self.image:
            return
        iw = self.image.width()
        ih = self.image.height()
        if iw == 0 or ih == 0:
            return
        # compute fit-to-viewport base scale if we have a scroll viewport parent
        base = 1.0
        parent = self.parent()
        if parent is not None:
            vw = parent.width()
            vh = parent.height()
            if vw > 0 and vh > 0:
                base = min(vw / iw, vh / ih)
        self.scale = base * self.manual_zoom
        dw = int(iw * self.scale)
        dh = int(ih * self.scale)
        # set widget to the display size so scrollbars reflect zoom/pan correctly
        self.setFixedSize(dw, dh)
        # offset within widget should be zero (drawing at 0,0)
        self.offset = QtCore.QPoint(0, 0)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.GlobalColor.black)
        if self.image:
            # draw scaled image at widget origin
            disp = self.image.scaled(
                int(self.image.width() * self.scale),
                int(self.image.height() * self.scale),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawImage(0, 0, disp)
        pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
        painter.setPen(pen)
        # draw current selection (convert image coords -> display coords)
        if self.current_img_rect:
            r = self.imgrect_to_display(self.current_img_rect)
            painter.drawRect(r)
        if self.fixed_img_rect:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 2)
            painter.setPen(pen)
            r = self.imgrect_to_display(self.fixed_img_rect)
            painter.drawRect(r)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 1)
        painter.setPen(pen)
        for r, idx in self.grid_rects:
            # r is image-space rect tuple
            img_r = QtCore.QRect(r[0], r[1], r[2], r[3])
            dr = self.imgrect_to_display(img_r)
            painter.drawRect(dr)
            painter.drawText(dr.topLeft() + QtCore.QPoint(3, 12), str(idx))

        # inspection view: draw only verdict markers and skip overlays
        if getattr(self, 'inspection_mode', False):
            painter.save()
            painter.setOpacity(1.0)
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            for r, idx in self.grid_rects:
                img_r = QtCore.QRect(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                dr = self.imgrect_to_display(img_r)
                verdict = None
                try:
                    verdict = self.inspection_results.get(idx)
                except Exception:
                    verdict = None
                if verdict is None:
                    continue
                # size text relative to cell size
                try:
                    s = max(10.0, min(dr.width(), dr.height()) * 0.45)
                except Exception:
                    s = 18.0
                f = painter.font()
                f.setPointSizeF(float(s))
                painter.setFont(f)
                if verdict:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 3))
                    painter.drawText(dr, QtCore.Qt.AlignmentFlag.AlignCenter, 'X')
                else:
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 3))
                    painter.drawText(dr, QtCore.Qt.AlignmentFlag.AlignCenter, 'O')
            painter.restore()
            return

        # draw overlays for ALL units on the main canvas
        mode = getattr(self, 'overlay_mode', 'Defect')
        if mode != 'None' and getattr(self, 'cell_overlays', None):
            painter.setOpacity(0.55)
            for r, idx in self.grid_rects:
                ov = self.cell_overlays.get(idx)
                if not ov:
                    continue
                img_r = QtCore.QRect(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                dr = self.imgrect_to_display(img_r)
                if mode in ('Segmentation', 'Both'):
                    seg_pm = ov.get('seg')
                    if isinstance(seg_pm, QtGui.QPixmap):
                        painter.drawPixmap(
                            dr.topLeft(),
                            seg_pm.scaled(
                                dr.size(),
                                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                QtCore.Qt.TransformationMode.SmoothTransformation,
                            ),
                        )
                if mode in ('Defect', 'Both'):
                    defect_pm = ov.get('defect')
                    if isinstance(defect_pm, QtGui.QPixmap):
                        painter.drawPixmap(
                            dr.topLeft(),
                            defect_pm.scaled(
                                dr.size(),
                                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                QtCore.Qt.TransformationMode.SmoothTransformation,
                            ),
                        )
            painter.setOpacity(1.0)
        # draw selected mask overlay if available
        if self.selected_cell_index is not None and self.selected_mask_pixmap:
            # find rect for selected cell
            for r, idx in self.grid_rects:
                if idx == self.selected_cell_index:
                    img_r = QtCore.QRect(r[0], r[1], r[2], r[3])
                    dr = self.imgrect_to_display(img_r)
                    # mask pixmap is in image coords with same size as img_r
                    mask_scaled = self.selected_mask_pixmap.scaled(
                        dr.size(),
                        QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    painter.setOpacity(0.6)
                    painter.drawPixmap(dr.topLeft(), mask_scaled)
                    painter.setOpacity(1.0)
                    break

        # draw erosion outline if present (in image coordinates, scaled to display)
        if self.erosion_path is not None:
            painter.save()
            pen = QtGui.QPen(QtGui.QColor(0, 255, 255), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setWorldTransform(QtGui.QTransform().scale(self.scale, self.scale), False)
            painter.drawPath(self.erosion_path)
            painter.restore()

        # draw exclusion edit overlay (single shape) + resize handle/arrow
        if getattr(self, 'exclusion_edit_mode', False):
            painter.save()
            pen = QtGui.QPen(QtGui.QColor(255, 0, 255), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

            handle_center = None
            if self.exclusion_edit_shape == 'rect' and isinstance(self.exclusion_edit_rect, QtCore.QRect):
                dr = self.imgrect_to_display(self.exclusion_edit_rect)
                painter.drawRect(dr)
                handle_center = dr.bottomRight()
            elif self.exclusion_edit_shape == 'circle' and self.exclusion_edit_circle is not None:
                try:
                    cx, cy, r = self.exclusion_edit_circle
                    rect = QtCore.QRect(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
                    dr = self.imgrect_to_display(rect)
                    painter.drawEllipse(dr)
                    handle_center = QtCore.QPoint(dr.right(), dr.center().y())
                except Exception:
                    handle_center = None

            if handle_center is not None:
                # small arrow-like handle: a filled triangle + short line
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 255)))
                size = 10
                p1 = QtCore.QPoint(handle_center.x(), handle_center.y())
                p2 = QtCore.QPoint(handle_center.x() - size, handle_center.y())
                p3 = QtCore.QPoint(handle_center.x(), handle_center.y() - size)
                painter.drawPolygon(QtGui.QPolygon([p1, p2, p3]))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawLine(handle_center, QtCore.QPoint(handle_center.x() - size * 2, handle_center.y() - size * 2))

            painter.restore()
    def mousePressEvent(self, event):
        if not self.image:
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.pos()

            # exclusion edit: grab the resize handle
            if getattr(self, 'exclusion_edit_mode', False):
                handle_pos = None
                if self.exclusion_edit_shape == 'rect' and isinstance(self.exclusion_edit_rect, QtCore.QRect):
                    dr = self.imgrect_to_display(self.exclusion_edit_rect)
                    handle_pos = dr.bottomRight()
                elif self.exclusion_edit_shape == 'circle' and self.exclusion_edit_circle is not None:
                    try:
                        cx, cy, r = self.exclusion_edit_circle
                        rect = QtCore.QRect(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
                        dr = self.imgrect_to_display(rect)
                        handle_pos = QtCore.QPoint(dr.right(), dr.center().y())
                    except Exception:
                        handle_pos = None

                if handle_pos is not None:
                    dx = pos.x() - handle_pos.x()
                    dy = pos.y() - handle_pos.y()
                    if (dx * dx + dy * dy) <= (14 * 14):
                        self._excl_dragging_handle = True
                        # anchor data depends on shape
                        if self.exclusion_edit_shape == 'rect' and isinstance(self.exclusion_edit_rect, QtCore.QRect):
                            r = self.exclusion_edit_rect
                            self._excl_drag_anchor = (
                                int(r.x()),
                                int(r.y()),
                                int(r.width()),
                                int(r.height()),
                            )
                        elif self.exclusion_edit_shape == 'circle' and self.exclusion_edit_circle is not None:
                            cx, cy, rr = self.exclusion_edit_circle
                            self._excl_drag_anchor = (int(cx), int(cy), int(rr))
                        return

            img_pt = self.display_to_img(pos)
            # if drawing is disabled, treat click as selection only
            if not self.drawing_enabled:
                for r, idx in self.grid_rects:
                    x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                    if x <= img_pt.x() < x + w and y <= img_pt.y() < y + h:
                        self.cellClicked.emit(idx)
                        return
            # if not currently drawing, treat as click to select cell
            # but if we're in exclusion_mode, allow starting a drag instead
            if self.start_img_pos is None and not self.current_img_rect and not self.exclusion_mode:
                # find cell under point
                for r, idx in self.grid_rects:
                    x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                    if x <= img_pt.x() < x + w and y <= img_pt.y() < y + h:
                        self.cellClicked.emit(idx)
                        return
            # if drawing disabled, do not start a drag
            if not self.drawing_enabled:
                return
            self.start_img_pos = img_pt
            self.current_img_rect = QtCore.QRect(img_pt, QtCore.QSize())
            self.update()

    def mouseMoveEvent(self, event):
        if getattr(self, '_excl_dragging_handle', False) and getattr(self, 'exclusion_edit_mode', False):
            img_pt = self.display_to_img(event.pos())
            if self.exclusion_edit_shape == 'rect' and isinstance(self.exclusion_edit_rect, QtCore.QRect):
                try:
                    ax, ay, aw, ah = self._excl_drag_anchor
                except Exception:
                    ax, ay, aw, ah = int(self.exclusion_edit_rect.x()), int(self.exclusion_edit_rect.y()), int(self.exclusion_edit_rect.width()), int(self.exclusion_edit_rect.height())
                # resize from bottom-right while keeping top-left fixed
                new_w = max(1, int(img_pt.x() - ax))
                new_h = max(1, int(img_pt.y() - ay))
                self.exclusion_edit_rect = QtCore.QRect(int(ax), int(ay), int(new_w), int(new_h))
                self.update()
                self.exclusionEditUpdated.emit({'shape': 'rect', 'w': int(new_w), 'h': int(new_h)})
                return
            if self.exclusion_edit_shape == 'circle' and self.exclusion_edit_circle is not None:
                try:
                    cx, cy, rr = self._excl_drag_anchor
                except Exception:
                    cx, cy, rr = self.exclusion_edit_circle
                # handle is on the right side of the circle; radius follows x distance
                new_r = max(1, int(abs(img_pt.x() - cx)))
                self.exclusion_edit_circle = (int(cx), int(cy), int(new_r))
                self.update()
                self.exclusionEditUpdated.emit({'shape': 'circle', 'r': int(new_r)})
                return

        if self.start_img_pos is not None:
            img_pt = self.display_to_img(event.pos())
            self.current_img_rect = QtCore.QRect(self.start_img_pos, img_pt).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and getattr(self, '_excl_dragging_handle', False):
            self._excl_dragging_handle = False
            self._excl_drag_anchor = None
            # commit current geometry
            if getattr(self, 'exclusion_edit_mode', False):
                if self.exclusion_edit_shape == 'rect' and isinstance(self.exclusion_edit_rect, QtCore.QRect):
                    self.exclusionEditCommitted.emit({'shape': 'rect', 'w': int(self.exclusion_edit_rect.width()), 'h': int(self.exclusion_edit_rect.height())})
                elif self.exclusion_edit_shape == 'circle' and self.exclusion_edit_circle is not None:
                    try:
                        _, _, r = self.exclusion_edit_circle
                        self.exclusionEditCommitted.emit({'shape': 'circle', 'r': int(r)})
                    except Exception:
                        pass
            return

        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.current_img_rect:
            if self.exclusion_mode:
                # emit exclusion rect in image coordinates
                excl = self.current_img_rect.normalized()
                self.current_exclusion_rect = excl
                # stop exclusion mode
                self.exclusion_mode = False
                # restore drawing_enabled immediately to avoid any gap where a drag can redefine the base unit
                try:
                    prev = self._drawing_enabled_before_exclusion
                    if isinstance(prev, bool):
                        self.drawing_enabled = prev
                except Exception:
                    pass
                self.start_img_pos = None
                self.current_img_rect = None
                self.update()
                self.exclusionDrawn.emit(excl)
                return
            # normal base unit selection
            self.fixed_img_rect = self.current_img_rect
            self.start_img_pos = None
            self.current_img_rect = None
            self.grid_rects = []
            self.update()
            # notify listeners that a new base unit has been set
            self.selectionChanged.emit()

    def clear_grid(self):
        self.grid_rects = []
        self.update()

    def set_exclusion_edit(self, shape, rect=None, circle=None):
        self.exclusion_edit_mode = True
        self.exclusion_edit_shape = shape
        self.exclusion_edit_rect = rect
        self.exclusion_edit_circle = circle
        self.update()

    def clear_exclusion_edit(self):
        self.exclusion_edit_mode = False
        self.exclusion_edit_shape = None
        self.exclusion_edit_rect = None
        self.exclusion_edit_circle = None
        self._excl_dragging_handle = False
        self._excl_drag_anchor = None
        self.update()

    def imgrect_to_display(self, QRect_img):
        # QRect_img is QRect in image coordinates
        x = int(QRect_img.x() * self.scale)
        y = int(QRect_img.y() * self.scale)
        w = int(QRect_img.width() * self.scale)
        h = int(QRect_img.height() * self.scale)
        return QtCore.QRect(x, y, w, h)

    def display_to_img(self, QPoint_disp):
        # convert a display QPoint to image QPoint (clamped)
        px = QPoint_disp.x()
        py = QPoint_disp.y()
        ix = int(px / self.scale)
        iy = int(py / self.scale)
        ix = max(0, min(self.image.width() - 1, ix))
        iy = max(0, min(self.image.height() - 1, iy))
        return QtCore.QPoint(ix, iy)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Indexing UI - Mold Segmentation')

        # Multi-image support (used by the Defect -> Foreign material panel)
        # Each image keeps its own cached results so switching is instant.
        self._images = []  # list[str]
        # Path list aligned 1:1 with the image combo box indices.
        # We maintain this ourselves because some Fluent ComboBox variants don't reliably preserve userData/currentData.
        self._image_combo_paths = []  # list[str]
        self._image_states = {}  # path -> {'seg': list[QPixmap|None], 'def': list[QPixmap|None], 'inspection': dict[int,bool]}
        self._current_image_path = None
        self._current_image_size = None  # (w,h)
        # Reference/original image: indexing + exclusions are defined here.
        # Segmentation is recomputed per image (because there can be small shifts between captures).
        self._reference_image_path = None
        self._reference_image_size = None  # (w,h)
        # Segmentation-anchored exclusion alignment (XY shift only): {grid_idx: (cx, cy)} in unit-local coords
        self._exclusion_ref_centroids = {}
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        # left: image in scroll area
        self.img_widget = ImageWidget()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.img_widget)
        # put scroll inside a container so we can overlay zoom buttons on its viewport
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.scroll)
        hbox.addWidget(container, 1)

        # create zoom buttons as children of the scroll viewport so they overlay image
        self.zoom_in_btn = QtWidgets.QPushButton('+', parent=self.scroll.viewport())
        self.zoom_out_btn = QtWidgets.QPushButton('-', parent=self.scroll.viewport())
        self.ensure_fit_btn = QtWidgets.QPushButton('Fit', parent=self.scroll.viewport())
        self.ensure_fit_btn.setFixedSize(64, 28)
        self.ensure_fit_btn.clicked.connect(self.ensure_fit_view)
        self.zoom_in_btn.setFixedSize(36, 36)
        self.zoom_out_btn.setFixedSize(36, 36)
        self.zoom_in_btn.clicked.connect(lambda: self.img_widget_zoom(1.25))
        self.zoom_out_btn.clicked.connect(lambda: self.img_widget_zoom(1/1.25))
        # position will be updated via eventFilter
        self.scroll.viewport().installEventFilter(self)

        # right: controls (Fluent widgets + Pivot navigation)
        ctrl = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(ctrl)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        def _lbl(text: str, tip: str = None):
            w = QtWidgets.QLabel(text)
            if tip:
                w.setToolTip(tip)
            return w

        tips = {
            'Units:': 'How many units (cells) exist across the image.',
            'Blocks:': 'If units are grouped into blocks, set how many blocks you have.',
            'Unit spacing X (px):': 'Gap between neighboring units (left/right).',
            'Unit spacing Y (px):': 'Gap between neighboring units (up/down).',
            'Block spacing X (px):': 'Extra gap between blocks (left/right).',
            'Block spacing Y (px):': 'Extra gap between blocks (up/down).',
            'Exclusion #': 'Pick which exclusion slot you want to draw.',
            'Add exclusion': 'Draw an area to ignore across all units (e.g., a known mark).',
            'Segmentation Method:': 'How the app separates the mold surface from the background.',
            'Gaussian blur kernel:': 'Smoothing amount. Higher = smoother, but can hide small details.',
            'Morph kernel size:': 'Cleans up the mask. Higher = stronger cleanup.',
            'Adaptive block size:': 'Window size used for adaptive method. Must be an odd number.',
            'Adaptive C:': 'How strict the adaptive method is. Higher can remove more.',
            'Threshold:': 'Sensitivity for defect detection. Lower finds more; higher finds fewer.',
            'Min area (px):': 'Ignore tiny specks smaller than this size.',
            'Mask erosion (px):': 'Shrink the area inward so edges are ignored.',
            'Overlay mode:': 'What is drawn on the image (segmentation/defects/both).'
        }
        # Make this page compute a meaningful minimum height (enables scrolling)
        try:
            v.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        except Exception:
            pass

        load_btn = PrimaryPushButton('Load Image')
        load_btn.clicked.connect(self.load_image)
        v.addWidget(load_btn)

        # Validation note
        note = _lbl(
            'Expect image 4096x3000 (or it will still work).',
            'This was designed for the mold camera image size, but other sizes can still work.'
        )
        v.addWidget(note)

        self.units_x = SpinBox(); self.units_x.setRange(0, 100); self.units_x.setValue(0)
        self.units_y = SpinBox(); self.units_y.setRange(0, 100); self.units_y.setValue(0)
        self.blocks_x = SpinBox(); self.blocks_x.setRange(0, 50); self.blocks_x.setValue(0)
        self.blocks_y = SpinBox(); self.blocks_y.setRange(0, 50); self.blocks_y.setValue(0)

        form = QtWidgets.QFormLayout()
        units_row = QtWidgets.QHBoxLayout()
        units_row.addWidget(QtWidgets.QLabel('X'))
        units_row.addWidget(self.units_x)
        units_row.addSpacing(6)
        units_row.addWidget(QtWidgets.QLabel('Y'))
        units_row.addWidget(self.units_y)
        self.units_x.setToolTip('Units along X (left to right).')
        self.units_y.setToolTip('Units along Y (top to bottom).')
        form.addRow(_lbl('Units:', tips.get('Units:')), units_row)

        blocks_row = QtWidgets.QHBoxLayout()
        blocks_row.addWidget(QtWidgets.QLabel('X'))
        blocks_row.addWidget(self.blocks_x)
        blocks_row.addSpacing(6)
        blocks_row.addWidget(QtWidgets.QLabel('Y'))
        blocks_row.addWidget(self.blocks_y)
        self.blocks_x.setToolTip('Blocks along X (left to right).')
        self.blocks_y.setToolTip('Blocks along Y (top to bottom).')
        form.addRow(_lbl('Blocks:', tips.get('Blocks:')), blocks_row)
        v.addLayout(form)

        # spacings: sliders + spinboxes for X/Y for units and blocks
        self.unit_space_x = SpinBox(); self.unit_space_x.setRange(0, 1000); self.unit_space_x.setValue(0)
        self.unit_space_y = SpinBox(); self.unit_space_y.setRange(0, 1000); self.unit_space_y.setValue(0)
        self.block_space_x = SpinBox(); self.block_space_x.setRange(0, 2000); self.block_space_x.setValue(0)
        self.block_space_y = SpinBox(); self.block_space_y.setRange(0, 2000); self.block_space_y.setValue(0)

        self.unit_space_x_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.unit_space_x_slider.setRange(0, 1000); self.unit_space_x_slider.setValue(0)
        self.unit_space_y_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.unit_space_y_slider.setRange(0, 1000); self.unit_space_y_slider.setValue(0)
        self.block_space_x_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.block_space_x_slider.setRange(0, 2000); self.block_space_x_slider.setValue(0)
        self.block_space_y_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.block_space_y_slider.setRange(0, 2000); self.block_space_y_slider.setValue(0)

        def _add_slider_row(title: str, slider: QtWidgets.QSlider, spin: QtWidgets.QAbstractSpinBox):
            row = QtWidgets.QHBoxLayout()
            lbl = _lbl(title, tips.get(title))
            lbl.setMinimumWidth(130)
            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            if tips.get(title):
                slider.setToolTip(tips.get(title))
                spin.setToolTip(tips.get(title))
            v.addLayout(row)

        _add_slider_row('Unit spacing X (px):', self.unit_space_x_slider, self.unit_space_x)
        _add_slider_row('Unit spacing Y (px):', self.unit_space_y_slider, self.unit_space_y)
        _add_slider_row('Block spacing X (px):', self.block_space_x_slider, self.block_space_x)
        _add_slider_row('Block spacing Y (px):', self.block_space_y_slider, self.block_space_y)

        # wire sliders and spinboxes together
        self.unit_space_x_slider.valueChanged.connect(self.unit_space_x.setValue)
        self.unit_space_y_slider.valueChanged.connect(self.unit_space_y.setValue)
        self.block_space_x_slider.valueChanged.connect(self.block_space_x.setValue)
        self.block_space_y_slider.valueChanged.connect(self.block_space_y.setValue)
        self.unit_space_x.valueChanged.connect(self.unit_space_x_slider.setValue)
        self.unit_space_y.valueChanged.connect(self.unit_space_y_slider.setValue)
        self.block_space_x.valueChanged.connect(self.block_space_x_slider.setValue)
        self.block_space_y.valueChanged.connect(self.block_space_y_slider.setValue)

        self.apply_btn = PrimaryPushButton('Apply Indexing')
        self.apply_btn.clicked.connect(self.apply_indexing)
        v.addWidget(self.apply_btn)
        # add unlock/edit button to re-enable drawing
        self.edit_btn = ToggleButton('Unlock Editing')
        try:
            self.edit_btn.setCheckable(True)
        except Exception:
            pass
        self.edit_btn.toggled.connect(self.toggle_editing)
        v.addWidget(self.edit_btn)

        export_btn = PushButton('Export grid JSON')
        export_btn.clicked.connect(self.export_grid)
        import_btn = PushButton('Import grid JSON')
        import_btn.clicked.connect(self.import_grid)

        io_row = QtWidgets.QHBoxLayout(); io_row.addWidget(export_btn); io_row.addWidget(import_btn)
        v.addLayout(io_row)

        # Internal thumbnails list (not shown in the UI).
        # This is kept as an internal data store for per-unit pixmaps/masks.
        self.thumb_list = QtWidgets.QListWidget()
        self.thumb_list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.thumb_list.setIconSize(QtCore.QSize(128, 128))
        self.thumb_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.thumb_list.setMovement(QtWidgets.QListView.Movement.Static)
        self.thumb_list.hide()

        # Exclusions: add/exclusion index and shape
        excl_box = QtWidgets.QHBoxLayout()
        self.excl_index = SpinBox(); self.excl_index.setRange(0, 0); self.excl_index.setValue(0)
        self.excl_shape = ComboBox(); self.excl_shape.addItems(['rectangle', 'circle'])
        self.add_excl_btn = PushButton('Add exclusion')
        self.add_excl_btn.clicked.connect(self.add_exclusion)

        # Modify dropdown (Edit/Delete)
        self.modify_excl_btn = PushButton('Modify')
        self.modify_excl_btn.clicked.connect(self.open_modify_exclusion_dialog)

        excl_box.addWidget(_lbl('Exclusion #', tips.get('Exclusion #')))
        excl_box.addWidget(self.excl_index)
        excl_box.addWidget(self.excl_shape)
        excl_box.addWidget(self.add_excl_btn)
        excl_box.addWidget(self.modify_excl_btn)
        v.addLayout(excl_box)


        self.excl_index.setToolTip(tips.get('Exclusion #'))
        self.excl_shape.setToolTip('Choose the exclusion shape to draw (rectangle or circle).')
        self.add_excl_btn.setToolTip(tips.get('Add exclusion'))
        self.modify_excl_btn.setToolTip('Modify the selected exclusion (edit or delete).')

        self.exclusions = []
        self._exclusion_edit_active = False
        self._modify_dialog = None
        self._exclusion_edit_timer = QtCore.QTimer(self)
        self._exclusion_edit_timer.setSingleShot(True)
        self._exclusion_edit_timer.setInterval(250)
        self._exclusion_edit_timer.timeout.connect(self.run_segmentation_all)

        self.excl_index.valueChanged.connect(self.on_exclusion_index_changed)
        self.excl_shape.currentIndexChanged.connect(lambda _: self.on_exclusion_index_changed())

        # initialize disabled state until an exclusion exists
        try:
            self.on_exclusion_index_changed()
        except Exception:
            pass

        # segmentation controls
        v.addWidget(_lbl('Segmentation Method:', tips.get('Segmentation Method:')))
        self.seg_method = ComboBox()
        self.seg_method.addItems(['otsu', 'adaptive'])
        self.seg_method.setToolTip('Otsu = automatic threshold. Adaptive = handles uneven lighting better.')
        v.addWidget(self.seg_method)
        self.gauss_spin = SpinBox(); self.gauss_spin.setRange(0, 31); self.gauss_spin.setValue(3)
        self.morph_spin = SpinBox(); self.morph_spin.setRange(0, 31); self.morph_spin.setValue(3)
        self.adapt_block = SpinBox(); self.adapt_block.setRange(3, 201); self.adapt_block.setValue(51)
        self.adapt_C = SpinBox(); self.adapt_C.setRange(-50, 50); self.adapt_C.setValue(10)
        form2 = QtWidgets.QFormLayout()
        self.gauss_spin.setToolTip(tips.get('Gaussian blur kernel:'))
        self.morph_spin.setToolTip(tips.get('Morph kernel size:'))
        self.adapt_block.setToolTip(tips.get('Adaptive block size:'))
        self.adapt_C.setToolTip(tips.get('Adaptive C:'))
        form2.addRow(_lbl('Gaussian blur kernel:', tips.get('Gaussian blur kernel:')), self.gauss_spin)
        form2.addRow(_lbl('Morph kernel size:', tips.get('Morph kernel size:')), self.morph_spin)
        form2.addRow(_lbl('Adaptive block size:', tips.get('Adaptive block size:')), self.adapt_block)
        form2.addRow(_lbl('Adaptive C:', tips.get('Adaptive C:')), self.adapt_C)
        v.addLayout(form2)
        run_seg_btn = PrimaryPushButton('Run Segmentation')
        run_seg_btn.clicked.connect(self.run_segmentation_all)
        v.addWidget(run_seg_btn)
        export_masks_btn = PushButton('Export Masks + CSV')
        export_masks_btn.clicked.connect(self.export_masks_and_csv)
        v.addWidget(export_masks_btn)

        v.addStretch(1)

        # Defect page (will be shown via Pivot navigation)
        defect_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(defect_tab)
        dv.setContentsMargins(8, 8, 8, 8)
        dv.setSpacing(6)
        # Make this page compute a meaningful minimum height (enables scrolling)
        try:
            dv.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        except Exception:
            pass

        # Fluent sub-navigation for defect types
        defect_pivot = Pivot() if Pivot is not None else None
        defect_stack = QtWidgets.QStackedWidget()
        if defect_pivot is not None:
            dv.addWidget(defect_pivot)
        dv.addWidget(defect_stack)

        # Example defect subtab: Particle detection
        particle_tab = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(particle_tab)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)
        try:
            pv.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        except Exception:
            pass
        pv.addWidget(QtWidgets.QLabel('Foreign material detection settings'))

        # Multi-image import + selector (requested)
        img_row = QtWidgets.QHBoxLayout()
        img_row.addWidget(_lbl('Image:', 'Select which loaded image is shown.'))
        self.image_combo = ComboBox()
        try:
            self.image_combo.setMinimumWidth(220)
        except Exception:
            pass
        self.image_combo.setToolTip('Switch between loaded images.')
        img_row.addWidget(self.image_combo, 1)
        self.add_images_btn = PushButton('Add images')
        self.add_images_btn.setToolTip('Load multiple images so you can switch between them.')
        self.add_images_btn.clicked.connect(self.add_images)
        img_row.addWidget(self.add_images_btn)
        pv.addLayout(img_row)
        self.defect_method = ComboBox()
        self.defect_method.addItems(['threshold', 'canny'])
        self.defect_threshold = SpinBox(); self.defect_threshold.setRange(0, 255); self.defect_threshold.setValue(24)
        self.defect_min_area = SpinBox(); self.defect_min_area.setRange(0, 100000); self.defect_min_area.setValue(20)
        # mask erosion: shrink segmentation mask by this many pixels before detection
        self.defect_mask_erode = SpinBox(); self.defect_mask_erode.setRange(0, 200); self.defect_mask_erode.setValue(6)
        # overlay display mode
        self.overlay_mode = ComboBox()
        self.overlay_mode.addItems(['None', 'Segmentation', 'Defect', 'Both'])
        self.overlay_mode.setCurrentIndex(2)

        defect_form = QtWidgets.QFormLayout()
        self.defect_method.setToolTip('Threshold = simple + fast. Canny = edge-based (more sensitive).')
        self.defect_threshold.setToolTip(tips.get('Threshold:'))
        self.defect_min_area.setToolTip(tips.get('Min area (px):'))
        self.defect_mask_erode.setToolTip(tips.get('Mask erosion (px):'))
        self.overlay_mode.setToolTip(tips.get('Overlay mode:'))
        defect_form.addRow(_lbl('Method:', 'How the app finds foreign material.'), self.defect_method)
        defect_form.addRow(_lbl('Threshold:', tips.get('Threshold:')), self.defect_threshold)
        defect_form.addRow(_lbl('Min area (px):', tips.get('Min area (px):')), self.defect_min_area)
        defect_form.addRow(_lbl('Mask erosion (px):', tips.get('Mask erosion (px):')), self.defect_mask_erode)
        defect_form.addRow(_lbl('Overlay mode:', tips.get('Overlay mode:')), self.overlay_mode)
        pv.addLayout(defect_form)
        self.overlay_mode.currentIndexChanged.connect(self.on_overlay_mode_changed)

        # Live update (debounced) for defect parameters
        self._defect_autoupdate_timer = QtCore.QTimer(self)
        self._defect_autoupdate_timer.setSingleShot(True)
        self._defect_autoupdate_timer.timeout.connect(self._auto_update_defect_selected_unit)
        self.defect_threshold.valueChanged.connect(self.schedule_defect_autoupdate)
        self.defect_min_area.valueChanged.connect(self.schedule_defect_autoupdate)
        # recompute erosion outline when mask-erode value changes
        if hasattr(self, 'defect_mask_erode'):
            self.defect_mask_erode.valueChanged.connect(lambda _: self.update_erosion_outline(self.img_widget.selected_cell_index))
            self.defect_mask_erode.valueChanged.connect(self.schedule_defect_autoupdate)
        # unit index selector for testing
        self.defect_unit_spin = SpinBox(); self.defect_unit_spin.setRange(0, 0); self.defect_unit_spin.setValue(0)
        defect_form2 = QtWidgets.QFormLayout()
        defect_form2.addRow('Unit index to test:', self.defect_unit_spin)
        pv.addLayout(defect_form2)

        # test buttons (same actions, more compact layout)
        test_btn = PrimaryPushButton('Test on unit')
        test_btn.clicked.connect(self.test_defect_detection)
        test_all_btn = PrimaryPushButton('Test All Units')
        test_all_btn.clicked.connect(self.test_defect_detection_all)
        test_row = QtWidgets.QHBoxLayout(); test_row.addWidget(test_btn); test_row.addWidget(test_all_btn)
        pv.addLayout(test_row)

        # Inspection toggle (Fluent switch)
        try:
            self.run_insp_btn = SwitchButton('Run Inspection')
        except Exception:
            self.run_insp_btn = ToggleButton('Run Inspection')
        try:
            self.run_insp_btn.setChecked(False)
        except Exception:
            pass
        if hasattr(self.run_insp_btn, 'toggled'):
            self.run_insp_btn.toggled.connect(self.on_inspection_toggled)
        elif hasattr(self.run_insp_btn, 'checkedChanged'):
            self.run_insp_btn.checkedChanged.connect(self.on_inspection_toggled)
        pv.addWidget(self.run_insp_btn)
        pv.addStretch(1)
        defect_stack.addWidget(particle_tab)
        if defect_pivot is not None:
            defect_pivot.addItem('foreign', 'Foreign material', onClick=lambda: defect_stack.setCurrentWidget(particle_tab))

        # Placeholder for additional defect types
        crack_tab = QtWidgets.QWidget()
        cvlay = QtWidgets.QVBoxLayout(crack_tab)
        try:
            cvlay.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        except Exception:
            pass
        cvlay.addWidget(QtWidgets.QLabel('Crack detection (placeholder)'))
        cvlay.addStretch(1)
        defect_stack.addWidget(crack_tab)
        if defect_pivot is not None:
            defect_pivot.addItem('crack', 'Crack', onClick=lambda: defect_stack.setCurrentWidget(crack_tab))
            defect_pivot.setCurrentItem('foreign')
        else:
            defect_stack.setCurrentWidget(particle_tab)

        # create a right-side container with fluent Pivot navigation + SCROLLABLE pages + a log terminal underneath
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.right_pivot = Pivot() if Pivot is not None else None
        self.right_stack = QtWidgets.QStackedWidget()
        self.right_stack.addWidget(ctrl)
        self.right_stack.addWidget(defect_tab)
        if self.right_pivot is not None:
            right_layout.addWidget(self.right_pivot)
            self.right_pivot.addItem('main', 'Main', onClick=lambda: self.right_stack.setCurrentWidget(ctrl))
            self.right_pivot.addItem('defect', 'Defect', onClick=lambda: self.right_stack.setCurrentWidget(defect_tab))
            self.right_pivot.setCurrentItem('main')

        # Scroll area so the settings panel can be navigated on smaller displays (e.g., 1920x1080)
        self.right_scroll = QtWidgets.QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_scroll.setWidget(self.right_stack)
        right_layout.addWidget(self.right_scroll, 1)
        # log terminal (plain text, read-only)
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1000)
        self.log_output.setMinimumHeight(160)
        right_layout.addWidget(QtWidgets.QLabel('Log'))
        right_layout.addWidget(self.log_output)
        hbox.addWidget(right_container)

        self.resize(1200, 600)
        # live update connections: update preview whenever parameters change
        self.units_x.valueChanged.connect(self.update_grid_preview)
        self.units_y.valueChanged.connect(self.update_grid_preview)
        self.blocks_x.valueChanged.connect(self.update_grid_preview)
        self.blocks_y.valueChanged.connect(self.update_grid_preview)
        self.unit_space_x.valueChanged.connect(self.update_grid_preview)
        self.unit_space_y.valueChanged.connect(self.update_grid_preview)
        self.block_space_x.valueChanged.connect(self.update_grid_preview)
        self.block_space_y.valueChanged.connect(self.update_grid_preview)
        # sliders should also trigger preview
        self.unit_space_x_slider.valueChanged.connect(self.update_grid_preview)
        self.unit_space_y_slider.valueChanged.connect(self.update_grid_preview)
        self.block_space_x_slider.valueChanged.connect(self.update_grid_preview)
        self.block_space_y_slider.valueChanged.connect(self.update_grid_preview)
        self.img_widget.selectionChanged.connect(self.update_grid_preview)
        self.img_widget.cellClicked.connect(self.on_cell_clicked)
        # Thumbnail preview is hidden in improved_UI, so selection changes come from
        # canvas clicks / unit-index controls rather than the list widget.
        self.img_widget.exclusionDrawn.connect(self.on_exclusion_drawn)
        self.img_widget.exclusionEditUpdated.connect(self.on_exclusion_edit_updated)
        self.img_widget.exclusionEditCommitted.connect(self.on_exclusion_edit_committed)
        self.img_widget.imageFilesDropped.connect(self.on_image_files_dropped)

        # switching images from combo
        try:
            self.image_combo.currentIndexChanged.connect(self.on_image_combo_changed)
        except Exception:
            pass

    def _snapshot_current_results(self):
        """Capture current per-unit masks + inspection results for the active image."""
        if not self._current_image_path:
            return
        seg_masks = []
        def_masks = []
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            seg_masks.append(item.data(ROLE_BASE + 1) if item is not None else None)
            def_masks.append(item.data(ROLE_BASE + 2) if item is not None else None)
        self._image_states[self._current_image_path] = {
            'seg': seg_masks,
            'def': def_masks,
            'inspection': dict(getattr(self.img_widget, 'inspection_results', {}) or {}),
        }

    def _get_reference_seg_masks(self):
        """Return the reference (original image) segmentation masks list, or None if not available."""
        if not self._reference_image_path:
            return None
        st = self._image_states.get(self._reference_image_path) or {}
        seg_masks = st.get('seg')
        if not seg_masks:
            return None
        for pm in seg_masks:
            if isinstance(pm, QtGui.QPixmap):
                return seg_masks
        return None

    def _apply_reference_seg_masks_to_current(self) -> bool:
        """Apply reference segmentation masks (template) to the currently loaded image units."""
        seg_masks = self._get_reference_seg_masks()
        if not seg_masks:
            return False
        n = self.thumb_list.count()
        for i in range(n):
            item = self.thumb_list.item(i)
            if item is None:
                continue
            if i < len(seg_masks) and isinstance(seg_masks[i], QtGui.QPixmap):
                item.setData(ROLE_BASE + 1, seg_masks[i])
            else:
                item.setData(ROLE_BASE + 1, None)
        return True

    def _restore_results_for_path(self, path: str):
        """Restore cached masks + inspection results for `path` (if present)."""
        st = self._image_states.get(path)
        if not st:
            return
        seg_masks = st.get('seg') or []
        def_masks = st.get('def') or []
        n = self.thumb_list.count()
        for i in range(n):
            item = self.thumb_list.item(i)
            if item is None:
                continue
            if i < len(seg_masks) and isinstance(seg_masks[i], QtGui.QPixmap):
                item.setData(ROLE_BASE + 1, seg_masks[i])
            else:
                item.setData(ROLE_BASE + 1, None)
            if i < len(def_masks) and isinstance(def_masks[i], QtGui.QPixmap):
                item.setData(ROLE_BASE + 2, def_masks[i])
            else:
                item.setData(ROLE_BASE + 2, None)
        # Do not automatically enable inspection mode here; switching logic decides.
        try:
            self.img_widget.inspection_results = dict(st.get('inspection') or {})
        except Exception:
            pass

    def _ensure_image_registered(self, path: str, switch_to: bool = False):
        if not path:
            return
        # add to combo if missing
        if path not in self._images:
            self._images.append(path)
            self._image_combo_paths.append(path)
            name = os.path.basename(path)
            # Always add without userData; we rely on _image_combo_paths for correctness.
            self.image_combo.addItem(name)
        if switch_to:
            try:
                idx = self._image_combo_paths.index(path)
            except ValueError:
                return
            try:
                with QtCore.QSignalBlocker(self.image_combo):
                    self.image_combo.setCurrentIndex(idx)
            except Exception:
                # best effort
                self.image_combo.setCurrentIndex(idx)

    def add_images(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            'Add images',
            '.',
            'Images (*.png *.jpg *.tif *.bmp)'
        )
        if not paths:
            return
        # register all; switch to the first newly added one if nothing loaded yet
        first = paths[0]
        for p in paths:
            self._ensure_image_registered(p, switch_to=False)
        if self._current_image_path is None:
            if self._reference_image_path is None:
                self._reference_image_path = first
                try:
                    arr = cv2.imread(first, cv2.IMREAD_UNCHANGED)
                    if arr is not None:
                        self._reference_image_size = (int(arr.shape[1]), int(arr.shape[0]))
                except Exception:
                    pass
            self._switch_to_image(first)
            self._ensure_image_registered(first, switch_to=True)

    def on_image_files_dropped(self, paths):
        # Drag-and-drop import onto the canvas: accept TIFF only.
        try:
            paths = list(paths) if isinstance(paths, (list, tuple)) else []
        except Exception:
            paths = []
        paths = [p for p in paths if isinstance(p, str) and p.lower().endswith(('.tif', '.tiff'))]
        if not paths:
            return

        # Register all dropped paths, then switch to the first.
        first = paths[0]
        for p in paths:
            self._ensure_image_registered(p, switch_to=False)
        self._ensure_image_registered(first, switch_to=True)

        # Mirror load_image() behavior.
        if self._reference_image_path is None:
            self._reference_image_path = first
            try:
                arr = cv2.imread(first, cv2.IMREAD_UNCHANGED)
                if arr is not None:
                    self._reference_image_size = (int(arr.shape[1]), int(arr.shape[0]))
            except Exception:
                pass

        self.statusBar().showMessage('Loading dropped image...', 2000)
        self._switch_to_image(first)

    def on_image_combo_changed(self, _idx: int):
        # ignore if combo is empty
        try:
            if getattr(self.image_combo, 'count')() <= 0:
                return
        except Exception:
            pass
        # read selected path (always via our internal mapping)
        path = None
        try:
            idx = int(self.image_combo.currentIndex())
        except Exception:
            idx = -1
        if 0 <= idx < len(self._image_combo_paths):
            path = self._image_combo_paths[idx]
        if not path:
            return
        if path == self._current_image_path:
            return
        self._switch_to_image(str(path))

    def _switch_to_image(self, path: str):
        """Switch the main canvas to `path`, preserving per-image results."""
        if not path:
            return

        if not os.path.exists(path):
            QtWidgets.QMessageBox.critical(self, 'Error', f'Image path not found:\n{path}')
            return

        # snapshot current results before switching
        self._snapshot_current_results()

        # Prefer OpenCV for size probing to avoid Qt TIFF plugin warnings for some TIFF variants.
        new_size = None
        try:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is not None:
                new_size = (int(arr.shape[1]), int(arr.shape[0]))
        except Exception:
            new_size = None

        # Establish reference/original image on first load.
        if self._reference_image_path is None:
            self._reference_image_path = path
            self._reference_image_size = new_size

        # Requirement: keep the same indexing/exclusions/masks as the original image.
        # If a grid/base-unit exists, block switching to an image with a different size.
        if self._reference_image_size is not None and new_size is not None and new_size != self._reference_image_size and (
            bool(self.img_widget.grid_rects) or bool(self.img_widget.fixed_img_rect)
        ):
            QtWidgets.QMessageBox.information(
                self,
                'Image size differs',
                'This image has a different size than the original image.\n\n'
                'Because indexing and masks must match the original, switching is blocked.\n\n'
                'Please use images with the same resolution as the original.'
            )
            return

        try:
            self.img_widget.load_image(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', str(e))
            return

        self._current_image_path = path
        self._current_image_size = (int(self.img_widget.image.width()), int(self.img_widget.image.height()))

        # Lock editing/indexing actions when viewing a non-reference image.
        is_reference = (self._reference_image_path is None) or (path == self._reference_image_path)
        try:
            if not is_reference:
                self.img_widget.drawing_enabled = False
                if hasattr(self, 'edit_btn') and self.edit_btn is not None:
                    self.edit_btn.setEnabled(False)
                    with QtCore.QSignalBlocker(self.edit_btn):
                        self.edit_btn.setChecked(False)
                    self.edit_btn.setText('Unlock Editing')
                if hasattr(self, 'apply_btn') and self.apply_btn is not None:
                    self.apply_btn.setEnabled(False)
                if hasattr(self, 'add_excl_btn') and self.add_excl_btn is not None:
                    self.add_excl_btn.setEnabled(False)
            else:
                if hasattr(self, 'edit_btn') and self.edit_btn is not None:
                    self.edit_btn.setEnabled(True)
                if hasattr(self, 'apply_btn') and self.apply_btn is not None:
                    self.apply_btn.setEnabled(bool(getattr(self.img_widget, 'drawing_enabled', True)))
                if hasattr(self, 'add_excl_btn') and self.add_excl_btn is not None:
                    self.add_excl_btn.setEnabled(True)
        except Exception:
            pass

        # reset transient visuals
        self.thumb_list.clear()
        self.img_widget.selected_cell_index = None
        self.img_widget.selected_mask_pixmap = None
        self.img_widget.cell_overlays = {}
        self.img_widget.erosion_path = None

        # keep inspection toggle state, but clear active inspection rendering until we decide what to do
        self.img_widget.inspection_mode = False
        self.img_widget.inspection_results = {}

        # rebuild per-unit pixmaps for this image if a grid exists
        if self.img_widget.grid_rects:
            self.populate_thumbnails()
            # restore per-image defect + inspection cached results
            self._restore_results_for_path(path)
            self.refresh_thumbnail_icons()
            self.refresh_canvas_overlays()

            # Auto-run segmentation for the newly selected image.
            # This is important because the mold surface can shift slightly between captures.
            try:
                self.run_segmentation_all()
            except Exception:
                pass

        # If inspection is currently ON, recompute segmentation + inspection for this image.
        inspection_on = False
        try:
            inspection_on = bool(self.run_insp_btn.isChecked())
        except Exception:
            inspection_on = False
        if inspection_on and self.img_widget.grid_rects:
            self.run_inspection()

        self.img_widget.update()

    def toggle_editing(self, checked: bool):
        # checked True => enable drawing/editing
        self.img_widget.drawing_enabled = checked
        if checked:
            self.edit_btn.setText('Lock Editing')
            self.apply_btn.setEnabled(True)
        else:
            self.edit_btn.setText('Unlock Editing')
            # keep apply enabled so user can re-lock by clicking Apply
            self.apply_btn.setEnabled(True)

    def schedule_defect_autoupdate(self, *_):
        # Debounce rapid UI changes (spinbox arrows / mouse wheel)
        # Any parameter change exits inspection mode back to overlays.
        try:
            if getattr(self.img_widget, 'inspection_mode', False):
                # also untoggle the button if present
                if hasattr(self, 'run_insp_btn') and self.run_insp_btn is not None:
                    with QtCore.QSignalBlocker(self.run_insp_btn):
                        self.run_insp_btn.setChecked(False)
                self.exit_inspection_mode(force_overlay_mode='Both')
        except Exception:
            pass
        try:
            if hasattr(self, '_defect_autoupdate_timer') and self._defect_autoupdate_timer is not None:
                self._defect_autoupdate_timer.start(250)
        except Exception:
            pass

    def _auto_update_defect_selected_unit(self):
        # Recompute defect mask for the currently selected unit, silently.
        row = self.thumb_list.currentRow()
        if row < 0 or row >= self.thumb_list.count():
            return
        item = self.thumb_list.item(row)
        pix = item.data(ROLE_BASE)
        seg_mask_pm = item.data(ROLE_BASE + 1)
        if not isinstance(pix, QtGui.QPixmap) or not isinstance(seg_mask_pm, QtGui.QPixmap):
            return
        pm_mask = self._detect_defects_on_pix(pix, seg_mask_pm, verbose=False)
        item.setData(ROLE_BASE + 2, pm_mask if isinstance(pm_mask, QtGui.QPixmap) else None)
        # refresh overlays to reflect the new mask values
        if self.img_widget.selected_cell_index == row:
            self.update_selected_overlay(row)
        self.refresh_thumbnail_icons()
        self.refresh_canvas_overlays()

    def exit_inspection_mode(self, force_overlay_mode: str = 'Both'):
        # Leave inspection mode and restore overlay rendering.
        try:
            if getattr(self.img_widget, 'inspection_mode', False):
                self.img_widget.inspection_mode = False
                self.img_widget.inspection_results = {}
        except Exception:
            pass
        # restore overlays
        try:
            if hasattr(self, 'overlay_mode') and self.overlay_mode is not None and force_overlay_mode:
                with QtCore.QSignalBlocker(self.overlay_mode):
                    self.overlay_mode.setCurrentText(str(force_overlay_mode))
        except Exception:
            pass
        self.update_selected_overlay(self.img_widget.selected_cell_index)
        self.refresh_thumbnail_icons()
        self.refresh_canvas_overlays()
        self.img_widget.update()

    def on_inspection_toggled(self, checked: bool):
        # Toggle inspection mode: ON => compute + show X/O, OFF => show overlays.
        if checked:
            ok = self.run_inspection()
            if not ok:
                # reset toggle if inspection could not run
                try:
                    with QtCore.QSignalBlocker(self.run_insp_btn):
                        self.run_insp_btn.setChecked(False)
                except Exception:
                    pass
        else:
            self.exit_inspection_mode(force_overlay_mode='Both')

    def on_cell_clicked(self, idx):
        # select thumbnail and show mask overlay for this cell
        if idx < self.thumb_list.count():
            self.thumb_list.setCurrentRow(idx)
            self.img_widget.selected_cell_index = idx
            # keep the defect "Unit index to test" in sync with click selection
            try:
                if hasattr(self, 'defect_unit_spin') and self.defect_unit_spin is not None:
                    if 0 <= idx <= self.defect_unit_spin.maximum():
                        self.defect_unit_spin.setValue(int(idx))
            except Exception:
                pass
            self.update_selected_overlay(idx)
            self.img_widget.update()

    def on_thumbnail_selected(self, row):
        if row < 0 or row >= self.thumb_list.count():
            self.img_widget.selected_cell_index = None
            self.img_widget.selected_mask_pixmap = None
            self.img_widget.update()
            return
        item = self.thumb_list.item(row)
        self.img_widget.selected_cell_index = row
        # keep the defect "Unit index to test" in sync with list selection
        try:
            if hasattr(self, 'defect_unit_spin') and self.defect_unit_spin is not None:
                if 0 <= row <= self.defect_unit_spin.maximum():
                    self.defect_unit_spin.setValue(int(row))
        except Exception:
            pass
        self.update_selected_overlay(row)

        # center and zoom to selected cell and move zoom buttons near it
        if row >= 0 and row < len(self.img_widget.grid_rects):
            self.center_on_cell(row)

    def test_defect_detection(self):
        # run defect detection on the currently selected unit (click a unit, then press Test)
        try:
            if getattr(self.img_widget, 'inspection_mode', False):
                if hasattr(self, 'run_insp_btn') and self.run_insp_btn is not None:
                    with QtCore.QSignalBlocker(self.run_insp_btn):
                        self.run_insp_btn.setChecked(False)
                self.exit_inspection_mode(force_overlay_mode='Both')
        except Exception:
            pass
        row = self.thumb_list.currentRow()
        if row < 0 and hasattr(self, 'defect_unit_spin'):
            row = int(self.defect_unit_spin.value())
        if row < 0 or row >= self.thumb_list.count():
            QtWidgets.QMessageBox.information(self, 'Info', 'Select a valid unit index first.')
            return
        item = self.thumb_list.item(row)
        pix = item.data(ROLE_BASE)
        if not isinstance(pix, QtGui.QPixmap):
            QtWidgets.QMessageBox.information(self, 'Info', 'No thumbnail image available for this unit.')
            return
        seg_mask_pm = item.data(ROLE_BASE + 1)
        if not isinstance(seg_mask_pm, QtGui.QPixmap):
            QtWidgets.QMessageBox.information(self, 'Info', 'No segmentation mask for this unit — run segmentation first.')
            return
        # When testing defects, show BOTH segmentation (green) + defect (red)
        try:
            if hasattr(self, 'overlay_mode'):
                self.overlay_mode.setCurrentText('Both')
        except Exception:
            pass
        pm_mask = self._detect_defects_on_pix(pix, seg_mask_pm)
        # store (or clear) defect mask, then refresh icons for all units
        item.setData(ROLE_BASE + 2, pm_mask if isinstance(pm_mask, QtGui.QPixmap) else None)
        self.refresh_thumbnail_icons()
        self.refresh_canvas_overlays()
        if pm_mask is None:
            QtWidgets.QMessageBox.information(self, 'Info', 'No defects found (or detection failed).')
            # still select + zoom so user can inspect the unit
            self.thumb_list.setCurrentRow(row)
            self.img_widget.selected_cell_index = row
            self.update_selected_overlay(row)
            self.center_on_cell(row)
            return
        # always select + zoom to the tested unit and show overlays
        self.thumb_list.setCurrentRow(row)
        self.img_widget.selected_cell_index = row
        self.update_selected_overlay(row)
        self.center_on_cell(row)
        # compute area and log result
        try:
            qim = pm_mask.toImage()
            arr = segmentation.qimage_to_gray_array(qim)
            stats = segmentation.mask_stats(arr)
            area = stats['area']
        except Exception:
            area = 0
        verdict = 'NG' if area >= int(self.defect_min_area.value()) else 'OK'
        self.log(f'Unit {row}: defect area={area} px -> {verdict}')

    def _detect_defects_on_pix(self, pix: QtGui.QPixmap, seg_mask_pix: QtGui.QPixmap = None, verbose: bool = True):
        # returns a QPixmap mask (grayscale) highlighting defects, or None
        def _dlog(msg: str):
            if verbose:
                self.log(msg)
        qimg = pix.toImage()
        gray = segmentation.qimage_to_gray_array(qimg)
        seg_bin = None
        # if segmentation mask provided, scale and apply it to restrict detection area
        if isinstance(seg_mask_pix, QtGui.QPixmap):
            seg_qimg = seg_mask_pix.toImage().scaled(
                qimg.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            seg_arr = segmentation.qimage_to_gray_array(seg_qimg)
            erode_px = int(self.defect_mask_erode.value()) if hasattr(self, 'defect_mask_erode') else 0
            # Use the segmentation mask exactly as the ROI (match what the Segmentation overlay shows)
            seg_bin = (seg_arr > 0).astype(np.uint8) * 255
            try:
                seg_area0 = int((seg_bin > 0).sum())
            except Exception:
                seg_area0 = 0
            _dlog(f'Seg mask area (roi)={seg_area0}, erode_px={erode_px}')
            if erode_px > 0:
                try:
                    seg_bin = cv2.erode(seg_bin, None, iterations=erode_px)
                except Exception:
                    pass
            # Keep only the largest connected ROI component after erosion.
            # IMPORTANT: do NOT use filled external contours here, because that would fill internal holes
            # (including user exclusions). Use connected components so holes remain holes.
            try:
                cc_src = (seg_bin > 0).astype(np.uint8)
                nlab, labels, stats, _ = cv2.connectedComponentsWithStats(cc_src, connectivity=8)
                if nlab > 1:
                    # skip background label 0
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    best = 1 + int(np.argmax(areas))
                    seg_bin = (labels == best).astype(np.uint8) * 255
            except Exception:
                pass
            # if segmentation mask is empty after normalization/erosion, skip detection
            if seg_bin is None or seg_bin.sum() == 0:
                _dlog('Segmentation mask empty after erode — skipping detection for this unit')
                return None
        method = str(self.defect_method.currentText())
        thr = int(self.defect_threshold.value())
        if method == 'threshold':
            # Local anomaly detection: threshold the absolute difference from a local median background.
            # This is much more stable than a global gray threshold for spotting foreign material.
            k = 21
            if k % 2 == 0:
                k += 1
            bg = cv2.medianBlur(gray, k)
            resid = cv2.absdiff(gray, bg)
            _, mask = cv2.threshold(resid, thr, 255, cv2.THRESH_BINARY)
            if seg_bin is not None:
                mask = cv2.bitwise_and(mask, seg_bin)
            # clean small pepper noise
            try:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            except Exception:
                pass
            _dlog(f'Residual mask area={int((mask > 0).sum())}')
        else:
            mask = cv2.Canny(gray, max(1, thr//2), max(2, thr))
            if seg_bin is not None:
                mask = cv2.bitwise_and(mask, seg_bin)
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask2 = np.zeros_like(mask)
        min_area = int(self.defect_min_area.value())
        # allow very large defects, but reject "whole part" masks (shouldn't happen often with residual-based mask)
        try:
            seg_area = int((seg_bin > 0).sum()) if seg_bin is not None else int(gray.shape[0] * gray.shape[1])
        except Exception:
            seg_area = int(gray.shape[0] * gray.shape[1])
        max_area = max(min_area, int(seg_area * 0.98))
        _dlog(f'Defect area filter: min={min_area}, max={max_area}, seg_area={seg_area}')
        found = False
        for c in cnts:
            a = cv2.contourArea(c)
            if a >= min_area and a <= max_area:
                cv2.drawContours(mask2, [c], -1, 255, -1)
                found = True
            else:
                if a >= min_area:
                    _dlog(f'Skipping large contour area={int(a)} (>max={max_area})')
        if not found:
            return None
        h_m, w_m = mask2.shape
        bytes_per_line = w_m
        # IMPORTANT: detach from temporary numpy/bytes buffer to avoid native crashes
        qimg_mask = QtGui.QImage(
            mask2.data.tobytes(),
            w_m,
            h_m,
            bytes_per_line,
            QtGui.QImage.Format.Format_Grayscale8,
        ).copy()
        pm_mask = QtGui.QPixmap.fromImage(qimg_mask)
        return pm_mask

    def test_defect_detection_all(self):
        # run defect detection on all thumbnails and update thumbnails/icons
        try:
            if getattr(self.img_widget, 'inspection_mode', False):
                if hasattr(self, 'run_insp_btn') and self.run_insp_btn is not None:
                    with QtCore.QSignalBlocker(self.run_insp_btn):
                        self.run_insp_btn.setChecked(False)
                self.exit_inspection_mode(force_overlay_mode='Both')
        except Exception:
            pass
        count = self.thumb_list.count()
        if count == 0:
            QtWidgets.QMessageBox.information(self, 'Info', 'No units available.')
            return
        self.statusBar().showMessage('Running defect detection on all units...')
        # When batch testing defects, show BOTH segmentation (green) + defect (red)
        try:
            if hasattr(self, 'overlay_mode'):
                self.overlay_mode.setCurrentText('Both')
        except Exception:
            pass
        processed = 0
        for row in range(count):
            item = self.thumb_list.item(row)
            pix = item.data(ROLE_BASE)
            if not isinstance(pix, QtGui.QPixmap):
                self.log(f'Unit {row}: no thumbnail, skipping')
                continue
            seg_mask_pm = item.data(ROLE_BASE + 1)
            if not isinstance(seg_mask_pm, QtGui.QPixmap):
                self.log(f'Unit {row}: no segmentation mask, skipping')
                continue
            pm_mask = self._detect_defects_on_pix(pix, seg_mask_pm)
            # store (or clear) defect mask; icons will be refreshed for all items after the loop
            item.setData(ROLE_BASE + 2, pm_mask if isinstance(pm_mask, QtGui.QPixmap) else None)
            if pm_mask:
                # compute area and verdict and log
                try:
                    qim = pm_mask.toImage()
                    arr = segmentation.qimage_to_gray_array(qim)
                    stats = segmentation.mask_stats(arr)
                    area = stats['area']
                except Exception:
                    area = 0
                verdict = 'NG' if area >= int(self.defect_min_area.value()) else 'OK'
                self.log(f'Unit {row}: defect area={area} px -> {verdict}')
                processed += 1
            else:
                self.log(f'Unit {row}: no defects')
        # show overlays on ALL thumbnails according to the current overlay mode
        self.refresh_thumbnail_icons()
        self.refresh_canvas_overlays()
        self.statusBar().showMessage(f'Defect detection completed: {processed}/{count} units had detections', 4000)
        # ensure something is selected so the main view shows overlays immediately
        if self.thumb_list.currentRow() < 0 and self.thumb_list.count() > 0:
            self.thumb_list.setCurrentRow(0)
        if self.img_widget.selected_cell_index is not None:
            self.update_selected_overlay(self.img_widget.selected_cell_index)
            # Do not auto-zoom/center after batch runs; keep user's current view.

    def run_inspection(self) -> bool:
        # Run defect detection across all units and show only X/O verdict markers (no overlays)
        count = self.thumb_list.count()
        if count == 0:
            QtWidgets.QMessageBox.information(self, 'Info', 'No units available.')
            return False

        # Ensure segmentation mask exists for the current image
        has_seg = False
        for i in range(count):
            it = self.thumb_list.item(i)
            if it is not None and isinstance(it.data(ROLE_BASE + 1), QtGui.QPixmap):
                has_seg = True
                break
        if not has_seg:
            self.run_segmentation_all()
            for i in range(count):
                it = self.thumb_list.item(i)
                if it is not None and isinstance(it.data(ROLE_BASE + 1), QtGui.QPixmap):
                    has_seg = True
                    break
            if not has_seg:
                QtWidgets.QMessageBox.information(
                    self,
                    'Segmentation mask missing',
                    'Segmentation could not be computed for this image.\n\n'
                    'Run segmentation once and try again.'
                )
                return False
        self.statusBar().showMessage('Running inspection on all units...')

        results = {}
        ng_count = 0
        min_area = int(self.defect_min_area.value()) if hasattr(self, 'defect_min_area') else 0

        for row in range(count):
            item = self.thumb_list.item(row)
            try:
                grid_idx = int(item.text())
            except Exception:
                grid_idx = row

            pix = item.data(ROLE_BASE)
            seg_mask_pm = item.data(ROLE_BASE + 1)
            if not isinstance(pix, QtGui.QPixmap) or not isinstance(seg_mask_pm, QtGui.QPixmap):
                # no data => leave as unknown (no marker)
                continue

            pm_mask = self._detect_defects_on_pix(pix, seg_mask_pm, verbose=False)
            # store defect mask so returning to overlay view is instant
            item.setData(ROLE_BASE + 2, pm_mask if isinstance(pm_mask, QtGui.QPixmap) else None)

            if pm_mask is None:
                results[grid_idx] = False
                continue

            # compute area and verdict like the existing "Test" flow
            try:
                qim = pm_mask.toImage()
                arr = segmentation.qimage_to_gray_array(qim)
                stats = segmentation.mask_stats(arr)
                area = int(stats.get('area', 0))
            except Exception:
                area = 0

            is_ng = area >= min_area
            results[grid_idx] = bool(is_ng)
            if is_ng:
                ng_count += 1

        # switch to inspection mode: hide overlays and show X/O
        self.img_widget.inspection_results = results
        self.img_widget.inspection_mode = True
        self.img_widget.update()
        self.statusBar().showMessage(f'Inspection complete: {ng_count}/{count} units NG', 4000)
        return True

    def center_on_cell(self, row: int):
        # Ensure the cell at `row` is visible and centered with an appropriate zoom.
        if row < 0 or row >= len(self.img_widget.grid_rects):
            return
        r, idx = self.img_widget.grid_rects[row]
        img_r = QtCore.QRect(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        vp = self.scroll.viewport()
        vw = vp.width(); vh = vp.height()
        iw = self.img_widget.image.width(); ih = self.img_widget.image.height()
        base = min(vw / iw, vh / ih) if iw and ih else 1.0
        frac = 0.8
        desired_scale_x = (vw * frac) / max(1, img_r.width())
        desired_scale_y = (vh * frac) / max(1, img_r.height())
        desired_scale = min(desired_scale_x, desired_scale_y)
        if base > 0:
            self.img_widget.manual_zoom = max(0.1, desired_scale / base)
        # update widget scale/size
        self.img_widget.updateScale()
        self.img_widget.update()
        QtWidgets.QApplication.processEvents()
        # compute display rect and center it in scroll
        dr = self.img_widget.imgrect_to_display(img_r)
        center_x = dr.x() + dr.width() // 2
        center_y = dr.y() + dr.height() // 2
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        target_h = max(0, center_x - vw // 2)
        target_v = max(0, center_y - vh // 2)
        target_h = min(hbar.maximum(), target_h)
        target_v = min(vbar.maximum(), target_v)
        hbar.setValue(int(target_h))
        vbar.setValue(int(target_v))
        # move zoom buttons near the selected cell top-right (clamped to viewport)
        viewport_x = dr.x() - hbar.value()
        viewport_y = dr.y() - vbar.value()
        btn_x = viewport_x + dr.width() - self.zoom_in_btn.width()
        btn_y = viewport_y
        btn_x = max(8, min(vw - self.zoom_in_btn.width() - 8, btn_x))
        btn_y = max(8, min(vh - self.zoom_in_btn.height() - 8, btn_y))
        self.zoom_in_btn.move(int(btn_x), int(btn_y))
        self.zoom_out_btn.move(int(max(8, btn_x - self.zoom_out_btn.width() - 6)), int(btn_y))
        # segmentation param changes update nothing automatically; user runs explicitly
        # segmentation params: debounce and run automatically
        self._seg_debounce_timer = QtCore.QTimer(self)
        self._seg_debounce_timer.setSingleShot(True)
        self._seg_debounce_timer.setInterval(400)  # ms
        self._seg_debounce_timer.timeout.connect(self.run_segmentation_all)
        # connect segmentation control changes to debounce timer
        self.seg_method.currentIndexChanged.connect(lambda _: self._seg_debounce_timer.start())
        self.gauss_spin.valueChanged.connect(lambda _: self._seg_debounce_timer.start())
        self.morph_spin.valueChanged.connect(lambda _: self._seg_debounce_timer.start())
        self.adapt_block.valueChanged.connect(lambda _: self._seg_debounce_timer.start())
        self.adapt_C.valueChanged.connect(lambda _: self._seg_debounce_timer.start())
    def add_exclusion(self):
        # Exclusions must match the original image (reference) so they apply consistently.
        if (
            self._reference_image_path
            and self._current_image_path
            and self._current_image_path != self._reference_image_path
        ):
            QtWidgets.QMessageBox.information(
                self,
                'Exclusions locked',
                'Exclusions are defined on the original image only.\n\n'
                'Switch to the original image to add or edit exclusions.'
            )
            return
        # zoom to first unit and enable drawing an exclusion
        if not self.img_widget.grid_rects:
            QtWidgets.QMessageBox.information(self, 'Info', 'Create indexing first before adding exclusions.')
            return
        # zoom to first unit and enable exclusion drawing
        self.on_thumbnail_selected(0)
        # enable drawing temporarily regardless of edit lock
        self.img_widget.exclusion_mode = True
        # IMPORTANT: do not permanently unlock base-unit editing.
        # Store the previous state on the widget so it can restore it immediately on mouse release.
        try:
            self.img_widget._drawing_enabled_before_exclusion = bool(getattr(self.img_widget, 'drawing_enabled', True))
        except Exception:
            self.img_widget._drawing_enabled_before_exclusion = True
        self.img_widget.drawing_enabled = True
        # update UI hint
        self.statusBar().showMessage('Draw exclusion on the selected unit (click-drag).', 4000)

    def on_exclusion_drawn(self, excl_rect):
        # excl_rect is a QRect in image coordinates; base unit is grid_rects[0]
        if not self.img_widget.grid_rects:
            return
        base_r, _ = self.img_widget.grid_rects[0]
        bx, by, bw, bh = int(base_r[0]), int(base_r[1]), int(base_r[2]), int(base_r[3])
        # clamp intersection
        x = max(bx, excl_rect.x()); y = max(by, excl_rect.y())
        x2 = min(bx + bw, excl_rect.x() + excl_rect.width()); y2 = min(by + bh, excl_rect.y() + excl_rect.height())
        if x2 <= x or y2 <= y:
            self.statusBar().showMessage('Invalid exclusion (outside unit).', 3000)
            return
        rel_x = x - bx; rel_y = y - by; rel_w = x2 - x; rel_h = y2 - y
        shape = str(self.excl_shape.currentText())
        if shape == 'rectangle':
            excl = {'shape': 'rect', 'x': int(rel_x), 'y': int(rel_y), 'w': int(rel_w), 'h': int(rel_h)}
        else:
            # circle: take bounding rect center and radius
            cx = rel_x + rel_w/2.0; cy = rel_y + rel_h/2.0
            r = int(min(rel_w, rel_h) / 2.0)
            excl = {'shape': 'circle', 'cx': int(cx), 'cy': int(cy), 'r': int(r)}
        self.exclusions.append(excl)
        # update exclusion index range
        self.excl_index.setRange(0, max(0, len(self.exclusions)-1))
        self.excl_index.setValue(len(self.exclusions)-1)
        self.statusBar().showMessage(f'Added exclusion #{len(self.exclusions)-1}', 3000)
        # run segmentation to apply exclusions
        self.run_segmentation_all()

        # refresh settings UI + (optional) edit overlay
        try:
            self.on_exclusion_index_changed()
        except Exception:
            pass

        # Keep the UI controls consistent with the restored state.
        try:
            prev = getattr(self.img_widget, '_drawing_enabled_before_exclusion', None)
            if isinstance(prev, bool) and not prev:
                if hasattr(self, 'apply_btn') and self.apply_btn is not None:
                    self.apply_btn.setEnabled(False)
                if hasattr(self, 'edit_btn') and self.edit_btn is not None:
                    self.edit_btn.setText('Unlock Editing')
        except Exception:
            pass

    def open_modify_exclusion_dialog(self):
        # Exclusions must match the original image (reference) so they apply consistently.
        if (
            self._reference_image_path
            and self._current_image_path
            and self._current_image_path != self._reference_image_path
        ):
            QtWidgets.QMessageBox.information(
                self,
                'Exclusions locked',
                'Exclusions are defined on the original image only.\n\n'
                'Switch to the original image to add or edit exclusions.'
            )
            return

        if not getattr(self, 'exclusions', None):
            self.statusBar().showMessage('No exclusions to modify.', 2500)
            return

        # close previous dialog instance if still around
        try:
            if self._modify_dialog is not None:
                self._modify_dialog.close()
        except Exception:
            pass

        dlg = ModifyExclusionDialog(self)
        self._modify_dialog = dlg
        try:
            dlg.finished.connect(lambda *_: setattr(self, '_modify_dialog', None))
        except Exception:
            pass
        dlg.exec()

    def _selected_exclusion_index(self):
        try:
            idx = int(self.excl_index.value())
        except Exception:
            return None
        if not getattr(self, 'exclusions', None):
            return None
        if idx < 0 or idx >= len(self.exclusions):
            return None
        return idx

    def _get_base_unit_rect(self):
        if not self.img_widget.grid_rects:
            return None
        base_r, _ = self.img_widget.grid_rects[0]
        bx, by, bw, bh = int(base_r[0]), int(base_r[1]), int(base_r[2]), int(base_r[3])
        return bx, by, bw, bh

    def _get_exclusion_img_geometry(self, excl):
        base = self._get_base_unit_rect()
        if base is None:
            return None
        bx, by, bw, bh = base
        try:
            shape = excl.get('shape')
        except Exception:
            shape = None

        if shape == 'rect':
            x = int(excl.get('x', 0)); y = int(excl.get('y', 0))
            w = int(excl.get('w', 1)); h = int(excl.get('h', 1))
            # clamp to base unit bounds
            w = max(1, min(w, bw - x))
            h = max(1, min(h, bh - y))
            rect = QtCore.QRect(int(bx + x), int(by + y), int(w), int(h))
            return ('rect', rect, None)

        if shape == 'circle':
            cx = int(excl.get('cx', 0)); cy = int(excl.get('cy', 0)); r = int(excl.get('r', 1))
            r = max(1, r)
            # clamp so circle stays in base unit
            r = min(r, max(1, cx), max(1, cy), max(1, bw - cx - 1), max(1, bh - cy - 1))
            return ('circle', None, (int(bx + cx), int(by + cy), int(r)))

        return None

    def _clamp_exclusion_to_base_unit(self, excl: dict) -> dict:
        base = self._get_base_unit_rect()
        if base is None:
            return excl
        _, _, bw, bh = base
        shape = excl.get('shape')
        if shape == 'rect':
            x = int(excl.get('x', 0)); y = int(excl.get('y', 0))
            w = int(excl.get('w', 1)); h = int(excl.get('h', 1))
            x = max(0, min(x, max(0, bw - 1)))
            y = max(0, min(y, max(0, bh - 1)))
            w = max(1, min(w, max(1, bw - x)))
            h = max(1, min(h, max(1, bh - y)))
            excl['x'] = x; excl['y'] = y; excl['w'] = w; excl['h'] = h
            return excl
        if shape == 'circle':
            cx = int(excl.get('cx', 0)); cy = int(excl.get('cy', 0)); r = int(excl.get('r', 1))
            cx = max(0, min(cx, max(0, bw - 1)))
            cy = max(0, min(cy, max(0, bh - 1)))
            r = max(1, r)
            # stay within bounds
            r = min(r, max(1, cx), max(1, cy), max(1, bw - cx - 1), max(1, bh - cy - 1))
            excl['cx'] = cx; excl['cy'] = cy; excl['r'] = r
            return excl
        return excl

    def _refresh_exclusion_edit_overlay(self, idx: int):
        try:
            excl = self.exclusions[idx]
        except Exception:
            return
        geo = self._get_exclusion_img_geometry(excl)
        if geo is None:
            return
        shape, rect, circle = geo
        if shape == 'rect':
            self.img_widget.set_exclusion_edit('rect', rect=rect, circle=None)
        else:
            self.img_widget.set_exclusion_edit('circle', rect=None, circle=circle)

    def toggle_edit_exclusion(self):
        # Exclusions must match the original image (reference) so they apply consistently.
        if (
            self._reference_image_path
            and self._current_image_path
            and self._current_image_path != self._reference_image_path
        ):
            QtWidgets.QMessageBox.information(
                self,
                'Exclusions locked',
                'Exclusions are defined on the original image only.\n\n'
                'Switch to the original image to add or edit exclusions.'
            )
            return

        if not getattr(self, 'exclusions', None):
            self.statusBar().showMessage('No exclusions to edit.', 2500)
            return

        if getattr(self, '_exclusion_edit_active', False):
            self._exclusion_edit_active = False
            try:
                self.img_widget.clear_exclusion_edit()
            except Exception:
                pass
            self.statusBar().showMessage('Exclusion edit mode off.', 2500)
            return

        idx = self._selected_exclusion_index()
        if idx is None:
            self.statusBar().showMessage('Select an exclusion index first.', 2500)
            return

        # zoom to first unit for consistent editing
        try:
            self.on_thumbnail_selected(0)
        except Exception:
            pass

        self._exclusion_edit_active = True
        try:
            self._refresh_exclusion_edit_overlay(idx)
        except Exception:
            pass

        self.statusBar().showMessage('Edit exclusion: drag the arrow handle or change W/L.', 4000)

    def delete_exclusion(self):
        # Exclusions must match the original image (reference) so they apply consistently.
        if (
            self._reference_image_path
            and self._current_image_path
            and self._current_image_path != self._reference_image_path
        ):
            QtWidgets.QMessageBox.information(
                self,
                'Exclusions locked',
                'Exclusions are defined on the original image only.\n\n'
                'Switch to the original image to add or edit exclusions.'
            )
            return

        idx = self._selected_exclusion_index()
        if idx is None:
            self.statusBar().showMessage('No exclusion selected.', 2500)
            return
        self.delete_exclusion_at(idx)

    def delete_exclusion_at(self, idx: int):
        try:
            del self.exclusions[idx]
        except Exception:
            return

        # update range/value
        if not self.exclusions:
            self.excl_index.setRange(0, 0)
            self.excl_index.setValue(0)
            self._exclusion_edit_active = False
            try:
                self.img_widget.clear_exclusion_edit()
            except Exception:
                pass
        else:
            self.excl_index.setRange(0, len(self.exclusions) - 1)
            self.excl_index.setValue(min(idx, len(self.exclusions) - 1))
            self.on_exclusion_index_changed()

        # if dialog is open, refresh it
        try:
            if self._modify_dialog is not None:
                self._modify_dialog.reload_from_main()
        except Exception:
            pass

        self.statusBar().showMessage('Deleted exclusion.', 2500)
        self.run_segmentation_all()

    def on_exclusion_index_changed(self, *args):
        # update enable states + edit overlay refresh
        has_excl = bool(getattr(self, 'exclusions', None))
        try:
            self.modify_excl_btn.setEnabled(has_excl)
        except Exception:
            pass

        idx = self._selected_exclusion_index()
        if idx is None:
            try:
                self.img_widget.clear_exclusion_edit()
            except Exception:
                pass
            return

        # if currently editing, refresh overlay to match the newly selected exclusion
        if getattr(self, '_exclusion_edit_active', False):
            try:
                self._refresh_exclusion_edit_overlay(idx)
            except Exception:
                pass

    def on_exclusion_edit_updated(self, info):
        # live update from the on-canvas drag handle
        idx = self._selected_exclusion_index()
        if idx is None:
            return
        if not isinstance(info, dict):
            return
        excl = self.exclusions[idx]
        shape = excl.get('shape')
        if info.get('shape') == 'rect' and shape == 'rect':
            w = int(info.get('w', excl.get('w', 1)))
            h = int(info.get('h', excl.get('h', 1)))
            excl['w'] = max(1, w)
            excl['h'] = max(1, h)
            # sync dialog fields if open
            try:
                if self._modify_dialog is not None:
                    self._modify_dialog.sync_from_main()
            except Exception:
                pass
        if info.get('shape') == 'circle' and shape == 'circle':
            r = int(info.get('r', excl.get('r', 1)))
            excl['r'] = max(1, r)
            try:
                if self._modify_dialog is not None:
                    self._modify_dialog.sync_from_main()
            except Exception:
                pass

    def on_exclusion_edit_committed(self, info):
        # commit from the on-canvas drag handle
        self.on_exclusion_edit_updated(info)
        self.run_segmentation_all()


    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', '.', 'Images (*.png *.jpg *.tif *.bmp)')
        if path:
            self._ensure_image_registered(path, switch_to=True)
            # If no reference image yet, make this the reference/original.
            if self._reference_image_path is None:
                self._reference_image_path = path
                try:
                    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if arr is not None:
                        self._reference_image_size = (int(arr.shape[1]), int(arr.shape[0]))
                except Exception:
                    pass
            self._switch_to_image(path)

    def apply_indexing(self):
        # Indexing must be defined on the original (reference) image so it stays consistent across images.
        if self._reference_image_path is None and self._current_image_path is not None:
            self._reference_image_path = self._current_image_path
            try:
                self._reference_image_size = (
                    int(self.img_widget.image.width()),
                    int(self.img_widget.image.height()),
                )
            except Exception:
                self._reference_image_size = None

        if (
            self._reference_image_path
            and self._current_image_path
            and self._current_image_path != self._reference_image_path
        ):
            QtWidgets.QMessageBox.information(
                self,
                'Indexing locked',
                'Indexing can only be applied on the original image so it stays consistent across images.\n\n'
                'Switch back to the original image, then apply indexing.'
            )
            return

        if not self.img_widget.fixed_img_rect:
            QtWidgets.QMessageBox.information(self, 'Info', 'Please draw the first unit with click-drag on image.')
            return
        # call the preview generator and then show done message
        count = self.update_grid_preview()
        # lock drawing so user can't change base unit until unlocked
        self.img_widget.drawing_enabled = False
        self.edit_btn.setChecked(False)
        self.edit_btn.setText('Unlock Editing')
        self.apply_btn.setEnabled(False)
        QtWidgets.QMessageBox.information(self, 'Done', f'Generated {count} unit bounding boxes. Editing locked.')
        return

    def update_grid_preview(self):
        # generate grid from current parameters; returns count
        if not self.img_widget.fixed_img_rect or not self.img_widget.image:
            return 0
        r = self.img_widget.fixed_img_rect
        ux = self.units_x.value(); uy = self.units_y.value()
        bx = self.blocks_x.value(); by = self.blocks_y.value()
        sux = self.unit_space_x.value(); suy = self.unit_space_y.value()
        sbx = self.block_space_x.value(); sby = self.block_space_y.value()

        unit_w = r.width(); unit_h = r.height()
        grid = []
        idx = 0
        for byi in range(by):
            for uyi in range(uy):
                for bxi in range(bx):
                    for uxi in range(ux):
                        x = r.x() + bxi * (ux * unit_w + (ux - 1) * sux + sbx) + uxi * (unit_w + sux)
                        y = r.y() + byi * (uy * unit_h + (uy - 1) * suy + sby) + uyi * (unit_h + suy)
                        grid.append(((int(x), int(y), int(unit_w), int(unit_h)), idx))
                        idx += 1

        self.img_widget.grid_rects = grid
        self.img_widget.update()
        self.populate_thumbnails()
        # grid changed: reference centroids no longer valid
        try:
            self._exclusion_ref_centroids = {}
        except Exception:
            pass
        return len(grid)

    def run_segmentation_all(self):
        if not self.img_widget.grid_rects or not self.img_widget.image:
            self.statusBar().showMessage('Segmentation skipped: no grid available', 3000)
            return
        # When running segmentation (including after exclusions), switch overlay to Segmentation
        try:
            if hasattr(self, 'overlay_mode'):
                self.overlay_mode.setCurrentText('Segmentation')
        except Exception:
            pass
        # Build a cache of reference segmentation masks (unit-local) for exclusion alignment.
        ref_seg_bins = None
        ref_path = getattr(self, '_reference_image_path', None)
        cur_path = getattr(self, '_current_image_path', None)
        if ref_path and cur_path and ref_path != cur_path:
            try:
                st = self._image_states.get(ref_path) or {}
                ref_masks = st.get('seg') or []
                if ref_masks:
                    ref_seg_bins = {}
                    for i, pm in enumerate(ref_masks):
                        if not isinstance(pm, QtGui.QPixmap):
                            continue
                        try:
                            q = pm.toImage()
                            arr = segmentation.qimage_to_gray_array(q)
                            ref_seg_bins[i] = (arr > 0).astype(np.uint8) * 255
                        except Exception:
                            continue
            except Exception:
                ref_seg_bins = None

        def _largest_component_centroid(bin_mask: np.ndarray):
            if bin_mask is None or bin_mask.size == 0:
                return None
            try:
                src = (bin_mask > 0).astype(np.uint8)
                nlab, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
                if nlab <= 1:
                    return None
                areas = stats[1:, cv2.CC_STAT_AREA]
                best = 1 + int(np.argmax(areas))
                ys, xs = np.where(labels == best)
                if xs.size == 0:
                    return None
                return (float(xs.mean()), float(ys.mean()))
            except Exception:
                try:
                    ys, xs = np.where(bin_mask > 0)
                    if xs.size == 0:
                        return None
                    return (float(xs.mean()), float(ys.mean()))
                except Exception:
                    return None

        # Determine whether we're segmenting the reference image now.
        is_reference = bool(ref_path and cur_path and ref_path == cur_path)
        if is_reference:
            # Reset on fresh reference segmentation run
            try:
                self._exclusion_ref_centroids = {}
            except Exception:
                pass

        # iterate thumbnails and compute masks
        for idx, (r, _) in enumerate(self.img_widget.grid_rects):
            x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            qimg_crop = self.img_widget.image.copy(x, y, w, h)
            gray = segmentation.qimage_to_gray_array(qimg_crop)
            method = str(self.seg_method.currentText())
            mask = segmentation.segment_cell(gray, method=method,
                                             adapt_block=self.adapt_block.value(),
                                             adapt_C=self.adapt_C.value(),
                                             gaussian_blur=self.gauss_spin.value(),
                                             morph_kernel=self.morph_spin.value())

            # Pre-exclusion mask used for alignment anchors.
            pre_excl_bin = (mask > 0).astype(np.uint8) * 255

            # If this is the reference image, record the reference centroid for this unit.
            if is_reference:
                try:
                    c_ref = _largest_component_centroid(pre_excl_bin)
                    if c_ref is not None:
                        self._exclusion_ref_centroids[int(idx)] = (float(c_ref[0]), float(c_ref[1]))
                except Exception:
                    pass

            # Estimate per-unit XY shift vs reference (based on segmentation ROI centroid) so exclusions track the mold.
            dx = 0
            dy = 0
            if not is_reference:
                try:
                    c1 = _largest_component_centroid(pre_excl_bin)
                    c0 = None
                    # Prefer persisted reference centroids (works even if reference segmentation isn't loaded now).
                    try:
                        rc = self._exclusion_ref_centroids.get(int(idx))
                        if rc is not None:
                            c0 = (float(rc[0]), float(rc[1]))
                    except Exception:
                        c0 = None
                    # Fallback to reference masks cache if available
                    if c0 is None and ref_seg_bins is not None and idx in ref_seg_bins:
                        c0 = _largest_component_centroid(ref_seg_bins.get(idx))

                    if c0 is not None and c1 is not None:
                        dx = int(round(c1[0] - c0[0]))
                        dy = int(round(c1[1] - c0[1]))
                except Exception:
                    dx = 0
                    dy = 0

            # apply any user-defined exclusions (relative to unit; shifted by dx/dy)
            for excl in getattr(self, 'exclusions', []):
                try:
                    if excl.get('shape') == 'rect':
                        ex = int(excl.get('x', 0)) + dx
                        ey = int(excl.get('y', 0)) + dy
                        ew = int(excl.get('w', 0)); eh = int(excl.get('h', 0))
                        x0 = max(0, ex); y0 = max(0, ey)
                        x1 = min(w, ex + ew); y1 = min(h, ey + eh)
                        if x1 > x0 and y1 > y0:
                            mask[y0:y1, x0:x1] = 0
                    else:
                        # circle
                        cx = int(excl.get('cx', 0)) + dx
                        cy = int(excl.get('cy', 0)) + dy
                        r = int(excl.get('r', 0))
                        if r > 0:
                            yy, xx = np.ogrid[:h, :w]
                            circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r ** 2)
                            mask[circle] = 0
                except Exception:
                    # be resilient to malformed exclusion entries
                    continue
            # convert mask to QPixmap (image-size)
            h_m, w_m = mask.shape
            bytes_per_line = w_m
            # IMPORTANT: detach from temporary numpy/bytes buffer to avoid native crashes
            qimg_mask = QtGui.QImage(
                mask.data.tobytes(),
                w_m,
                h_m,
                bytes_per_line,
                QtGui.QImage.Format.Format_Grayscale8,
            ).copy()
            pm_mask = QtGui.QPixmap.fromImage(qimg_mask)
            # store full-resolution mask in corresponding thumbnail item if exists
            # find thumbnail item by index
            if idx < self.thumb_list.count():
                item = self.thumb_list.item(idx)
                item.setData(ROLE_BASE + 1, pm_mask)
                # thumbnail icons are refreshed after the loop according to overlay mode
            # if this cell is currently selected, update main overlay
            if self.img_widget.selected_cell_index == idx:
                # refresh selected overlay according to current mode
                self.update_selected_overlay(idx)
        # repaint main image to show overlays if any
        self.refresh_thumbnail_icons()
        self.refresh_canvas_overlays()
        self.img_widget.update()
        self.statusBar().showMessage('Segmentation completed', 2000)

        # Cache masks for this image for fast switching.
        self._snapshot_current_results()

    def on_overlay_mode_changed(self, *_):
        # update selected overlay and all thumbnail icons
        # If the user picks an overlay mode, leave inspection mode.
        try:
            if getattr(self.img_widget, 'inspection_mode', False):
                self.img_widget.inspection_mode = False
                self.img_widget.inspection_results = {}
        except Exception:
            pass
        self.update_selected_overlay()
        self.refresh_thumbnail_icons()
        self.refresh_canvas_overlays()

    def refresh_canvas_overlays(self):
        # Build tinted per-cell overlays for drawing on the full image canvas.
        overlays = {}
        for row in range(self.thumb_list.count()):
            item = self.thumb_list.item(row)
            try:
                grid_idx = int(item.text())
            except Exception:
                grid_idx = row
            seg_pm = item.data(ROLE_BASE + 1)
            defect_pm = item.data(ROLE_BASE + 2)
            seg_t = None
            defect_t = None
            if isinstance(seg_pm, QtGui.QPixmap):
                seg_t = self._tint_mask_pixmap(seg_pm, color=(0, 255, 0), alpha_val=140)
            if isinstance(defect_pm, QtGui.QPixmap):
                defect_t = self._tint_mask_pixmap(defect_pm, color=(255, 0, 0), alpha_val=180)
            overlays[grid_idx] = {'seg': seg_t, 'defect': defect_t}
        self.img_widget.cell_overlays = overlays
        try:
            self.img_widget.overlay_mode = str(self.overlay_mode.currentText())
        except Exception:
            pass
        self.img_widget.update()

    def refresh_thumbnail_icons(self, mode_override: str = None):
        # Update ALL thumbnail icons according to current overlay mode (or an override) and stored masks.
        mode = str(mode_override) if mode_override else (str(self.overlay_mode.currentText()) if hasattr(self, 'overlay_mode') else 'Segmentation')
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            base_pm = item.data(ROLE_BASE)
            if not isinstance(base_pm, QtGui.QPixmap):
                continue
            base_disp = base_pm.scaled(
                128,
                128,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            seg_pm = item.data(ROLE_BASE + 1)
            defect_pm = item.data(ROLE_BASE + 2)

            if mode == 'None':
                item.setIcon(QtGui.QIcon(base_disp))
                continue

            out = base_disp
            if mode in ('Segmentation', 'Both') and isinstance(seg_pm, QtGui.QPixmap):
                seg_scaled = seg_pm.scaled(
                    base_disp.size(),
                    QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                out = self._make_overlay_pixmap(out, seg_scaled, color=(0, 255, 0))
            if mode in ('Defect', 'Both') and isinstance(defect_pm, QtGui.QPixmap):
                defect_scaled = defect_pm.scaled(
                    base_disp.size(),
                    QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                out = self._make_overlay_pixmap(out, defect_scaled, color=(255, 0, 0))

            item.setIcon(QtGui.QIcon(out))

    def _make_overlay_pixmap(self, pix, mask_pix, color=(255, 0, 0), alpha_val=200):
        # overlay mask (colored) on cell pixmap
        base = QtGui.QPixmap(pix)
        mask = QtGui.QPixmap(mask_pix)
        # ensure same size
        mask = mask.scaled(
            base.size(),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        result = QtGui.QPixmap(base.size())
        result.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(result)
        p.drawPixmap(0, 0, base)
        p.setOpacity(0.5)
        # tint mask with requested color
        tinted = self._tint_mask_pixmap(mask, color=color, alpha_val=alpha_val)
        p.drawPixmap(0, 0, tinted)
        p.end()
        return result

    def _tint_mask_pixmap(self, mask_pix, color=(255, 0, 0), alpha_val=200):
        # create a colored ARGB pixmap where mask non-zero pixels get the given color and alpha
        mask = QtGui.QPixmap(mask_pix)
        mask_img = mask.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        h = mask_img.height(); w = mask_img.width()
        bits = mask_img.bits()
        bc = getattr(mask_img, "sizeInBytes", None)
        bc = bc() if callable(bc) else mask_img.byteCount()
        bits.setsize(int(bc))
        arr = np.frombuffer(bits, np.uint8).reshape((h, w, 4))
        mask_alpha = (arr[:, :, 0] > 0).astype(np.uint8) * alpha_val
        out_img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        out_img.fill(0)
        ob = out_img.bits()
        bc2 = getattr(out_img, "sizeInBytes", None)
        bc2 = bc2() if callable(bc2) else out_img.byteCount()
        ob.setsize(int(bc2))
        oarr = np.frombuffer(ob, np.uint8).reshape((h, w, 4))
        # assign color (B,G,R order in QImage byte layout)
        oarr[:, :, 0] = color[2] if len(color) >= 3 else 0
        oarr[:, :, 1] = color[1] if len(color) >= 2 else 0
        oarr[:, :, 2] = color[0] if len(color) >= 1 else 0
        oarr[:, :, 3] = mask_alpha
        # Detach underlying buffer before Qt starts painting it
        return QtGui.QPixmap.fromImage(out_img.copy())

    def _combine_mask_pixmaps(self, seg_mask_pix, defect_mask_pix):
        # return a single ARGB pixmap combining seg (green) and defect (red) masks
        if seg_mask_pix is None and defect_mask_pix is None:
            return None
        # prefer seg size if available
        ref = seg_mask_pix if isinstance(seg_mask_pix, QtGui.QPixmap) else defect_mask_pix
        base_size = ref.size()
        seg_t = None
        defect_t = None
        if isinstance(seg_mask_pix, QtGui.QPixmap):
            seg_t = self._tint_mask_pixmap(seg_mask_pix.scaled(base_size), color=(0, 255, 0), alpha_val=160)
        if isinstance(defect_mask_pix, QtGui.QPixmap):
            defect_t = self._tint_mask_pixmap(defect_mask_pix.scaled(base_size), color=(255, 0, 0), alpha_val=200)
        result = QtGui.QPixmap(base_size)
        result.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(result)
        if seg_t:
            p.drawPixmap(0, 0, seg_t)
        if defect_t:
            p.drawPixmap(0, 0, defect_t)
        p.end()
        return result

    def update_selected_overlay(self, row: int = None):
        # Set `self.img_widget.selected_mask_pixmap` based on `self.overlay_mode` and the given/selected row
        if row is None:
            row = self.img_widget.selected_cell_index
        if row is None or row < 0 or row >= self.thumb_list.count():
            self.img_widget.selected_mask_pixmap = None
            self.img_widget.update()
            return
        item = self.thumb_list.item(row)
        seg_pm = item.data(ROLE_BASE + 1)
        defect_pm = item.data(ROLE_BASE + 2)
        mode = str(self.overlay_mode.currentText()) if hasattr(self, 'overlay_mode') else 'Segmentation'
        selected_pm = None
        if mode == 'None':
            selected_pm = None
        elif mode == 'Segmentation':
            selected_pm = self._tint_mask_pixmap(seg_pm, color=(0, 255, 0), alpha_val=200) if isinstance(seg_pm, QtGui.QPixmap) else None
        elif mode == 'Defect':
            selected_pm = self._tint_mask_pixmap(defect_pm, color=(255, 0, 0), alpha_val=220) if isinstance(defect_pm, QtGui.QPixmap) else None
        else:  # Both
            selected_pm = self._combine_mask_pixmaps(seg_pm if isinstance(seg_pm, QtGui.QPixmap) else None,
                                                     defect_pm if isinstance(defect_pm, QtGui.QPixmap) else None)
        self.img_widget.selected_mask_pixmap = selected_pm
        # also compute and display erosion outline for segmentation mask if requested
        self.update_erosion_outline(row)
        self.img_widget.update()

    def update_erosion_outline(self, row: int = None):
        # compute erosion outline for the segmentation mask of `row` and store as image-space QPainterPath
        if row is None:
            row = self.img_widget.selected_cell_index
        if row is None or row < 0 or row >= len(self.img_widget.grid_rects):
            self.img_widget.erosion_path = None
            self.img_widget.update()
            return
        item = self.thumb_list.item(row)
        seg_pm = item.data(ROLE_BASE + 1)
        if not isinstance(seg_pm, QtGui.QPixmap):
            # fallback: draw inset rectangle from base unit by erode_px so user sees effect
            erode_px = int(self.defect_mask_erode.value()) if hasattr(self, 'defect_mask_erode') else 0
            r, idx = self.img_widget.grid_rects[row]
            ux, uy, uw, uh = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            inx = ux + erode_px; iny = uy + erode_px; inw = max(0, uw - 2 * erode_px); inh = max(0, uh - 2 * erode_px)
            if inw <= 0 or inh <= 0:
                self.img_widget.erosion_path = None
                self.img_widget.update()
                return
            path = QtGui.QPainterPath()
            path.addRect(QtCore.QRectF(inx, iny, inw, inh))
            self.img_widget.erosion_path = path
            self.img_widget.update()
            return
        # convert segmentation pixmap to binary ROI exactly as stored (match Segmentation overlay)
        qimg = seg_pm.toImage()
        seg_arr = segmentation.qimage_to_gray_array(qimg)
        seg_bin = (seg_arr > 0).astype(np.uint8) * 255
        erode_px = int(self.defect_mask_erode.value()) if hasattr(self, 'defect_mask_erode') else 0
        try:
            seg_area0 = int((seg_bin > 0).sum())
        except Exception:
            seg_area0 = 0
        # avoid spamming the log on every slider move; uncomment if you need debug output
        # self.log(f'Erosion outline roi_area={seg_area0}, erode_px={erode_px}')
        # erode by user parameter (in pixels)
        if erode_px > 0:
            try:
                seg_bin = cv2.erode(seg_bin, None, iterations=erode_px)
            except Exception:
                pass
        # find contours on eroded mask (unit-local coords) and keep only the largest
        cnts, _ = cv2.findContours(seg_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self.img_widget.erosion_path = None
            self.img_widget.update()
            return
        try:
            cnts = [max(cnts, key=cv2.contourArea)]
        except Exception:
            pass
        # build QPainterPath in IMAGE coordinates by mapping unit-local points to absolute image coords
        path = QtGui.QPainterPath()
        # unit top-left in image coords
        r, idx = self.img_widget.grid_rects[row]
        ux, uy = int(r[0]), int(r[1])
        for ci, c in enumerate(cnts):
            try:
                pts = c.reshape(-1, 2)
            except Exception:
                continue
            if pts.size == 0:
                continue
            # first point
            p0x = int(ux + int(pts[0][0]))
            p0y = int(uy + int(pts[0][1]))
            path.moveTo(p0x, p0y)
            for pi in range(1, pts.shape[0]):
                px = int(ux + int(pts[pi][0]))
                py = int(uy + int(pts[pi][1]))
                path.lineTo(px, py)
            path.closeSubpath()
        self.img_widget.erosion_path = path
        self.img_widget.update()

    def img_widget_zoom(self, factor: float):
        # apply zoom multiplier to ImageWidget
        self.img_widget.manual_zoom *= factor
        self.img_widget.updateScale()
        # keep erosion outline aligned across zoom levels
        self.update_erosion_outline(self.img_widget.selected_cell_index)
        self.img_widget.update()

    def ensure_fit_view(self):
        # reset manual zoom and fit the image to viewport, reset scrollbars to origin
        self.img_widget.manual_zoom = 1.0
        self.img_widget.updateScale()
        # keep erosion outline aligned after fit-to-view
        self.update_erosion_outline(self.img_widget.selected_cell_index)
        self.img_widget.update()
        QtWidgets.QApplication.processEvents()
        # reset scrollbars to show top-left (fit)
        if hasattr(self, 'scroll'):
            hbar = self.scroll.horizontalScrollBar()
            vbar = self.scroll.verticalScrollBar()
            hbar.setValue(0)
            vbar.setValue(0)
        # reposition buttons via eventFilter
        QtWidgets.QApplication.processEvents()

    def eventFilter(self, source, event):
        # reposition zoom buttons when scroll viewport resizes
        if event.type() == QtCore.QEvent.Type.Resize and source == getattr(self, 'scroll', None).viewport():
            vp = source
            w = vp.width(); h = vp.height()
            margin = 12
            bx = w - self.zoom_in_btn.width() - margin
            by = h - self.zoom_in_btn.height() - margin
            self.zoom_in_btn.move(bx, by)
            self.zoom_out_btn.move(bx - self.zoom_out_btn.width() - 6, by)
            # ensure fit button at top-right
            fit_x = w - self.ensure_fit_btn.width() - margin
            fit_y = margin
            self.ensure_fit_btn.move(fit_x, fit_y)
        return super().eventFilter(source, event)

    def log(self, text: str):
        # append text to the read-only log output if present
        try:
            if hasattr(self, 'log_output') and self.log_output is not None:
                self.log_output.appendPlainText(str(text))
        except Exception:
            pass

    def _segmask_to_object_binary(self, seg_arr):
        # Normalize a segmentation mask array to a single-object binary mask (0/255 uint8).
        # This handles masks that might be inverted or contain background as the largest component.
        try:
            bw = (seg_arr > 0).astype(np.uint8) * 255
            h_m, w_m = bw.shape
            area_total = h_m * w_m
            cnts, _ = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return np.zeros_like(bw)
            # find largest contour and its area
            largest = max(cnts, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest)
            # if the largest contour covers most of the crop, it's likely background => invert
            if largest_area >= 0.5 * area_total:
                # invert mask and find largest object in inverted space
                inv = ((bw == 0).astype(np.uint8) * 255)
                cnts2, _ = cv2.findContours(inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts2:
                    best = max(cnts2, key=cv2.contourArea)
                    mask = np.zeros_like(bw)
                    cv2.drawContours(mask, [best], -1, 255, -1)
                    return mask
                else:
                    # nothing found in inverted mask; fall back to bw
                    return bw
            else:
                # largest is likely the object; return its filled contour
                mask = np.zeros_like(bw)
                cv2.drawContours(mask, [largest], -1, 255, -1)
                return mask
        except Exception:
            return (seg_arr > 0).astype(np.uint8) * 255

    def export_masks_and_csv(self):
        if not self.img_widget.grid_rects:
            QtWidgets.QMessageBox.information(self, 'Info', 'No grid available. Create indexing first.')
            return
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder to save masks', '.')
        if not dirpath:
            return
        csv_rows = []
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            pm_mask = item.data(ROLE_BASE + 1)
            if not isinstance(pm_mask, QtGui.QPixmap):
                continue
            fname = f'mask_{i:04d}.png'
            full = os.path.join(dirpath, fname)
            pm_mask.save(full)
            # compute stats from saved mask using cv2
            img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
            stats = segmentation.mask_stats(img)
            csv_rows.append({'index': i, 'mask': fname, 'area': stats['area'], 'centroid_x': stats['centroid'][0], 'centroid_y': stats['centroid'][1]})
        # write CSV
        csv_path = os.path.join(dirpath, 'masks_summary.csv')
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=['index', 'mask', 'area', 'centroid_x', 'centroid_y'])
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        QtWidgets.QMessageBox.information(self, 'Saved', f'Exported {len(csv_rows)} masks + summary to {dirpath}')

    def export_grid(self):
        if not self.img_widget.grid_rects:
            QtWidgets.QMessageBox.information(self, 'Info', 'No grid to export. Apply indexing first.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save grid JSON', 'grid.json', 'JSON (*.json)')
        if not path:
            return
        boxes = []
        for r, idx in self.img_widget.grid_rects:
            # r is tuple (x,y,w,h)
            boxes.append({'index': idx, 'x': int(r[0]), 'y': int(r[1]), 'w': int(r[2]), 'h': int(r[3])})
        # metadata to allow deterministic import later
        meta = {
            'image_width': self.img_widget.image.width() if self.img_widget.image else None,
            'image_height': self.img_widget.image.height() if self.img_widget.image else None,
            'units_x': self.units_x.value(),
            'units_y': self.units_y.value(),
            'blocks_x': self.blocks_x.value(),
            'blocks_y': self.blocks_y.value(),
            'unit_space_x': self.unit_space_x.value(),
            'unit_space_y': self.unit_space_y.value(),
            'block_space_x': self.block_space_x.value(),
            'block_space_y': self.block_space_y.value(),
        }
        # base unit
        if self.img_widget.fixed_img_rect:
            fir = self.img_widget.fixed_img_rect
            meta['base_unit'] = {'x': int(fir.x()), 'y': int(fir.y()), 'w': int(fir.width()), 'h': int(fir.height())}
        # include exclusions + alignment anchors (XY shift relative to segmentation centroid)
        ref_centroids = {}
        try:
            for k, v in (getattr(self, '_exclusion_ref_centroids', {}) or {}).items():
                try:
                    ref_centroids[str(int(k))] = {'cx': float(v[0]), 'cy': float(v[1])}
                except Exception:
                    continue
        except Exception:
            ref_centroids = {}

        exports = {
            'version': 2,
            'metadata': meta,
            'boxes': boxes,
            'exclusions': getattr(self, 'exclusions', []),
            'exclusion_alignment': {
                'type': 'seg_centroid_xy',
                'ref_centroids': ref_centroids,
            },
        }
        with open(path, 'w') as f:
            json.dump(exports, f, indent=2)
        QtWidgets.QMessageBox.information(self, 'Saved', f'Wrote {len(boxes)} boxes + metadata to {path}')

    def export_combined_json(self):
        if not self.img_widget.grid_rects:
            QtWidgets.QMessageBox.information(self, 'Info', 'No grid to export. Apply indexing first.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save combined JSON (with embedded masks)', 'grid_with_masks.json', 'JSON (*.json)')
        if not path:
            return
        boxes = []
        for r, idx in self.img_widget.grid_rects:
            boxes.append({'index': idx, 'x': int(r[0]), 'y': int(r[1]), 'w': int(r[2]), 'h': int(r[3])})
        meta = {
            'image_width': self.img_widget.image.width() if self.img_widget.image else None,
            'image_height': self.img_widget.image.height() if self.img_widget.image else None,
            'units_x': self.units_x.value(),
            'units_y': self.units_y.value(),
            'blocks_x': self.blocks_x.value(),
            'blocks_y': self.blocks_y.value(),
            'unit_space_x': self.unit_space_x.value(),
            'unit_space_y': self.unit_space_y.value(),
            'block_space_x': self.block_space_x.value(),
            'block_space_y': self.block_space_y.value(),
        }
        if self.img_widget.fixed_img_rect:
            fir = self.img_widget.fixed_img_rect
            meta['base_unit'] = {'x': int(fir.x()), 'y': int(fir.y()), 'w': int(fir.width()), 'h': int(fir.height())}
        masks_out = []
        # collect masks from thumbnails (UserRole+1)
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            pm_mask = item.data(ROLE_BASE + 1)
            if isinstance(pm_mask, QtGui.QPixmap):
                qim = pm_mask.toImage()
                buf = QtCore.QBuffer()
                buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
                qim.save(buf, 'PNG')
                raw = bytes(buf.data())
                b64 = base64.b64encode(raw).decode('ascii')
                masks_out.append({'index': i, 'mask_b64': b64})
        exports = {'metadata': meta, 'boxes': boxes, 'exclusions': getattr(self, 'exclusions', []), 'masks': masks_out}
        try:
            with open(path, 'w') as f:
                json.dump(exports, f)
            QtWidgets.QMessageBox.information(self, 'Saved', f'Wrote combined JSON with {len(masks_out)} embedded masks to {path}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to write JSON: {e}')

    def import_grid(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open grid JSON', '.', 'JSON (*.json)')
        if not path:
            return
        if not self.img_widget.image:
            QtWidgets.QMessageBox.information(self, 'Info', 'Load an image first before importing a grid.')
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to read JSON: {e}')
            return
        grid = []
        # support two formats: legacy list-of-boxes or new dict with 'boxes' and 'metadata'
        boxes = None
        if isinstance(data, dict) and 'boxes' in data:
            boxes = data['boxes']
            meta = data.get('metadata', {})
            # load exclusions if present
            try:
                self.exclusions = data.get('exclusions', []) or []
            except Exception:
                self.exclusions = []

            # load exclusion alignment anchors (optional)
            try:
                align = data.get('exclusion_alignment', {}) or {}
                if isinstance(align, dict) and align.get('type') == 'seg_centroid_xy':
                    refc = align.get('ref_centroids', {}) or {}
                    centroids = {}
                    if isinstance(refc, dict):
                        for k, vv in refc.items():
                            try:
                                kk = int(k)
                                cx = float(vv.get('cx'))
                                cy = float(vv.get('cy'))
                                centroids[kk] = (cx, cy)
                            except Exception:
                                continue
                    self._exclusion_ref_centroids = centroids
            except Exception:
                pass
        elif isinstance(data, list):
            boxes = data
            meta = {}
        else:
            boxes = []
            meta = {}

        for item in boxes:
            try:
                idx = item.get('index', None)
                x = int(item['x']); y = int(item['y']); w = int(item['w']); h = int(item['h'])
            except Exception:
                continue
            if idx is None:
                idx = len(grid)
            grid.append(((x, y, w, h), idx))
        if not grid:
            QtWidgets.QMessageBox.information(self, 'Info', 'No valid boxes found in JSON.')
            return
        self.img_widget.grid_rects = grid
        # if metadata available, prefer deterministic fill of UI
        try:
            # metadata may have been set earlier when reading dict
            if 'meta' in locals() and isinstance(meta, dict) and meta:
                ux = int(meta.get('units_x', 0))
                uy = int(meta.get('units_y', 0))
                bx = int(meta.get('blocks_x', 1))
                by = int(meta.get('blocks_y', 1))
                sux = int(meta.get('unit_space_x', 0))
                suy = int(meta.get('unit_space_y', 0))
                sbx = int(meta.get('block_space_x', 0))
                sby = int(meta.get('block_space_y', 0))
                if ux > 0: self.units_x.setValue(ux)
                if uy > 0: self.units_y.setValue(uy)
                self.blocks_x.setValue(max(1, bx))
                self.blocks_y.setValue(max(1, by))
                self.unit_space_x.setValue(sux)
                self.unit_space_y.setValue(suy)
                self.block_space_x.setValue(sbx)
                self.block_space_y.setValue(sby)
                # set base unit if present
                bu = meta.get('base_unit')
                if bu:
                    self.img_widget.fixed_img_rect = QtCore.QRect(int(bu.get('x', 0)), int(bu.get('y', 0)), int(bu.get('w', 1)), int(bu.get('h', 1)))
        except Exception:
            pass
        self.img_widget.update()
        self.populate_thumbnails()

        # update exclusion UI index range
        try:
            if getattr(self, 'exclusions', None):
                self.excl_index.setRange(0, max(0, len(self.exclusions) - 1))
                self.excl_index.setValue(min(int(self.excl_index.value()), len(self.exclusions) - 1))
            else:
                self.excl_index.setRange(0, 0)
                self.excl_index.setValue(0)
            self.on_exclusion_index_changed()
        except Exception:
            pass
        QtWidgets.QMessageBox.information(self, 'Imported', f'Imported {len(grid)} boxes from {path} — indexing fields updated from metadata when available')

    def import_mask(self):
        # Import a JSON that may contain metadata/boxes/exclusions and embedded masks (base64) or mask file references,
        # or select a folder next to a JSON that contains mask_XXXX.png files.
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open mask JSON (or a JSON next to mask files)', '.', 'JSON (*.json);;All Files (*)')
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to read JSON: {e}')
            return

        # if JSON contains boxes metadata, reuse import logic
        boxes = None
        meta = {}
        if isinstance(data, dict) and 'boxes' in data:
            boxes = data['boxes']
            meta = data.get('metadata', {})
            try:
                self.exclusions = data.get('exclusions', []) or []
            except Exception:
                self.exclusions = []
        elif isinstance(data, dict) and 'masks' in data and 'boxes' in data:
            boxes = data['boxes']
            meta = data.get('metadata', {})
            try:
                self.exclusions = data.get('exclusions', []) or []
            except Exception:
                self.exclusions = []
        else:
            # if JSON doesn't have boxes, try to find a nearby folder with mask images
            boxes = None

        if boxes:
            grid = []
            for item in boxes:
                try:
                    idx = item.get('index', None)
                    x = int(item['x']); y = int(item['y']); w = int(item['w']); h = int(item['h'])
                except Exception:
                    continue
                if idx is None:
                    idx = len(grid)
                grid.append(((x, y, w, h), idx))
            if not grid:
                QtWidgets.QMessageBox.information(self, 'Info', 'No valid boxes found in JSON.')
                return
            self.img_widget.grid_rects = grid
            # prefer metadata to fill UI
            try:
                if isinstance(meta, dict) and meta:
                    ux = int(meta.get('units_x', 0)); uy = int(meta.get('units_y', 0))
                    bx = int(meta.get('blocks_x', 1)); by = int(meta.get('blocks_y', 1))
                    sux = int(meta.get('unit_space_x', 0)); suy = int(meta.get('unit_space_y', 0))
                    sbx = int(meta.get('block_space_x', 0)); sby = int(meta.get('block_space_y', 0))
                    if ux > 0: self.units_x.setValue(ux)
                    if uy > 0: self.units_y.setValue(uy)
                    self.blocks_x.setValue(max(1, bx)); self.blocks_y.setValue(max(1, by))
                    self.unit_space_x.setValue(sux); self.unit_space_y.setValue(suy)
                    self.block_space_x.setValue(sbx); self.block_space_y.setValue(sby)
                    bu = meta.get('base_unit')
                    if bu:
                        self.img_widget.fixed_img_rect = QtCore.QRect(int(bu.get('x', 0)), int(bu.get('y', 0)), int(bu.get('w', 1)), int(bu.get('h', 1)))
            except Exception:
                pass
            self.populate_thumbnails()

            # load masks embedded in JSON (base64) or referenced files
            masks_list = data.get('masks', []) if isinstance(data, dict) else []
            json_dir = os.path.dirname(path)
            if masks_list:
                for m in masks_list:
                    try:
                        idx = int(m.get('index', -1))
                        if idx < 0 or idx >= self.thumb_list.count():
                            continue
                        pm_mask = None
                        if 'mask_b64' in m:
                            b = base64.b64decode(m['mask_b64'])
                            qim = QtGui.QImage.fromData(b)
                            if not qim.isNull():
                                pm_mask = QtGui.QPixmap.fromImage(qim)
                        elif 'mask_file' in m:
                            mf = m['mask_file']
                            if not os.path.isabs(mf):
                                mf = os.path.join(json_dir, mf)
                            if os.path.exists(mf):
                                pm_mask = QtGui.QPixmap(mf)
                        if pm_mask:
                            item = self.thumb_list.item(idx)
                            item.setData(ROLE_BASE + 1, pm_mask)
                            thumb_pm = item.data(ROLE_BASE)
                            if isinstance(thumb_pm, QtGui.QPixmap):
                                overlay = self._make_overlay_pixmap(
                                    thumb_pm,
                                    pm_mask.scaled(
                                        thumb_pm.size(),
                                        QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                        QtCore.Qt.TransformationMode.SmoothTransformation,
                                    ),
                                )
                                item.setIcon(QtGui.QIcon(overlay))
                    except Exception:
                        continue
            else:
                # also try reading mask_####.png files next to JSON
                for i in range(self.thumb_list.count()):
                    f = os.path.join(json_dir, f'mask_{i:04d}.png')
                    if os.path.exists(f):
                        pm = QtGui.QPixmap(f)
                        item = self.thumb_list.item(i)
                        item.setData(ROLE_BASE + 1, pm)
                        thumb_pm = item.data(ROLE_BASE)
                        if isinstance(thumb_pm, QtGui.QPixmap):
                            overlay = self._make_overlay_pixmap(
                                thumb_pm,
                                pm.scaled(
                                    thumb_pm.size(),
                                    QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                    QtCore.Qt.TransformationMode.SmoothTransformation,
                                ),
                            )
                            item.setIcon(QtGui.QIcon(overlay))

            # refresh selected overlay if needed
            if self.img_widget.selected_cell_index is not None:
                self.update_selected_overlay(self.img_widget.selected_cell_index)
            QtWidgets.QMessageBox.information(self, 'Imported', f'Imported grid + masks from {path}')
            return

        # Fallback: if JSON didn't contain boxes, try loading a folder of masks (user selects folder)
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder containing mask_XXXX.png files', os.path.dirname(path))
        if not dirpath:
            QtWidgets.QMessageBox.information(self, 'Info', 'No boxes or masks found in JSON and no folder selected.')
            return
        # try to load mask files into existing thumbnails
        loaded = 0
        for i in range(self.thumb_list.count()):
            f = os.path.join(dirpath, f'mask_{i:04d}.png')
            if os.path.exists(f):
                pm = QtGui.QPixmap(f)
                item = self.thumb_list.item(i)
                item.setData(ROLE_BASE + 1, pm)
                thumb_pm = item.data(ROLE_BASE)
                if isinstance(thumb_pm, QtGui.QPixmap):
                    overlay = self._make_overlay_pixmap(
                        thumb_pm,
                        pm.scaled(
                            thumb_pm.size(),
                            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation,
                        ),
                    )
                    item.setIcon(QtGui.QIcon(overlay))
                loaded += 1
        QtWidgets.QMessageBox.information(self, 'Imported', f'Loaded {loaded} masks from {dirpath}')
        if self.img_widget.selected_cell_index is not None:
            self.update_selected_overlay(self.img_widget.selected_cell_index)

    def populate_thumbnails(self):
        self.thumb_list.clear()
        if not self.img_widget.grid_rects or not self.img_widget.image:
            return
        base = QtGui.QPixmap.fromImage(self.img_widget.image)
        for r, idx in self.img_widget.grid_rects:
            # r is (x,y,w,h)
            sub = base.copy(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            icon = QtGui.QIcon(
                sub.scaled(
                    128,
                    128,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
            item = QtWidgets.QListWidgetItem(icon, str(idx))
            # store pixmap for export
            item.setData(ROLE_BASE, sub)
            self.thumb_list.addItem(item)
        # update defect unit spin range if present
        if hasattr(self, 'defect_unit_spin'):
            n = max(0, self.thumb_list.count() - 1)
            self.defect_unit_spin.setRange(0, n)
            # keep a valid default (0) when units exist
            if self.thumb_list.count() > 0:
                try:
                    self.defect_unit_spin.setValue(min(int(self.defect_unit_spin.value()), n))
                except Exception:
                    self.defect_unit_spin.setValue(0)

    def export_thumbnails(self):
        if self.thumb_list.count() == 0:
            QtWidgets.QMessageBox.information(self, 'Info', 'No thumbnails to export. Apply indexing first.')
            return
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder to save thumbnails', '.')
        if not dirpath:
            return
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            pm = item.data(ROLE_BASE)
            if isinstance(pm, QtGui.QPixmap):
                fname = f"unit_{i:04d}.png"
                pm.save(QtCore.QDir.cleanPath(QtCore.QDir(dirpath).filePath(fname)))
        QtWidgets.QMessageBox.information(self, 'Saved', f'Exported {self.thumb_list.count()} thumbnails to {dirpath}')


class ModifyExclusionDialog(QtWidgets.QDialog):
    def __init__(self, main: MainWindow):
        super().__init__(main)
        self.setWindowTitle('Modify exclusion')
        self.setModal(True)
        self._main = main
        self._loading = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel('Exclusion #'))
        self.idx = SpinBox(); self.idx.setRange(0, max(0, len(main.exclusions) - 1))
        try:
            self.idx.setValue(int(main.excl_index.value()))
        except Exception:
            self.idx.setValue(0)
        top.addWidget(self.idx)
        top.addStretch(1)
        v.addLayout(top)

        self.shape_lbl = QtWidgets.QLabel('')
        v.addWidget(self.shape_lbl)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        # position
        self.pos_x = SpinBox(); self.pos_x.setRange(0, 20000)
        self.pos_y = SpinBox(); self.pos_y.setRange(0, 20000)
        # size
        self.size_w = SpinBox(); self.size_w.setRange(1, 20000)
        self.size_l = SpinBox(); self.size_l.setRange(1, 20000)
        self.radius = SpinBox(); self.radius.setRange(1, 20000)

        form.addRow('X (px):', self.pos_x)
        form.addRow('Y (px):', self.pos_y)
        form.addRow('W (px):', self.size_w)
        form.addRow('L (px):', self.size_l)
        form.addRow('R (px):', self.radius)
        v.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.edit_toggle = ToggleButton('Edit on image')
        try:
            self.edit_toggle.setCheckable(True)
        except Exception:
            pass
        self.delete_btn = PushButton('Delete')
        self.close_btn = PrimaryPushButton('Close')
        btns.addWidget(self.edit_toggle)
        btns.addStretch(1)
        btns.addWidget(self.delete_btn)
        btns.addWidget(self.close_btn)
        v.addLayout(btns)

        self.close_btn.clicked.connect(self.accept)
        self.delete_btn.clicked.connect(self._on_delete)
        self.idx.valueChanged.connect(self._on_index_changed)
        self.edit_toggle.toggled.connect(self._on_edit_toggled)

        for w in (self.pos_x, self.pos_y, self.size_w, self.size_l, self.radius):
            w.valueChanged.connect(self._on_fields_changed)

        self.reload_from_main()

    def _base_unit_size(self):
        b = self._main._get_base_unit_rect()
        if b is None:
            return None
        _, _, bw, bh = b
        return int(bw), int(bh)

    def _selected_idx(self):
        try:
            i = int(self.idx.value())
        except Exception:
            return None
        if i < 0 or i >= len(self._main.exclusions):
            return None
        return i

    def reload_from_main(self):
        self._loading = True
        try:
            self.idx.setRange(0, max(0, len(self._main.exclusions) - 1))
            cur = self._selected_idx()
            if cur is None:
                self.shape_lbl.setText('')
                return

            excl = self._main.exclusions[cur]
            shape = excl.get('shape')
            if shape == 'rect':
                self.shape_lbl.setText('Shape: rectangle (top-left x/y)')
            else:
                self.shape_lbl.setText('Shape: circle (center x/y)')

            bw_bh = self._base_unit_size()
            if bw_bh is not None:
                bw, bh = bw_bh
                self.pos_x.setRange(0, max(0, bw - 1))
                self.pos_y.setRange(0, max(0, bh - 1))
                self.size_w.setRange(1, max(1, bw))
                self.size_l.setRange(1, max(1, bh))
                self.radius.setRange(1, max(1, min(bw, bh)))

            if shape == 'rect':
                self.pos_x.setValue(int(excl.get('x', 0)))
                self.pos_y.setValue(int(excl.get('y', 0)))
                self.size_w.setValue(int(excl.get('w', 1)))
                self.size_l.setValue(int(excl.get('h', 1)))
                self.radius.setValue(1)
                self.size_w.setEnabled(True)
                self.size_l.setEnabled(True)
                self.radius.setEnabled(False)
            else:
                self.pos_x.setValue(int(excl.get('cx', 0)))
                self.pos_y.setValue(int(excl.get('cy', 0)))
                self.radius.setValue(int(excl.get('r', 1)))
                self.size_w.setValue(1)
                self.size_l.setValue(1)
                self.size_w.setEnabled(False)
                self.size_l.setEnabled(False)
                self.radius.setEnabled(True)

            # edit toggle reflects main
            try:
                self.edit_toggle.blockSignals(True)
                self.edit_toggle.setChecked(bool(self._main._exclusion_edit_active))
            finally:
                try:
                    self.edit_toggle.blockSignals(False)
                except Exception:
                    pass

        finally:
            self._loading = False

    def sync_from_main(self):
        if self._loading:
            return
        self.reload_from_main()

    def _on_index_changed(self, *_):
        if self._loading:
            return
        try:
            self._main.excl_index.setValue(int(self.idx.value()))
        except Exception:
            pass
        self.reload_from_main()
        if getattr(self._main, '_exclusion_edit_active', False):
            try:
                self._main._refresh_exclusion_edit_overlay(int(self.idx.value()))
            except Exception:
                pass

    def _on_edit_toggled(self, checked: bool):
        if self._loading:
            return
        try:
            self._main.excl_index.setValue(int(self.idx.value()))
        except Exception:
            pass
        if checked and not getattr(self._main, '_exclusion_edit_active', False):
            self._main.toggle_edit_exclusion()
        if (not checked) and getattr(self._main, '_exclusion_edit_active', False):
            self._main.toggle_edit_exclusion()
        self.reload_from_main()

    def _on_delete(self):
        i = self._selected_idx()
        if i is None:
            return
        self._main.delete_exclusion_at(i)
        self.reload_from_main()

    def _on_fields_changed(self, *_):
        if self._loading:
            return
        i = self._selected_idx()
        if i is None:
            return

        excl = self._main.exclusions[i]
        shape = excl.get('shape')
        if shape == 'rect':
            excl['x'] = int(self.pos_x.value())
            excl['y'] = int(self.pos_y.value())
            excl['w'] = int(self.size_w.value())
            excl['h'] = int(self.size_l.value())
        else:
            excl['cx'] = int(self.pos_x.value())
            excl['cy'] = int(self.pos_y.value())
            excl['r'] = int(self.radius.value())

        self._main._clamp_exclusion_to_base_unit(excl)
        self.reload_from_main()

        if getattr(self._main, '_exclusion_edit_active', False):
            try:
                self._main._refresh_exclusion_edit_overlay(i)
            except Exception:
                pass

        try:
            self._main._exclusion_edit_timer.start()
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.showFullScreen()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
