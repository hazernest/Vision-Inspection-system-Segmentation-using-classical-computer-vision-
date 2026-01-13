import sys
import json
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import base64
import csv
import segmentation
import cv2
import numpy as np


class ImageWidget(QtWidgets.QWidget):
    selectionChanged = QtCore.pyqtSignal()
    cellClicked = QtCore.pyqtSignal(int)
    exclusionDrawn = QtCore.pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent)
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
        # selected cell index and mask pixmap (in image coordinates)
        self.selected_cell_index = None
        self.selected_mask_pixmap = None
        # optional QPainterPath outlining eroded mask (in display coordinates)
        self.erosion_path = None
        # per-cell overlays to draw on the main canvas: {grid_idx: {'seg': QPixmap|None, 'defect': QPixmap|None}}
        self.cell_overlays = {}
        # current overlay mode for full-canvas drawing
        self.overlay_mode = 'Defect'

    def load_image(self, path):
        img = QtGui.QImage(path)
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
        painter.fillRect(self.rect(), QtCore.Qt.black)
        if self.image:
            # draw scaled image at widget origin
            disp = self.image.scaled(int(self.image.width() * self.scale), int(self.image.height() * self.scale), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
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
                        painter.drawPixmap(dr.topLeft(), seg_pm.scaled(dr.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation))
                if mode in ('Defect', 'Both'):
                    defect_pm = ov.get('defect')
                    if isinstance(defect_pm, QtGui.QPixmap):
                        painter.drawPixmap(dr.topLeft(), defect_pm.scaled(dr.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation))
            painter.setOpacity(1.0)
        # draw selected mask overlay if available
        if self.selected_cell_index is not None and self.selected_mask_pixmap:
            # find rect for selected cell
            for r, idx in self.grid_rects:
                if idx == self.selected_cell_index:
                    img_r = QtCore.QRect(r[0], r[1], r[2], r[3])
                    dr = self.imgrect_to_display(img_r)
                    # mask pixmap is in image coords with same size as img_r
                    mask_scaled = self.selected_mask_pixmap.scaled(dr.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                    painter.setOpacity(0.6)
                    painter.drawPixmap(dr.topLeft(), mask_scaled)
                    painter.setOpacity(1.0)
                    break
                # draw erosion outline if present
                if self.erosion_path is not None:
                    pen = QtGui.QPen(QtGui.QColor(0, 255, 255), 2)
                    pen.setStyle(QtCore.Qt.SolidLine)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawPath(self.erosion_path)
    def mousePressEvent(self, event):
        if not self.image:
            return
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
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
        if self.start_img_pos is not None:
            img_pt = self.display_to_img(event.pos())
            self.current_img_rect = QtCore.QRect(self.start_img_pos, img_pt).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.current_img_rect:
            if self.exclusion_mode:
                # emit exclusion rect in image coordinates
                excl = self.current_img_rect.normalized()
                self.current_exclusion_rect = excl
                # stop exclusion mode
                self.exclusion_mode = False
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

        # right: controls
        ctrl = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(ctrl)

        load_btn = QtWidgets.QPushButton('Load Image')
        load_btn.clicked.connect(self.load_image)
        v.addWidget(load_btn)

        # Validation note
        note = QtWidgets.QLabel('Expect image 4096x3000 (or it will still work).')
        v.addWidget(note)

        self.units_x = QtWidgets.QSpinBox(); self.units_x.setRange(0, 100); self.units_x.setValue(0)
        self.units_y = QtWidgets.QSpinBox(); self.units_y.setRange(0, 100); self.units_y.setValue(0)
        self.blocks_x = QtWidgets.QSpinBox(); self.blocks_x.setRange(0, 50); self.blocks_x.setValue(0)
        self.blocks_y = QtWidgets.QSpinBox(); self.blocks_y.setRange(0, 50); self.blocks_y.setValue(0)

        form = QtWidgets.QFormLayout()
        form.addRow('Units X:', self.units_x)
        form.addRow('Units Y:', self.units_y)
        form.addRow('Blocks X:', self.blocks_x)
        form.addRow('Blocks Y:', self.blocks_y)
        v.addLayout(form)

        # spacings: sliders + spinboxes for X/Y for units and blocks
        self.unit_space_x = QtWidgets.QSpinBox(); self.unit_space_x.setRange(0, 1000); self.unit_space_x.setValue(0)
        self.unit_space_y = QtWidgets.QSpinBox(); self.unit_space_y.setRange(0, 1000); self.unit_space_y.setValue(0)
        self.block_space_x = QtWidgets.QSpinBox(); self.block_space_x.setRange(0, 2000); self.block_space_x.setValue(0)
        self.block_space_y = QtWidgets.QSpinBox(); self.block_space_y.setRange(0, 2000); self.block_space_y.setValue(0)

        self.unit_space_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.unit_space_x_slider.setRange(0, 1000); self.unit_space_x_slider.setValue(0)
        self.unit_space_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.unit_space_y_slider.setRange(0, 1000); self.unit_space_y_slider.setValue(0)
        self.block_space_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.block_space_x_slider.setRange(0, 2000); self.block_space_x_slider.setValue(0)
        self.block_space_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.block_space_y_slider.setRange(0, 2000); self.block_space_y_slider.setValue(0)

        v.addWidget(QtWidgets.QLabel('Unit spacing X (px):'))
        ux_layout = QtWidgets.QHBoxLayout(); ux_layout.addWidget(self.unit_space_x_slider); ux_layout.addWidget(self.unit_space_x); v.addLayout(ux_layout)
        v.addWidget(QtWidgets.QLabel('Unit spacing Y (px):'))
        uy_layout = QtWidgets.QHBoxLayout(); uy_layout.addWidget(self.unit_space_y_slider); uy_layout.addWidget(self.unit_space_y); v.addLayout(uy_layout)
        v.addWidget(QtWidgets.QLabel('Block spacing X (px):'))
        bx_layout = QtWidgets.QHBoxLayout(); bx_layout.addWidget(self.block_space_x_slider); bx_layout.addWidget(self.block_space_x); v.addLayout(bx_layout)
        v.addWidget(QtWidgets.QLabel('Block spacing Y (px):'))
        by_layout = QtWidgets.QHBoxLayout(); by_layout.addWidget(self.block_space_y_slider); by_layout.addWidget(self.block_space_y); v.addLayout(by_layout)

        # wire sliders and spinboxes together
        self.unit_space_x_slider.valueChanged.connect(self.unit_space_x.setValue)
        self.unit_space_y_slider.valueChanged.connect(self.unit_space_y.setValue)
        self.block_space_x_slider.valueChanged.connect(self.block_space_x.setValue)
        self.block_space_y_slider.valueChanged.connect(self.block_space_y.setValue)
        self.unit_space_x.valueChanged.connect(self.unit_space_x_slider.setValue)
        self.unit_space_y.valueChanged.connect(self.unit_space_y_slider.setValue)
        self.block_space_x.valueChanged.connect(self.block_space_x_slider.setValue)
        self.block_space_y.valueChanged.connect(self.block_space_y_slider.setValue)

        self.apply_btn = QtWidgets.QPushButton('Apply Indexing')
        self.apply_btn.clicked.connect(self.apply_indexing)
        v.addWidget(self.apply_btn)
        # add unlock/edit button to re-enable drawing
        self.edit_btn = QtWidgets.QPushButton('Unlock Editing')
        self.edit_btn.setCheckable(True)
        self.edit_btn.toggled.connect(self.toggle_editing)
        v.addWidget(self.edit_btn)

        export_btn = QtWidgets.QPushButton('Export grid JSON')
        export_btn.clicked.connect(self.export_grid)
        v.addWidget(export_btn)
        export_embed_btn = QtWidgets.QPushButton('Export combined JSON (embed masks)')
        export_embed_btn.clicked.connect(self.export_combined_json)
        v.addWidget(export_embed_btn)
        import_btn = QtWidgets.QPushButton('Import grid JSON')
        import_btn.clicked.connect(self.import_grid)
        v.addWidget(import_btn)
        import_mask_btn = QtWidgets.QPushButton('Import Mask JSON/Folder')
        import_mask_btn.clicked.connect(self.import_mask)
        v.addWidget(import_mask_btn)

        # thumbnails list and export
        self.thumb_list = QtWidgets.QListWidget()
        self.thumb_list.setViewMode(QtWidgets.QListView.IconMode)
        self.thumb_list.setIconSize(QtCore.QSize(128, 128))
        self.thumb_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.thumb_list.setMovement(QtWidgets.QListView.Static)
        v.addWidget(QtWidgets.QLabel('Unit Thumbnails:'))
        v.addWidget(self.thumb_list)
        export_thumbs_btn = QtWidgets.QPushButton('Export Thumbnails')
        export_thumbs_btn.clicked.connect(self.export_thumbnails)
        v.addWidget(export_thumbs_btn)

        # Exclusions: add/exclusion index and shape
        excl_box = QtWidgets.QHBoxLayout()
        self.excl_index = QtWidgets.QSpinBox(); self.excl_index.setRange(0, 0); self.excl_index.setValue(0)
        self.excl_shape = QtWidgets.QComboBox(); self.excl_shape.addItems(['rectangle', 'circle'])
        self.add_excl_btn = QtWidgets.QPushButton('Add exclusion')
        self.add_excl_btn.clicked.connect(self.add_exclusion)
        excl_box.addWidget(QtWidgets.QLabel('Exclusion #'))
        excl_box.addWidget(self.excl_index)
        excl_box.addWidget(self.excl_shape)
        excl_box.addWidget(self.add_excl_btn)
        v.addLayout(excl_box)

        self.exclusions = []

        # segmentation controls
        v.addWidget(QtWidgets.QLabel('Segmentation Method:'))
        self.seg_method = QtWidgets.QComboBox()
        self.seg_method.addItems(['otsu', 'adaptive'])
        v.addWidget(self.seg_method)
        self.gauss_spin = QtWidgets.QSpinBox(); self.gauss_spin.setRange(0, 31); self.gauss_spin.setValue(3)
        self.morph_spin = QtWidgets.QSpinBox(); self.morph_spin.setRange(0, 31); self.morph_spin.setValue(3)
        self.adapt_block = QtWidgets.QSpinBox(); self.adapt_block.setRange(3, 201); self.adapt_block.setValue(51)
        self.adapt_C = QtWidgets.QSpinBox(); self.adapt_C.setRange(-50, 50); self.adapt_C.setValue(10)
        form2 = QtWidgets.QFormLayout()
        form2.addRow('Gaussian blur kernel:', self.gauss_spin)
        form2.addRow('Morph kernel size:', self.morph_spin)
        form2.addRow('Adaptive block size:', self.adapt_block)
        form2.addRow('Adaptive C:', self.adapt_C)
        v.addLayout(form2)
        run_seg_btn = QtWidgets.QPushButton('Run Segmentation')
        run_seg_btn.clicked.connect(self.run_segmentation_all)
        v.addWidget(run_seg_btn)
        export_masks_btn = QtWidgets.QPushButton('Export Masks + CSV')
        export_masks_btn.clicked.connect(self.export_masks_and_csv)
        v.addWidget(export_masks_btn)

        v.addStretch(1)
        # put controls into a tab widget: Main and Defect tabs
        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.addTab(ctrl, 'Main')

        # Defect tab with nested sub-tabs for different defect types
        defect_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(defect_tab)
        dv.setContentsMargins(6, 6, 6, 6)
        defect_subtabs = QtWidgets.QTabWidget()

        # Example defect subtab: Particle detection
        particle_tab = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(particle_tab)
        pv.addWidget(QtWidgets.QLabel('Foreign material detection settings'))
        self.defect_method = QtWidgets.QComboBox()
        self.defect_method.addItems(['threshold', 'canny'])
        pv.addWidget(QtWidgets.QLabel('Method:'))
        pv.addWidget(self.defect_method)
        self.defect_threshold = QtWidgets.QSpinBox(); self.defect_threshold.setRange(0, 255); self.defect_threshold.setValue(128)
        pv.addWidget(QtWidgets.QLabel('Threshold:'))
        pv.addWidget(self.defect_threshold)
        self.defect_min_area = QtWidgets.QSpinBox(); self.defect_min_area.setRange(0, 100000); self.defect_min_area.setValue(20)
        pv.addWidget(QtWidgets.QLabel('Min area (px):'))
        pv.addWidget(self.defect_min_area)
        # mask erosion: shrink segmentation mask by this many pixels before detection
        self.defect_mask_erode = QtWidgets.QSpinBox(); self.defect_mask_erode.setRange(0, 200); self.defect_mask_erode.setValue(0)
        pv.addWidget(QtWidgets.QLabel('Mask erosion (px):'))
        pv.addWidget(self.defect_mask_erode)
        # overlay display mode
        pv.addWidget(QtWidgets.QLabel('Overlay mode:'))
        self.overlay_mode = QtWidgets.QComboBox()
        self.overlay_mode.addItems(['None', 'Segmentation', 'Defect', 'Both'])
        self.overlay_mode.setCurrentIndex(2)
        pv.addWidget(self.overlay_mode)
        self.overlay_mode.currentIndexChanged.connect(self.on_overlay_mode_changed)
        # recompute erosion outline when mask-erode value changes
        if hasattr(self, 'defect_mask_erode'):
            self.defect_mask_erode.valueChanged.connect(lambda _: self.update_erosion_outline(self.img_widget.selected_cell_index))
        # unit index selector for testing
        self.defect_unit_spin = QtWidgets.QSpinBox(); self.defect_unit_spin.setRange(0, 0); self.defect_unit_spin.setValue(0)
        pv.addWidget(QtWidgets.QLabel('Unit index to test:'))
        pv.addWidget(self.defect_unit_spin)
        # test buttons
        test_btn = QtWidgets.QPushButton('Test on unit')
        test_btn.clicked.connect(self.test_defect_detection)
        pv.addWidget(test_btn)
        test_all_btn = QtWidgets.QPushButton('Test All Units')
        test_all_btn.clicked.connect(self.test_defect_detection_all)
        pv.addWidget(test_all_btn)
        pv.addStretch(1)
        defect_subtabs.addTab(particle_tab, 'Foreign material')

        # Placeholder for additional defect types
        crack_tab = QtWidgets.QWidget()
        cvlay = QtWidgets.QVBoxLayout(crack_tab)
        cvlay.addWidget(QtWidgets.QLabel('Crack detection (placeholder)'))
        cvlay.addStretch(1)
        defect_subtabs.addTab(crack_tab, 'Crack')

        dv.addWidget(defect_subtabs)
        dv.addStretch(1)
        self.right_tabs.addTab(defect_tab, 'Defect')

        # create a right-side container with tabs and a log terminal underneath
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.right_tabs)
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
        self.thumb_list.currentRowChanged.connect(self.on_thumbnail_selected)
        self.img_widget.exclusionDrawn.connect(self.on_exclusion_drawn)

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

    def on_cell_clicked(self, idx):
        # select thumbnail and show mask overlay for this cell
        if idx < self.thumb_list.count():
            self.thumb_list.setCurrentRow(idx)
            self.img_widget.selected_cell_index = idx
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
        self.update_selected_overlay(row)

        # center and zoom to selected cell and move zoom buttons near it
        if row >= 0 and row < len(self.img_widget.grid_rects):
            self.center_on_cell(row)

    def test_defect_detection(self):
        # run defect detection on the unit specified by `defect_unit_spin`
        row = int(self.defect_unit_spin.value()) if hasattr(self, 'defect_unit_spin') else self.thumb_list.currentRow()
        if row < 0 or row >= self.thumb_list.count():
            QtWidgets.QMessageBox.information(self, 'Info', 'Select a valid unit index first.')
            return
        item = self.thumb_list.item(row)
        pix = item.data(QtCore.Qt.UserRole)
        if not isinstance(pix, QtGui.QPixmap):
            QtWidgets.QMessageBox.information(self, 'Info', 'No thumbnail image available for this unit.')
            return
        seg_mask_pm = item.data(QtCore.Qt.UserRole + 1)
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
        item.setData(QtCore.Qt.UserRole + 2, pm_mask if isinstance(pm_mask, QtGui.QPixmap) else None)
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

    def _detect_defects_on_pix(self, pix: QtGui.QPixmap, seg_mask_pix: QtGui.QPixmap = None):
        # returns a QPixmap mask (grayscale) highlighting defects, or None
        qimg = pix.toImage()
        gray = segmentation.qimage_to_gray_array(qimg)
        seg_bin = None
        # if segmentation mask provided, scale and apply it to restrict detection area
        if isinstance(seg_mask_pix, QtGui.QPixmap):
            seg_qimg = seg_mask_pix.toImage().scaled(qimg.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            seg_arr = segmentation.qimage_to_gray_array(seg_qimg)
            erode_px = int(self.defect_mask_erode.value()) if hasattr(self, 'defect_mask_erode') else 0
            # Use the segmentation mask exactly as the ROI (match what the Segmentation overlay shows)
            seg_bin = (seg_arr > 0).astype(np.uint8) * 255
            try:
                seg_area0 = int((seg_bin > 0).sum())
            except Exception:
                seg_area0 = 0
            self.log(f'Seg mask area (roi)={seg_area0}, erode_px={erode_px}')
            if erode_px > 0:
                try:
                    seg_bin = cv2.erode(seg_bin, None, iterations=erode_px)
                except Exception:
                    pass
            # if segmentation mask is empty after normalization/erosion, skip detection
            if seg_bin is None or seg_bin.sum() == 0:
                self.log(f'Segmentation mask empty after erode — skipping detection for this unit')
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
            self.log(f'Residual mask area={int((mask > 0).sum())}')
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
        self.log(f'Defect area filter: min={min_area}, max={max_area}, seg_area={seg_area}')
        found = False
        for c in cnts:
            a = cv2.contourArea(c)
            if a >= min_area and a <= max_area:
                cv2.drawContours(mask2, [c], -1, 255, -1)
                found = True
            else:
                if a >= min_area:
                    self.log(f'Skipping large contour area={int(a)} (>max={max_area})')
        if not found:
            return None
        h_m, w_m = mask2.shape
        bytes_per_line = w_m
        qimg_mask = QtGui.QImage(mask2.data.tobytes(), w_m, h_m, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        pm_mask = QtGui.QPixmap.fromImage(qimg_mask)
        return pm_mask

    def test_defect_detection_all(self):
        # run defect detection on all thumbnails and update thumbnails/icons
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
            pix = item.data(QtCore.Qt.UserRole)
            if not isinstance(pix, QtGui.QPixmap):
                self.log(f'Unit {row}: no thumbnail, skipping')
                continue
            seg_mask_pm = item.data(QtCore.Qt.UserRole + 1)
            if not isinstance(seg_mask_pm, QtGui.QPixmap):
                self.log(f'Unit {row}: no segmentation mask, skipping')
                continue
            pm_mask = self._detect_defects_on_pix(pix, seg_mask_pm)
            # store (or clear) defect mask; icons will be refreshed for all items after the loop
            item.setData(QtCore.Qt.UserRole + 2, pm_mask if isinstance(pm_mask, QtGui.QPixmap) else None)
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
        # zoom to first unit and enable drawing an exclusion
        if not self.img_widget.grid_rects:
            QtWidgets.QMessageBox.information(self, 'Info', 'Create indexing first before adding exclusions.')
            return
        # zoom to first unit and enable exclusion drawing
        self.on_thumbnail_selected(0)
        # enable drawing temporarily regardless of edit lock
        self.img_widget.exclusion_mode = True
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
    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', '.', 'Images (*.png *.jpg *.tif *.bmp)')
        if path:
            try:
                self.img_widget.load_image(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', str(e))
            # reset masks/thumbnails
            self.thumb_list.clear()
            self.img_widget.selected_cell_index = None
            self.img_widget.selected_mask_pixmap = None

    def apply_indexing(self):
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
            # apply any user-defined exclusions (relative to unit) to this mask
            for excl in getattr(self, 'exclusions', []):
                try:
                    if excl.get('shape') == 'rect':
                        ex = int(excl.get('x', 0)); ey = int(excl.get('y', 0))
                        ew = int(excl.get('w', 0)); eh = int(excl.get('h', 0))
                        x0 = max(0, ex); y0 = max(0, ey)
                        x1 = min(w, ex + ew); y1 = min(h, ey + eh)
                        if x1 > x0 and y1 > y0:
                            mask[y0:y1, x0:x1] = 0
                    else:
                        # circle
                        cx = int(excl.get('cx', 0)); cy = int(excl.get('cy', 0)); r = int(excl.get('r', 0))
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
            qimg_mask = QtGui.QImage(mask.data.tobytes(), w_m, h_m, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            pm_mask = QtGui.QPixmap.fromImage(qimg_mask)
            # store full-resolution mask in corresponding thumbnail item if exists
            # find thumbnail item by index
            if idx < self.thumb_list.count():
                item = self.thumb_list.item(idx)
                item.setData(QtCore.Qt.UserRole + 1, pm_mask)
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

    def on_overlay_mode_changed(self, *_):
        # update selected overlay and all thumbnail icons
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
            seg_pm = item.data(QtCore.Qt.UserRole + 1)
            defect_pm = item.data(QtCore.Qt.UserRole + 2)
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
            base_pm = item.data(QtCore.Qt.UserRole)
            if not isinstance(base_pm, QtGui.QPixmap):
                continue
            base_disp = base_pm.scaled(128, 128, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            seg_pm = item.data(QtCore.Qt.UserRole + 1)
            defect_pm = item.data(QtCore.Qt.UserRole + 2)

            if mode == 'None':
                item.setIcon(QtGui.QIcon(base_disp))
                continue

            out = base_disp
            if mode in ('Segmentation', 'Both') and isinstance(seg_pm, QtGui.QPixmap):
                seg_scaled = seg_pm.scaled(base_disp.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                out = self._make_overlay_pixmap(out, seg_scaled, color=(0, 255, 0))
            if mode in ('Defect', 'Both') and isinstance(defect_pm, QtGui.QPixmap):
                defect_scaled = defect_pm.scaled(base_disp.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                out = self._make_overlay_pixmap(out, defect_scaled, color=(255, 0, 0))

            item.setIcon(QtGui.QIcon(out))

    def _make_overlay_pixmap(self, pix, mask_pix, color=(255, 0, 0), alpha_val=200):
        # overlay mask (colored) on cell pixmap
        base = QtGui.QPixmap(pix)
        mask = QtGui.QPixmap(mask_pix)
        # ensure same size
        mask = mask.scaled(base.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        result = QtGui.QPixmap(base.size())
        result.fill(QtCore.Qt.transparent)
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
        mask_img = mask.toImage().convertToFormat(4)
        h = mask_img.height(); w = mask_img.width()
        bits = mask_img.bits(); bits.setsize(mask_img.byteCount())
        arr = np.frombuffer(bits, np.uint8).reshape((h, w, 4))
        mask_alpha = (arr[:, :, 0] > 0).astype(np.uint8) * alpha_val
        out_img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        out_img.fill(0)
        ob = out_img.bits(); ob.setsize(out_img.byteCount())
        oarr = np.frombuffer(ob, np.uint8).reshape((h, w, 4))
        # assign color (B,G,R order in QImage byte layout)
        oarr[:, :, 0] = color[2] if len(color) >= 3 else 0
        oarr[:, :, 1] = color[1] if len(color) >= 2 else 0
        oarr[:, :, 2] = color[0] if len(color) >= 1 else 0
        oarr[:, :, 3] = mask_alpha
        return QtGui.QPixmap.fromImage(out_img)

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
        result.fill(QtCore.Qt.transparent)
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
        seg_pm = item.data(QtCore.Qt.UserRole + 1)
        defect_pm = item.data(QtCore.Qt.UserRole + 2)
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
        # compute erosion outline for the segmentation mask of `row` and store as display QPainterPath
        if row is None:
            row = self.img_widget.selected_cell_index
        if row is None or row < 0 or row >= len(self.img_widget.grid_rects):
            self.img_widget.erosion_path = None
            self.img_widget.update()
            return
        item = self.thumb_list.item(row)
        seg_pm = item.data(QtCore.Qt.UserRole + 1)
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
            dr = self.img_widget.imgrect_to_display(QtCore.QRect(inx, iny, inw, inh))
            path = QtGui.QPainterPath()
            path.addRect(QtCore.QRectF(dr))
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
        self.log(f'Erosion outline roi_area={seg_area0}, erode_px={erode_px}')
        # erode by user parameter (in pixels)
        if erode_px > 0:
            try:
                seg_bin = cv2.erode(seg_bin, None, iterations=erode_px)
            except Exception:
                pass
        # find contours on eroded mask (unit-local coords)
        cnts, _ = cv2.findContours(seg_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self.img_widget.erosion_path = None
            self.img_widget.update()
            return
        # build QPainterPath in display coordinates by mapping unit-local points to absolute image display coords
        path = QtGui.QPainterPath()
        # unit top-left in image coords
        r, idx = self.img_widget.grid_rects[row]
        ux, uy = int(r[0]), int(r[1])
        scale = self.img_widget.scale
        for ci, c in enumerate(cnts):
            try:
                pts = c.reshape(-1, 2)
            except Exception:
                continue
            if pts.size == 0:
                continue
            # first point
            p0x = int((ux + int(pts[0][0])) * scale)
            p0y = int((uy + int(pts[0][1])) * scale)
            path.moveTo(p0x, p0y)
            for pi in range(1, pts.shape[0]):
                px = int((ux + int(pts[pi][0])) * scale)
                py = int((uy + int(pts[pi][1])) * scale)
                path.lineTo(px, py)
            path.closeSubpath()
        self.img_widget.erosion_path = path
        self.img_widget.update()

    def img_widget_zoom(self, factor: float):
        # apply zoom multiplier to ImageWidget
        self.img_widget.manual_zoom *= factor
        self.img_widget.updateScale()
        self.img_widget.update()

    def ensure_fit_view(self):
        # reset manual zoom and fit the image to viewport, reset scrollbars to origin
        self.img_widget.manual_zoom = 1.0
        self.img_widget.updateScale()
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
        if event.type() == QtCore.QEvent.Resize and source == getattr(self, 'scroll', None).viewport():
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
            pm_mask = item.data(QtCore.Qt.UserRole + 1)
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
        # include exclusions if present
        exports = {'metadata': meta, 'boxes': boxes, 'exclusions': getattr(self, 'exclusions', [])}
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
            pm_mask = item.data(QtCore.Qt.UserRole + 1)
            if isinstance(pm_mask, QtGui.QPixmap):
                qim = pm_mask.toImage()
                buf = QtCore.QBuffer()
                buf.open(QtCore.QIODevice.WriteOnly)
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
                            item.setData(QtCore.Qt.UserRole + 1, pm_mask)
                            thumb_pm = item.data(QtCore.Qt.UserRole)
                            if isinstance(thumb_pm, QtGui.QPixmap):
                                overlay = self._make_overlay_pixmap(thumb_pm, pm_mask.scaled(thumb_pm.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation))
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
                        item.setData(QtCore.Qt.UserRole + 1, pm)
                        thumb_pm = item.data(QtCore.Qt.UserRole)
                        if isinstance(thumb_pm, QtGui.QPixmap):
                            overlay = self._make_overlay_pixmap(thumb_pm, pm.scaled(thumb_pm.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation))
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
                item.setData(QtCore.Qt.UserRole + 1, pm)
                thumb_pm = item.data(QtCore.Qt.UserRole)
                if isinstance(thumb_pm, QtGui.QPixmap):
                    overlay = self._make_overlay_pixmap(thumb_pm, pm.scaled(thumb_pm.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation))
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
            icon = QtGui.QIcon(sub.scaled(128, 128, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            item = QtWidgets.QListWidgetItem(icon, str(idx))
            # store pixmap for export
            item.setData(QtCore.Qt.UserRole, sub)
            self.thumb_list.addItem(item)
        # update defect unit spin range if present
        if hasattr(self, 'defect_unit_spin'):
            n = max(0, self.thumb_list.count() - 1)
            self.defect_unit_spin.setRange(0, n)

    def export_thumbnails(self):
        if self.thumb_list.count() == 0:
            QtWidgets.QMessageBox.information(self, 'Info', 'No thumbnails to export. Apply indexing first.')
            return
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder to save thumbnails', '.')
        if not dirpath:
            return
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            pm = item.data(QtCore.Qt.UserRole)
            if isinstance(pm, QtGui.QPixmap):
                fname = f"unit_{i:04d}.png"
                pm.save(QtCore.QDir.cleanPath(QtCore.QDir(dirpath).filePath(fname)))
        QtWidgets.QMessageBox.information(self, 'Saved', f'Exported {self.thumb_list.count()} thumbnails to {dirpath}')


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.showFullScreen()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
