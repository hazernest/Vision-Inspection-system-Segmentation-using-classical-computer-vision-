Vision Inspection System (Classical CV)

This project is a PyQt6 desktop tool (with Fluent Widgets) for:

- Indexing a regular grid of units on a mold surface image
- Running per-unit classical vision segmentation
- Running per-unit defect detection ("Foreign material") constrained to the segmentation ROI
- Visualizing results as overlays (Segmentation = green, Defect = red) or as a simple inspection verdict view (X/O)

## Setup

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the UI:

```powershell
python indexing_ui.py
```

## Improvements

- Modernized UI: PyQt6 + Fluent Widgets, plus scrollable settings panels
- Multi-image workflow: load multiple images and switch while preserving per-image results
- Exclusions: add/modify/delete exclusions; edit via popup with X/Y + size and on-image resize handle
- Exclusion alignment: exclusions remain relative to segmentation via per-unit centroid XY shift, persisted in grid JSON (v2)
- Robust image loading: TIFF fallback via OpenCV when Qt fails to decode
- Segmentation/defect stability: hole filling in segmentation ROI; erosion keeps holes (exclusions) via connected components; safer QImage buffer handling

## Workflow

### 1) Index the grid

- Click **Load Image** and open your mold image (4096x3000 recommended, other sizes work).
- Click-drag on the image to draw the **first unit** (top-left unit).
- Enter **Units X / Units Y** and **Blocks X / Blocks Y**, plus **unit spacing** and **block spacing** (in pixels).
- Click **Apply Indexing** to generate the full grid.

Export/Import:

- **Export grid JSON** / **Import grid JSON**: save/load grid rectangles.

### 2) Run segmentation

- Click **Run Segmentation** to compute a segmentation mask for every unit.
- The main view can show overlays across all units (green segmentation, red defect).

Mask export/import:

- **Export Masks + CSV**: exports masks and a CSV summary.

### 3) Add exclusions (optional)

Exclusions let you remove areas from the segmentation mask (e.g., fixtures or irrelevant regions).

- Choose the exclusion shape (rectangle/circle), then use **Add exclusion** and draw on the unit.
- Exclusions are applied when segmentation is generated.

### 4) Defect detection (Foreign material)

In the **Defect** tab:

- **Test on unit**: runs defect detection on the currently selected unit (click a unit or its thumbnail first).
- **Test All Units**: runs defect detection across the entire grid.

Parameters:

- **Threshold**: detection sensitivity
- **Min area (px)**: minimum defect area for an NG verdict
- **Mask erosion (px)**: shrinks the segmentation ROI before detecting defects

Live updates:

- Changing **Threshold / Min area / Mask erosion** automatically updates the currently selected unit’s defect mask and overlays.
- The erosion outline is drawn (cyan) and stays aligned when zooming or using Fit.

### 5) Inspection mode (X/O)

- **Run Inspection** is a toggle:
	- ON: hides overlays and shows **X** for defect / **O** for OK on each unit.
	- OFF: returns to segmentation/defect overlays.

Notes:

- If you change Threshold/Min area/Erosion, inspection mode automatically exits back to overlays.
- Clicking **Test on unit** or **Test All Units** also exits inspection mode.
