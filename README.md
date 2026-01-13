Indexing UI for mold surface segmentation

Usage

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the UI:

```powershell
python indexing_ui.py
```

How to use

- Click "Load Image" and open your 4096x300 image (the UI will also accept other sizes).
- Click and drag on the image to draw a bounding box for the first unit (top-left unit).
- Enter `Units X`, `Units Y` (units per block) and `Blocks X`, `Blocks Y` (how many blocks).
- Set `Spacing (units, px)` and `Spacing (blocks, px)`.
- Click "Apply Indexing" to generate the grid overlay.
- Click "Export grid JSON" to save the bounding boxes.

Next steps

- Integrate segmentation methods for each unit (classical vision thresholds + morphology).
- Add per-unit preview and manual corrections.
- Optionally add SAM fallback for low-confidence units.
