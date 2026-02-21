# CompassLayer

A modular, resolution-independent icon detection and navigation system for game HUDs, optimized for AC Shadows.

## Features

- **Proportion-Based Architecture**: All logic (coordinates, offsets, UI scaling) works on a 0.0 to 1.0 relative scale, ensuring perfect compatibility across different monitor resolutions.
- **Masked Template Matching**: Supports 4-channel BGRA templates with alpha masks to ignore complex game backgrounds and improve tracking stability.
- **Smart OCR Interaction**: Dynamically triggers distance extraction (via Tesseract) only when the target is centered to optimize CPU performance.
- **Adaptive Visualizer**: UI elements (bounding boxes, text) automatically scale their thickness and size based on screen width.
- **Icon Processing Suite**: Includes a utility to automatically crop, center, and generate alpha masks from raw HUD screenshots.

## Project Structure

```text
AC_Shadows_Project/
├── main.py                # Main loop & process orchestrator
├── config.py              # Global settings & icon paths
├── core/                  # Logical core
│   ├── screen.py          # Screen capture (mss based)
│   ├── detector.py        # Masked template matching & NMS
│   └── ocr_engine.py      # Distance parsing (Tesseract)
├── utils/                 # Utilities
│   ├── visualizer.py      # Adaptive UI rendering
│   └── icon_processor.py  # Icon masking & centering tool
├── scripts/               # Maintenance & Test scripts
└── assets/                # Icon templates and test screenshots
```

## Installation

1. Install system dependencies:
   - **Tesseract OCR**: `brew install tesseract` (macOS) or `sudo apt install tesseract-ocr` (Linux).
2. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Process new icons
If you have a raw screenshot of an icon:
```bash
python utils/icon_processor.py assets/icons/my_icon.png
```
This generates `my_icon_centered.png` with a transparent background.

### 2. Run the Navigator
Update `config.py` with your icon paths and run:
```bash
python main.py
```

## Configuration

Settings in `config.py`:
- `ROI_HEIGHT_RATIO`: Area of the screen to search (default top 25%).
- `MATCH_THRESHOLD`: Matching strictness (0.8+ recommended for masked icons).
- `STRAIGHT_AHEAD_THRESHOLD`: How wide the "Straight" center lane is (default 2% of screen width).

## License

MIT
