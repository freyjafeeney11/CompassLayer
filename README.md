# CompassLayer

A modular, resolution-independent icon detection and navigation system with spatial audio guidance for game HUDs, optimized for AC Shadows.

## Features

- **Spatial Audio Engine**: Real-time HRTF binaural panning with threshold-based earcons and dynamic pulsing for main quests.
- **Accessibility Integration**: Fully navigable for blind players using French and English TTS announcements, and global hotkeys.
- **In-Game Settings Menu**: Live F7 menu to adjust ping volume, ping rate, TTS speed, TTS volume, and language without leaving the game.
- **Proportion-Based Architecture**: All UI scaling and detection logic works on a relative scale, ensuring compatibility across different monitor resolutions.
- **Masked Template Matching**: Supports 4-channel BGRA templates with alpha masks to ignore complex game backgrounds.

## Project Structure

```text
CompassLayer/
├── run_live.py            # Main live testing loop
├── package_app.py         # PyInstaller build script
├── config.py              # Global settings & icon paths
├── core/                  # Core modules
│   ├── audiofeedback.py   # Spatial audio & TTS controller
│   ├── offline_audio.py   # Offline testing renderer
│   ├── detector.py        # CV multi-scale detection
│   ├── screen.py          # Screen capture (mss)
│   ├── settings.py        # Runtime settings logic
│   ├── i18n.py            # Bilingual translation dictionary
│   └── ocr_engine.py      # Distance parsing (Tesseract)
├── utils/                 # Utilities (UI Visualizer & Icon Processor)
└── assets/                # Audio files and Icon templates
```

## Installation

1. Install **Tesseract OCR**: `brew install tesseract` (macOS) or download the Windows installer.
2. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: requires `pywin32` for optimal TTS sequential delivery, `pyttsx3` is supported as a fallback)*

## Usage

### 1. Live Navigation
Launch the script and switch to your game window:
```bash
python run_live.py
```
*Tip: Use `--verbose` or `-v` to show frame-by-frame debug output in the terminal.*

**Global Hotkeys:**
- **`F6`**: Read controls aloud (TTS)
- **`F7`**: Open/Close Settings Menu (use Arrow Keys to navigate and adjust)
- **`Shift + F8`**: Trigger Scan Mode (sweeps compass left-to-right)
- **`Shift + F9`**: Quit application

### 2. Process New Icons
```bash
python utils/icon_processor.py assets/icons/my_icon.png
```
Generates `my_icon_centered.png` with a transparent background.

### 3. Build Executable
```bash
python package_app.py
```
Compiles a standalone executable to the `dist/` folder.

## License

MIT
