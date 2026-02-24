# CCTV Change Detection System

A machine learning system that detects significant changes in CCTV footage, captures **before** and **after** frames, logs exact timestamps, and generates visual reports.

---

## What It Does

- Scans CCTV video frame-by-frame
- Detects the **exact moment** a change occurs (person enters, object moves, scene changes)
- Saves a **BEFORE** frame (just before the change)
- Saves an **AFTER** frame (just after the change)
- Saves a **DIFF MASK** (heat map showing exactly where the change occurred)
- Logs **timestamps** in real wall-clock time (e.g. `2024-06-01 08:32:47`)
- Generates a **timeline chart** of all change events
- Exports all results as a JSON report

---

## Quick Start (Google Colab)

### Step 1 — Open the Notebook

Upload `CCTV_Change_Detection.ipynb` to [Google Colab](https://colab.research.google.com/) or open it directly from Google Drive.

### Step 2 — Install Dependencies (Cell 1)

```python
!pip install opencv-python-headless numpy matplotlib Pillow tqdm -q
```

Run this once per Colab session. It installs all required libraries.

### Step 3 — Run Setup Cells (Cells 2-4)

Run Cells 2, 3, and 4 in order. These load imports, config, and the core engine.

### Step 4 — Try the Demo (Cell 5)

Cell 5 generates a synthetic 30-second test video with 3 simulated change events and runs detection on it. No video needed to test!

### Step 5 — Use Your Own Video (Cell 6)

**Option A - Upload from your computer:**
```python
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
```

**Option B - Use Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
video_path = '/content/drive/MyDrive/your_cctv_footage.mp4'
```

Then set your parameters and run Cell 6.

### Step 6 — Download Results (Cell 7)

Cell 7 zips all output files and downloads them to your computer.

---

## Configuration Parameters

These are found in the `Config` class. Adjust them for your footage:

| Parameter | Default | Description |
|---|---|---|
| `CHANGE_THRESHOLD` | `30` | Pixel intensity diff to count as change. Lower = more sensitive (try 10-50). |
| `MIN_CHANGED_AREA_PERCENT` | `2.0` | Minimum % of the frame that must change. Prevents tiny noise from triggering events. |
| `MOTION_BLUR_KERNEL` | `5` | Gaussian blur kernel for noise reduction. Must be an odd number. |
| `FRAME_SKIP` | `2` | Process every Nth frame. 1 = all frames (slow), 3 = every 3rd (fast). |
| `BUFFER_SECONDS` | `2.0` | How many seconds before and after an event to capture. |
| `MIN_EVENT_GAP_SECONDS` | `3.0` | Minimum time between two separate events. Prevents duplicate detections. |
| `OUTPUT_DIR` | `cctv_output` | Folder where all results are saved. |
| `DISPLAY_LIVE` | `False` | Set True to see frame-by-frame progress (much slower). |

### Tuning Guide

| Scenario | Recommended Settings |
|---|---|
| Very busy CCTV (street, crowd) | `CHANGE_THRESHOLD=40`, `MIN_CHANGED_AREA_PERCENT=5.0` |
| Quiet room (office, warehouse) | `CHANGE_THRESHOLD=20`, `MIN_CHANGED_AREA_PERCENT=1.0` |
| Night vision / low light | `CHANGE_THRESHOLD=15`, `MOTION_BLUR_KERNEL=7` |
| Long video (hours) | `FRAME_SKIP=3` or `FRAME_SKIP=5` |
| Short clip (minutes) | `FRAME_SKIP=1` |

---

## Output Files

All outputs are saved to `cctv_output/` (or your configured `OUTPUT_DIR`).

```
cctv_output/
   event_001_t25s_before.jpg     <- Frame just BEFORE change
   event_001_t25s_after.jpg      <- Frame just AFTER change
   event_001_t25s_diff.jpg       <- Heat map of what changed
   event_001_t25s_meta.json      <- Timestamp and score data
   event_002_t72s_before.jpg
   event_002_t72s_after.jpg
   event_002_t72s_diff.jpg
   event_002_t72s_meta.json
   timeline.png                  <- Full change timeline chart
   all_events.json               <- Combined JSON report
```

### Example all_events.json

```json
[
  {
    "event_id": 1,
    "timestamp_sec": 25.04,
    "wall_time": "2024-06-01 08:00:25",
    "change_score": 8.3
  },
  {
    "event_id": 2,
    "timestamp_sec": 72.16,
    "wall_time": "2024-06-01 08:01:12",
    "change_score": 12.7
  }
]
```

---

## How It Works

The system uses two complementary techniques and combines them:

**1. Background Subtraction (MOG2)**
OpenCV's MOG2 algorithm builds a statistical model of the background over time. Any pixel that deviates significantly from this model is flagged as foreground (i.e., something changed). It handles gradual lighting changes and shadows well, both common in CCTV footage.

**2. Frame Differencing**
Each frame is compared to the previous frame. Where pixel values differ beyond the `CHANGE_THRESHOLD`, those pixels are marked as changed. This catches fast-moving events.

**Combined + Cleaned**
The two masks are merged (union), then morphological operations remove tiny noise speckles and fill small gaps, producing a clean map of changed regions.

**Event Logic**
When the percentage of changed pixels exceeds `MIN_CHANGED_AREA_PERCENT`, an event is triggered. The system records the frame from the buffer (BEFORE), waits `BUFFER_SECONDS`, then captures the current frame (AFTER). A cooldown (`MIN_EVENT_GAP_SECONDS`) prevents the same continuous motion from generating dozens of duplicate events.

---

## Supported Video Formats

Any format supported by OpenCV: `.mp4` (recommended), `.avi`, `.mkv`, `.mov`, `.ts`, `.h264`, `.h265`

---

## Common Errors and Fixes

**Cannot open video / FileNotFoundError**
Check the file path is correct. Make sure the file was fully uploaded before running. Try re-uploading the file.

**pip install fails in Colab**
Use `opencv-python-headless` not `opencv-python`. The headless version works in Colab. If still failing, restart the Colab runtime (Runtime > Restart runtime) and try again.

**Too many events detected (false positives)**
Increase `MIN_CHANGED_AREA_PERCENT` (e.g., from 2.0 to 5.0). Increase `CHANGE_THRESHOLD` (e.g., from 30 to 45). Increase `MIN_EVENT_GAP_SECONDS` (e.g., from 3.0 to 10.0).

**Too few events (missing real changes)**
Decrease `MIN_CHANGED_AREA_PERCENT` (e.g., from 2.0 to 0.5). Decrease `CHANGE_THRESHOLD` (e.g., from 30 to 15). Set `FRAME_SKIP = 1` so no frames are skipped.

**Video processes but shows 0 events**
Run Cell 5 first (demo) to confirm the engine works. Check that your video actually has visible movement. Try setting `DISPLAY_LIVE = True` to watch frame-by-frame. Lower both thresholds significantly as a test.

**Out of memory in Colab**
Increase `FRAME_SKIP` to 3 or 5. Process a shorter clip first to test.

**Timestamps show 00:00:01 instead of real time**
Pass `start_str='2024-06-01 08:30:00'` with the actual recording start time. Format must be exactly `YYYY-MM-DD HH:MM:SS`.

---

## Requirements

All installed automatically by Cell 1:

```
opencv-python-headless >= 4.5
numpy >= 1.21
matplotlib >= 3.4
Pillow >= 8.0
tqdm >= 4.60
```

Python 3.7 or higher required (Google Colab uses 3.10+ by default).

---

## Project Files

| File | Description |
|---|---|
| `CCTV_Change_Detection.ipynb` | Main Google Colab notebook — use this |
| `cctv_change_detection.py` | Standalone Python script (same code, for local use) |
| `README.md` | This file |
