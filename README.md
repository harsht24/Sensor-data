# Sensor-data

**Dual Camera Data Collection Setup Guide**

This guide explains how to record synchronized data from the Intel RealSense D435 (RGB/Depth) and the Topdon TC001 (Thermal) camera using Python, OpenCV, OBS Studio, and YOLO.

---

### 1. Requirements

- Python 3.10
- OBS Studio
- Topdon TCView software
- Intel RealSense SDK

**Python Packages:**

```bash
pip install opencv-python pyrealsense2 ultralytics numpy
```

---

### 2. Hardware Setup

- **RealSense D435**: Plug into USB 3.0 and test with RealSense Viewer.
- **Topdon TC001**: Open TCView software to confirm the thermal feed is working.
- **OBS Studio**:
  - Download from: [https://obsproject.com/](https://obsproject.com/)
  - Install and open OBS.
  - Add a **"Window Capture"** source that selects the **TCView** window.
  - Go to **Tools > VirtualCam** and click **"Start Virtual Camera"** to enable OBS as a webcam source.

---

### 3. Identify Camera Indices

To find the correct index for RealSense and OBS Virtual Camera, run this helper script:

```python
import cv2

for i in range(5):
    print(f"\n Trying camera index {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera at index {i} is working.")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print(f"Camera at index {i} opened but returned no frame.")
    else:
        print(f"Camera index {i} not available.")
    cap.release()
```

Use this to determine the index for `THERMAL_CAM_INDEX` and the RealSense RGB stream.

---

### 4. Running the Script

1. Clone or create the `Dual.py` script.
2. In the script, set:
   ```python
   RECORD_DURATION_SEC = 10  # Change recording time (in seconds)
   THERMAL_CAM_INDEX = 2     # Set OBS Virtual Camera index
   ```
3. Run the script:
   ```bash
   python Dual.py
   ```

The script will:

- Capture RealSense and Thermal feeds
- Apply YOLO detections
- Save a side-by-side video (`output.mp4`)
- Store frames in `frames/realsense/` and `frames/thermal/`
- Log detections in `frame_log.csv`

---

### 5. Output Files

- `output.mp4` — Combined RealSense + Thermal video with detections
- `frames/realsense/` — RealSense annotated frames
- `frames/thermal/` — Thermal annotated frames
- `frame_log.csv` — YOLO detection logs with frame ID and bounding boxes

---
