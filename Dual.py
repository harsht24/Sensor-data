import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from threading import Thread
from collections import deque
import time
import csv
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Use deque for faster frame access and to avoid queue backlog
thermal_queue = deque(maxlen=1)
realsense_queue = deque(maxlen=1)

# Global flag to stop threads gracefully
running = True

# RealSense camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start RealSense
pipeline.start(config)
print("✅ RealSense stream started")

# Warm-up frames (RealSense needs a few frames before delivering clean data)
for _ in range(30):
    pipeline.wait_for_frames()

THERMAL_CAM_INDEX = 2  # Update this based on camera scan result
RECORD_DURATION_SEC = 25  # <<< Set your recording duration here (in seconds)
os.makedirs("frames/thermal", exist_ok=True)
os.makedirs("frames/realsense", exist_ok=True)

# CSV logger setup
log_file = open("frame_log.csv", "w", newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(["frame_id", "timestamp", "source", "class", "confidence", "x1", "y1", "x2", "y2"])

def open_camera_with_fallback(index):
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"✅ Opened camera index {index} with backend {backend}")
            return cap
    print(f"❌ Could not open camera index {index} with any backend.")
    return None

def capture_thermal():
    cap = open_camera_with_fallback(THERMAL_CAM_INDEX)
    if cap is None:
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ Thermal camera (OBS) opened successfully.")
    while running:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Thermal frame not received.")
            continue

        ts = time.time()
        thermal_queue.append((ts, frame))
        time.sleep(1/30)

def capture_realsense():
    try:
        while running:
            success, frames = pipeline.try_wait_for_frames()
            if not success:
                print("⏳ Waiting for RealSense frames...")
                time.sleep(0.05)
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                print("⚠️ Skipped empty color frame")
                continue

            color = np.asanyarray(color_frame.get_data())
            ts = time.time()
            realsense_queue.append((ts, color))
    except RuntimeError as e:
        if running:
            print("❌ RealSense error:", e)

def get_closest_frame(target_ts, queue, tolerance=0.1):
    for ts, frame in reversed(queue):
        if abs(ts - target_ts) <= tolerance:
            return frame
    return None

# Initialize video writer for 30 FPS recording using mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280*2, 480))  # Width doubled for side-by-side

# Start threads
Thread(target=capture_thermal, daemon=True).start()
Thread(target=capture_realsense, daemon=True).start()

# Frame ID and start time
frame_id = 0
start_time = time.time()

# Main loop
while True:
    if time.time() - start_time > RECORD_DURATION_SEC:
        print("✅ Recording complete.")
        break

    if realsense_queue:
        rs_ts, rs_frame = realsense_queue[-1]
        thermal_frame = get_closest_frame(rs_ts, thermal_queue)

        if thermal_frame is not None:
            frame_id += 1
            timestamp = time.time()

            rs_results = model(rs_frame, verbose=False)[0]
            thermal_results = model(thermal_frame, verbose=False)[0]

            rs_annot = rs_results.plot()
            thermal_annot = thermal_results.plot()

            thermal_resized = cv2.resize(thermal_annot, (rs_annot.shape[1], rs_annot.shape[0]))
            combined = np.hstack((rs_annot, thermal_resized))

            # Write video frame
            video_out.write(combined)

            # Save frames to separate folders
            cv2.imwrite(f"frames/thermal/thermal_{frame_id:04d}.png", thermal_resized)
            cv2.imwrite(f"frames/realsense/realsense_{frame_id:04d}.png", rs_annot)

            # Log detections
            for det in rs_results.boxes:
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                cls = int(det.cls[0])
                conf = float(det.conf[0])
                log_writer.writerow([frame_id, timestamp, "realsense", cls, conf, x1, y1, x2, y2])

            for det in thermal_results.boxes:
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                cls = int(det.cls[0])
                conf = float(det.conf[0])
                log_writer.writerow([frame_id, timestamp, "thermal", cls, conf, x1, y1, x2, y2])

            print(f"Frame {frame_id} @ {timestamp:.3f}")

            cv2.imshow("RealSense + Thermal YOLO", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    time.sleep(1/30)  # Lock frame rate to 30 FPS

# Graceful shutdown
running = False

# Allow threads to exit cleanly
time.sleep(0.5)
video_out.release()
log_file.close()
pipeline.stop()
cv2.destroyAllWindows()
