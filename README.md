# ⚽ Football Player Tracking using YOLOv11 and ByteTrack

This project implements a robust football player tracking system using the YOLOv11 object detection model and the ByteTrack multi-object tracking algorithm. The system is capable of handling re-identification, camera motion compensation, and bounding box smoothing for consistent tracking.

## 🚀 Features

- 📦 **YOLOv11-based Detection**  
  High-accuracy detection of players using Ultralytics YOLOv11 with a confidence threshold of 0.8.

- 🔄 **ByteTrack Tracking**  
  Assigns permanent Player IDs that remain consistent across occlusions and re-entries.

- 🎽 **Color-Based Re-identification**  
  Lost players are re-identified based on jersey color and position similarity.

- 📹 **Camera Motion Compensation**  
  Uses optical flow to reduce the effect of camera motion on tracking accuracy.

- 🧊 **Bounding Box Smoothing**  
  Exponential moving average applied to stabilize bounding box visuals.

## Dependencies
- OpenCV
- NumPy
- Ultratics(YOLO)
- Supervision(ByteTrack)
- Os 
---

## 🛠️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/player-tracking-yolov11.git
   cd player-tracking-yolov11
