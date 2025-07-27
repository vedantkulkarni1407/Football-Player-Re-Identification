import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- ByteTrack tracker config ---
import supervision as sv
tracker = sv.ByteTrack(
    track_activation_threshold=0.7,
    lost_track_buffer=300,
    frame_rate=60,
    minimum_consecutive_frames=2
)

print("Using ByteTrack tracker")

smoothing_factor = 0.7
smoothed_boxes = {}

# --- ID Management ---
next_player_id = 1
tracker_to_player_id = {}
used_player_ids = set()
lost_tracks = {}
track_colors = {}

# --- Frame-to-frame tracking data ---
prev_frame_data = {}  # player_id -> {'box': xyxy, 'color': rgb}

# --- Color distance ---
def color_distance(color1, color2):
    return np.linalg.norm(color1 - color2)

def get_jersey_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    jersey_height = int((y2 - y1) * 0.4)
    jersey = frame[y1:y1 + jersey_height, x1:x2]
    return np.mean(jersey, axis=(0, 1)) if jersey.size > 0 else np.array([0, 0, 0])

def estimate_camera_motion(prev_frame, curr_frame):
    if prev_frame is None:
        return np.array([0, 0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    if prev_pts is None or len(prev_pts) < 10:
        return np.array([0, 0])
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    if len(good_prev) < 10:
        return np.array([0, 0])
    return np.mean(good_curr - good_prev, axis=0).flatten()[:2]

def get_box_center(box):
    """Calculate center point of bounding box"""
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

def is_similar_detection(box1, color1, box2, color2, pos_threshold=50, color_threshold=60):
    """Check if two detections are similar based on position and color"""
    # Calculate position distance between box centers
    center1 = get_box_center(box1)
    center2 = get_box_center(box2)
    pos_dist = np.linalg.norm(center1 - center2)
    
    # Calculate color distance
    col_dist = color_distance(color1, color2)
    
    # Return True if both position and color are within thresholds
    return pos_dist < pos_threshold and col_dist < color_threshold

def find_matching_player_id(current_box, current_color, prev_frame_data):
    """Find the best matching player ID from previous frame"""
    best_match_id = None
    best_similarity_score = float('inf')
    
    for player_id, prev_data in prev_frame_data.items():
        prev_box = prev_data['box']
        prev_color = prev_data['color']
        
        if is_similar_detection(current_box, current_color, prev_box, prev_color):
            # Calculate combined similarity score (lower is better)
            pos_dist = np.linalg.norm(get_box_center(current_box) - get_box_center(prev_box))
            col_dist = color_distance(current_color, prev_color)
            similarity_score = 0.7 * pos_dist + 0.3 * col_dist
            
            if similarity_score < best_similarity_score:
                best_similarity_score = similarity_score
                best_match_id = player_id
    
    return best_match_id

def reidentify_player(new_det, lost_tracks, color_threshold=60, position_threshold=150):
    best_match = None
    best_score = float('inf')
    for pid, data in lost_tracks.items():
        color_dist = color_distance(new_det['color'], data['color'])
        pos_dist = np.linalg.norm(new_det['position'] - data['position'])
        score = 0.7 * color_dist + 0.3 * pos_dist
        if score < best_score and color_dist < color_threshold and pos_dist < position_threshold:
            best_score = score
            best_match = pid
    return best_match

def track_video(video_path, model_path):
    global next_player_id, tracker_to_player_id, used_player_ids, lost_tracks, track_colors, prev_frame_data

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("output_tracking.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        motion = estimate_camera_motion(prev_frame, frame)
        prev_frame = frame.copy()

        results = model(frame, conf=0.8, verbose=False)
        detections = []
        colors = []

        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if int(cls) != 2 or conf < 0.7:
                    continue
                box_np = box.cpu().numpy() - np.tile(motion, 2)
                box_np = np.clip(box_np, [0, 0, 0, 0], [width, height, width, height])
                detections.append((*box_np, conf.item()))
                colors.append(get_jersey_color(frame, box_np))

        if detections:
            det_arr = np.array(detections)
            sv_dets = sv.Detections(
                xyxy=det_arr[:, :4],
                confidence=det_arr[:, 4],
                class_id=np.zeros(len(det_arr))
            )
        else:
            sv_dets = sv.Detections.empty()
            colors = []

        tracks = tracker.update_with_detections(sv_dets)
        current_ids = set()
        current_frame_data = {}  # Store current frame data for next frame comparison

        # Process each detection
        for i in range(len(tracks.xyxy)):
            xyxy = tracks.xyxy[i]
            color = colors[i] if i < len(colors) else np.array([0, 0, 0])
            
            # First, try to match with previous frame data
            matched_player_id = find_matching_player_id(xyxy, color, prev_frame_data)
            
            if matched_player_id is not None:
                # Use the matched player ID
                player_id = matched_player_id
            else:
                # Check if we have a ByteTrack tracker ID
                if i < len(tracks.tracker_id):
                    track_id = tracks.tracker_id[i]
                    
                    if track_id not in tracker_to_player_id:
                        # Try to reidentify from lost tracks
                        new_det = {'color': color, 'position': np.array([xyxy[0], xyxy[1]])}
                        match_id = reidentify_player(new_det, lost_tracks)
                        if match_id is not None:
                            tracker_to_player_id[track_id] = match_id
                            del lost_tracks[match_id]
                        else:
                            # Assign new player ID
                            tracker_to_player_id[track_id] = next_player_id
                            used_player_ids.add(next_player_id)
                            next_player_id += 1
                    
                    player_id = tracker_to_player_id[track_id]
                else:
                    # Fallback: assign new player ID
                    player_id = next_player_id
                    used_player_ids.add(next_player_id)
                    next_player_id += 1

            current_ids.add(player_id)
            track_colors[player_id] = color

            # Apply smoothing
            if player_id in smoothed_boxes:
                smoothed_boxes[player_id] = smoothing_factor * xyxy + (1 - smoothing_factor) * smoothed_boxes[player_id]
            else:
                smoothed_boxes[player_id] = xyxy

            # Store current frame data for next frame comparison
            current_frame_data[player_id] = {
                'box': smoothed_boxes[player_id].copy(),
                'color': color.copy()
            }

            # Draw bounding box and label
            box = smoothed_boxes[player_id].astype(int)
            label = f"Player {player_id}"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Handle lost tracks
        lost_ids = set(tracker_to_player_id.values()) - current_ids
        for tid, pid in list(tracker_to_player_id.items()):
            if pid in lost_ids:
                box = smoothed_boxes.get(pid, np.array([0, 0, 0, 0]))
                lost_tracks[pid] = {
                    'color': track_colors.get(pid, np.array([0, 0, 0])),
                    'position': np.array([box[0], box[1]])
                }
                del tracker_to_player_id[tid]

        # Update previous frame data for next iteration
        prev_frame_data = current_frame_data.copy()

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Output saved as output_tracking.mp4")

def main():
    track_video("15sec_input_720p.mp4", "best.pt")

if __name__ == "__main__":
    main()
