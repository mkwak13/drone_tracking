
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# bbox center calculation
def center_of(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def track_drone(
    video_path,
    conf=0.05, # confidence level
    iou_threshold=0.2,
    max_age=5,
    stable_frames=3, # visible in 3 consecutive frames
    candidate_max_move=80, # move limit per frame
    max_missed_frames=5, # connect trajectory if detected within 5 frames
    visualize=False,
):
    model = YOLO("yolov8s.pt") # yolov8n for faster processing(less accurate)
    tracker = Sort(iou_threshold=iou_threshold, max_age=max_age)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    cv2.namedWindow("Drone Tracking", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Drone Tracking", 960, 540)

    # trajectory holder
    tracks_history = []

    # counter
    stable_count = 0
    missed_count = 0
    candidate_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, box.conf[0].item()])

        detections = (
            np.array(detections) if len(detections) else np.empty((0, 5))
        )
        tracks = tracker.update(detections)

        selected = None
        best_dist = float("inf")

        for trk in tracks:
            cx, cy = center_of(trk[:4])
            if candidate_center is None:
                selected = trk
                break

            dx = cx - candidate_center[0]
            dy = cy - candidate_center[1]
            dist = (dx * dx + dy * dy) ** 0.5
            # check continuity
            if dist < best_dist:
                best_dist = dist
                selected = trk

        if selected is not None:
            current_center = center_of(selected[:4])

            if candidate_center is None:
                candidate_center = current_center
                stable_count = 1
            else:
                dx = current_center[0] - candidate_center[0]
                dy = current_center[1] - candidate_center[1]
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < candidate_max_move:
                    stable_count += 1
                    candidate_center = current_center
                else:
                    candidate_center = current_center
                    stable_count = 1
                    missed_count = 0

            # treat as detected drone
            if stable_count == stable_frames:
                tracks_history.append(None)

            # record detected trajectory
            if stable_count >= stable_frames:
                missed_count = 0
                tracks_history.append(current_center)

                if visualize:
                    x1, y1, x2, y2 = map(int, selected[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, current_center, 4, (0, 0, 255), -1)

        else:
            missed_count += 1
            # drone is out of frame
            if missed_count > max_missed_frames:
                candidate_center = None
                stable_count = 0
                missed_count = 0
                tracks_history.append(None)

        if visualize:
            for i in range(1, len(tracks_history)):
                if tracks_history[i - 1] is None or tracks_history[i] is None:
                    continue
                cv2.line(
                    frame,
                    tracks_history[i - 1],
                    tracks_history[i],
                    (0, 0, 255),
                    4,
                )

            cv2.imshow("Drone Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    return tracks_history
