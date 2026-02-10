# run.py
import os
import cv2
from track_drone import track_drone

# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "video_path": "drone-tracking-datasets/dataset1/cam0.mp4",
    "output_dir": "outputs",
    "output_name": "dataset1_cam0_trajectory.png",
}

# -----------------------------
# Utils
# -----------------------------
def load_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read first frame")

    return frame


def draw_trajectory(bg, trajectory, thickness=4):
    for i in range(1, len(trajectory)):
        if trajectory[i - 1] is None or trajectory[i] is None:
            continue
        cv2.line(bg, trajectory[i - 1], trajectory[i], (0, 0, 255), thickness)
    return bg


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    trajectory = track_drone(
        video_path=CONFIG["video_path"],
        visualize=True,  # True if you want to see live
    )

    bg = load_first_frame(CONFIG["video_path"])
    bg = draw_trajectory(bg, trajectory)

    out_path = os.path.join(CONFIG["output_dir"], CONFIG["output_name"])
    cv2.imwrite(out_path, bg)

    print(f"Saved trajectory image to: {out_path}")


if __name__ == "__main__":
    main()
