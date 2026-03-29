import cv2
import os

from app.pipeline.pose_tracker import MotionTracker

VIDEO_PATH = "source/IMG_7925.MOV"  # change if needed


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"File not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    print(f"FPS: {fps}")

    tracker = MotionTracker()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        motion_score = tracker.compute_motion(frame)

        print(f"{timestamp:.2f}s: motion_score={motion_score:.3f}")

        frame_idx += 1

    cap.release()

    print("\n=== RESULT ===")
    print(f"Processed frames: {frame_idx}")


if __name__ == "__main__":
    main()