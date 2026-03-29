import cv2
import os

from app.pipeline.face_detector import FaceDetector
from app.pipeline.pose_tracker import MotionTracker
from app.pipeline.event_detector import detect_events
from app.config import RESIZE_WIDTH, RESIZE_HEIGHT

VIDEO_PATH = "source/IMG_7925.MOV"
NO_FACE_DIR = "no_face"


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"File not found: {VIDEO_PATH}")

    os.makedirs(NO_FACE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    print(f"FPS: {fps}")

    face_detector = FaceDetector()
    motion_tracker = MotionTracker()

    frame_records = []
    frame_idx = 0
    max_mouth_ratio = 0.0
    max_mouth_timestamp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        face_result = face_detector.process(frame_resized)
        motion_score = motion_tracker.compute_motion(frame_resized)

        mouth_ratio = face_result["mouth_ratio"]

        if mouth_ratio > max_mouth_ratio:
            max_mouth_ratio = mouth_ratio
            max_mouth_timestamp = timestamp

        frame_records.append({
            "timestamp": timestamp,
            "face_visible": face_result["face_visible"],
            "mouth_ratio": mouth_ratio,
            "motion_score": motion_score
        })

        if face_result["face_visible"]:
            print(
                f"{timestamp:.2f}s: FACE DETECTED | "
                f"mouth_ratio={mouth_ratio:.3f} | "
                f"motion_score={motion_score:.3f}"
            )
        else:
            print(
                f"{timestamp:.2f}s: no face | "
                f"motion_score={motion_score:.3f}"
            )

        frame_idx += 1

    cap.release()

    print(f"\nProcessed frames: {len(frame_records)}")
    print(f"Max mouth ratio: {max_mouth_ratio:.3f} at {max_mouth_timestamp:.2f}s")

    events = detect_events(frame_records, fps)

    print("\n=== DETECTED EVENTS ===")
    if not events:
        print("No events detected.")
    else:
        for event in events:
            print(event)


if __name__ == "__main__":
    main()