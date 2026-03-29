import os
from app.pipeline.video_loader import load_video
from app.pipeline.face_detector import FaceDetector
from app.pipeline.pose_tracker import MotionTracker
from app.pipeline.event_detector import detect_events
from app.pipeline.classifier import classify_segments
from app.pipeline.exporter import export_json

def run_pipeline(video_path: str):
    cap, fps, frame_count, duration = load_video(video_path)

    face_detector = FaceDetector()
    motion_tracker = MotionTracker()

    frame_records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        face_data = face_detector.process(frame)
        motion_score = motion_tracker.compute_motion(frame)

        frame_records.append({
            "timestamp": timestamp,
            "face_visible": face_data["face_visible"],
            "mouth_ratio": face_data["mouth_ratio"],
            "motion_score": motion_score
        })

        frame_idx += 1

    cap.release()

    events = detect_events(frame_records, fps)
    segments = classify_segments(frame_records, events, duration)

    avg_engagement = sum(1 if s["engagement"] == "high" else 0 for s in segments) / max(len(segments), 1)
    high_engagement_pct = avg_engagement * 100

    payload = {
        "filename": os.path.basename(video_path),
        "duration": round(duration, 2),
        "total_events": len(events),
        "avg_engagement": round(avg_engagement, 2),
        "high_engagement_pct": round(high_engagement_pct, 2),
        "segments": segments,
        "events": events
    }

    export_json("output.json", payload)
    return payload

if __name__ == "__main__":
    result = run_pipeline("sample.mp4")
    print(result)