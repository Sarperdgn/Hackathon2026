from app.config import SEGMENT_WINDOW_SECONDS

def classify_segments(frame_records, events, duration):
    segments = []
    segment_start = 0

    while segment_start < duration:
        segment_end = min(segment_start + SEGMENT_WINDOW_SECONDS, duration)

        segment_events = [
            e for e in events
            if segment_start <= float(e["timestamp"]) < segment_end
        ]

        yawn_count = sum(1 for e in segment_events if e["event_type"] == "yawning")
        inactivity_count = sum(1 for e in segment_events if e["event_type"] == "prolonged_inactivity")

        score = 1.0
        score -= 0.35 * yawn_count
        score -= 0.45 * inactivity_count

        label = "high" if score >= 0.5 else "low"
        confidence = min(1.0, abs(score - 0.5) + 0.5)

        segments.append({
            "segment_start": round(segment_start, 2),
            "segment_end": round(segment_end, 2),
            "engagement": label,
            "confidence": round(confidence, 2),
            "event_count": len(segment_events)
        })

        segment_start += SEGMENT_WINDOW_SECONDS

    return segments