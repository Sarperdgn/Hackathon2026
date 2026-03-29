from app.config import (
    YAWN_MOUTH_RATIO,
    INACTIVITY_THRESHOLD_SECONDS,
    MOTION_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    SEGMENT_WINDOW_SECONDS,
    MIN_EVENT_DURATION_SECONDS,
)


def get_segment_bounds(timestamp: float):
    segment_start = int(timestamp // SEGMENT_WINDOW_SECONDS) * SEGMENT_WINDOW_SECONDS
    segment_end = segment_start + SEGMENT_WINDOW_SECONDS
    return round(segment_start, 2), round(segment_end, 2)


def detect_yawns(frame_records, fps):
    events = []
    yawn_start = None

    for record in frame_records:
        mouth_open = (
            record["face_visible"]
            and record["mouth_ratio"] >= YAWN_MOUTH_RATIO
        )

        if mouth_open:
            if yawn_start is None:
                yawn_start = record["timestamp"]
        else:
            if yawn_start is not None:
                duration = record["timestamp"] - yawn_start

                # simpler, less strict confidence
                confidence = min(1.0, duration / 1.0)

                if duration >= MIN_EVENT_DURATION_SECONDS and confidence >= CONFIDENCE_THRESHOLD:
                    segment_start, segment_end = get_segment_bounds(yawn_start)

                    events.append({
                        "timestamp": f"{yawn_start:.2f}",
                        "event_type": "yawning",
                        "engagement": "low",
                        "confidence": round(confidence, 2),
                        "segment_start": f"{segment_start:.2f}",
                        "segment_end": f"{segment_end:.2f}",
                    })

                yawn_start = None

    if yawn_start is not None and len(frame_records) > 0:
        end_time = frame_records[-1]["timestamp"]
        duration = end_time - yawn_start
        confidence = min(1.0, duration / 1.0)

        if duration >= MIN_EVENT_DURATION_SECONDS and confidence >= CONFIDENCE_THRESHOLD:
            segment_start, segment_end = get_segment_bounds(yawn_start)

            events.append({
                "timestamp": f"{yawn_start:.2f}",
                "event_type": "yawning",
                "engagement": "low",
                "confidence": round(confidence, 2),
                "segment_start": f"{segment_start:.2f}",
                "segment_end": f"{segment_end:.2f}",
            })

    return events


def detect_inactivity(frame_records, fps):
    events = []
    inactivity_start = None

    for record in frame_records:
        inactive = (
            record["face_visible"]
            and record["motion_score"] < MOTION_THRESHOLD
        )

        if inactive:
            if inactivity_start is None:
                inactivity_start = record["timestamp"]
        else:
            if inactivity_start is not None:
                duration = record["timestamp"] - inactivity_start
                confidence = min(1.0, duration / (INACTIVITY_THRESHOLD_SECONDS + 5))

                if duration >= INACTIVITY_THRESHOLD_SECONDS and confidence >= CONFIDENCE_THRESHOLD:
                    segment_start, segment_end = get_segment_bounds(inactivity_start)

                    events.append({
                        "timestamp": f"{inactivity_start:.2f}",
                        "event_type": "prolonged_inactivity",
                        "engagement": "low",
                        "confidence": round(confidence, 2),
                        "segment_start": f"{segment_start:.2f}",
                        "segment_end": f"{segment_end:.2f}",
                    })

                inactivity_start = None

    if inactivity_start is not None and len(frame_records) > 0:
        end_time = frame_records[-1]["timestamp"]
        duration = end_time - inactivity_start
        confidence = min(1.0, duration / (INACTIVITY_THRESHOLD_SECONDS + 5))

        if duration >= INACTIVITY_THRESHOLD_SECONDS and confidence >= CONFIDENCE_THRESHOLD:
            segment_start, segment_end = get_segment_bounds(inactivity_start)

            events.append({
                "timestamp": f"{inactivity_start:.2f}",
                "event_type": "prolonged_inactivity",
                "engagement": "low",
                "confidence": round(confidence, 2),
                "segment_start": f"{segment_start:.2f}",
                "segment_end": f"{segment_end:.2f}",
            })

    return events


def detect_events(frame_records, fps):
    yawns = detect_yawns(frame_records, fps)
    inactivity = detect_inactivity(frame_records, fps)
    return yawns + inactivity