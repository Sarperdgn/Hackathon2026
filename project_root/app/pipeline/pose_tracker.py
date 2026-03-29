import cv2
import numpy as np

class MotionTracker:
    def __init__(self):
        self.prev_gray = None

    def compute_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0

        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            motion_score = float(np.mean(diff))

        self.prev_gray = gray
        return motion_score