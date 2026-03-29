import mediapipe as mp
import cv2
import math

mp_face_mesh = mp.solutions.face_mesh

class FaceDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return {
                "face_visible": False,
                "mouth_ratio": 0.0,
                "landmarks": None
            }

        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        def pt(i):
            return (landmarks[i].x * w, landmarks[i].y * h)

        upper_lip = pt(13)
        lower_lip = pt(14)
        left_mouth = pt(78)
        right_mouth = pt(308)

        mouth_height = math.dist(upper_lip, lower_lip)
        mouth_width = math.dist(left_mouth, right_mouth)

        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.0

        return {
            "face_visible": True,
            "mouth_ratio": mouth_ratio,
            "landmarks": landmarks
        }