import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class CerebralEngine:
    def __init__(self, face_model, hand_model):
        # Face Setup
        base_face = python.BaseOptions(model_asset_path=face_model)
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base_face, running_mode=vision.RunningMode.VIDEO, num_faces=1)
        )
        # Hand Setup
        base_hand = python.BaseOptions(model_asset_path=hand_model)
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_hand, running_mode=vision.RunningMode.VIDEO, num_hands=1)
        )

    def process(self, mp_image, timestamp):
        face_res = self.face_detector.detect_for_video(mp_image, timestamp)
        hand_res = self.hand_detector.detect_for_video(mp_image, timestamp)
        return face_res, hand_res