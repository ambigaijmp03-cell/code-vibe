import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import random

# 1. Setup Detectors
base_options_f = python.BaseOptions(model_asset_path='face_landmarker.task')
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base_options_f, running_mode=vision.RunningMode.VIDEO, num_faces=1)
)

base_options_h = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=base_options_h, running_mode=vision.RunningMode.VIDEO, num_hands=1)
)

# 2. State & Position Variables
# We start with the mesh at its 'Ghost' position on the left
current_x, current_y = 200, 200 
disintegration = 0.0
is_locked = True # Start locked (Solid)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    timestamp = int(time.time() * 1000)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    f_res = face_detector.detect_for_video(mp_image, timestamp)
    h_res = hand_detector.detect_for_video(mp_image, timestamp)

    # 3. ANCHOR LOGIC
    if h_res.hand_landmarks:
        lms = h_res.hand_landmarks[0]
        # Calculate pinch distance
        dist = np.sqrt((lms[4].x - lms[8].x)**2 + (lms[4].y - lms[8].y)**2)
        ix, iy = int(lms[8].x * w), int(lms[8].y * h)

        if dist > 0.08: 
            # SEPARATED: Track the hand and disintegrate
            is_locked = False
            current_x, current_y = ix, iy
            disintegration = min(disintegration + 2.0, 35.0)
            cv2.circle(frame, (ix, iy), 12, (0, 255, 255), 2)
        else: 
            # PINCHED: Stay exactly where you are and solidify
            is_locked = True
            disintegration = max(disintegration - 4.0, 0.0)
            # Visual feedback for the anchor point
            cv2.drawMarker(frame, (int(current_x), int(current_y)), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # 4. VOLUMETRIC RENDERING
    if f_res.face_landmarks:
        # We need the nose tip to calculate relative offsets for the 360 model
        nose = f_res.face_landmarks[0][1] 
        nw, nh = nose.x * w, nose.y * h

        for face_lms in f_res.face_landmarks:
            for i, lm in enumerate(face_lms):
                # Calculate how far this point is from the center of your real face
                rel_x = (lm.x * w) - nw
                rel_y = (lm.y * h) - nh
                z_depth = lm.z * 120 # Depth for jaw/ear noticeability
                
                # Place the point relative to the ANCHOR (current_x, current_y)
                cx = int(current_x + rel_x + (z_depth * (disintegration / 20)))
                cy = int(current_y + rel_y)
                
                # Apply 4D Disintegration Jitter
                if disintegration > 0:
                    cx += random.randint(-int(disintegration), int(disintegration))
                    cy += random.randint(-int(disintegration), int(disintegration))

                if 0 < cx < w and 0 < cy < h:
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)
                    
                    # Connection Lines for the Neat 360 Vision
                    if disintegration < 2 and i % 40 == 0:
                        next_lm = face_lms[(i + 1) % 478]
                        nx = int(current_x + (next_lm.x * w - nw))
                        ny = int(current_y + (next_lm.y * h - nh))
                        cv2.line(frame, (cx, cy), (nx, ny), (0, 100, 100), 1)

    # UI status
    status_text = "LOCKED / SOLID" if is_locked else "MOVING / 4D DUST"
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Cerebral Anchor System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break