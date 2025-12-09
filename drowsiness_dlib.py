import cv2
import dlib
import numpy as np
import platform
import subprocess
import time

# ================================
# Utility functions
# ================================
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # MAR = (|p51 - p57| + |p50 - p58| + |p52 - p56|) / (3 * |p48 - p54|)
    A = euclidean_distance(mouth[2], mouth[10])  # 51, 57
    B = euclidean_distance(mouth[1], mouth[11])  # 50, 58
    C = euclidean_distance(mouth[3], mouth[9])   # 52, 56
    D = euclidean_distance(mouth[0], mouth[6])   # 48, 54
    mar = (A + B + C) / (3.0 * D)
    return mar

# ================================
# Head pose estimation (pitch)
# ================================
def get_head_pose(shape_np, frame_shape):
    h, w = frame_shape[:2]

    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye left
        (43.3, 32.7, -26.0),      # Right eye right
        (-28.9, -28.9, -24.1),    # Left mouth corner
        (28.9, -28.9, -24.1)      # Right mouth corner
    ], dtype=np.float64)

    image_points = np.array([
        shape_np[30],  # Nose tip
        shape_np[8],   # Chin
        shape_np[36],  # Left eye left
        shape_np[45],  # Right eye right
        shape_np[48],  # Left mouth
        shape_np[54]   # Right mouth
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])   # pitch
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])

    pitch = np.degrees(x)
    return pitch

# ================================
# Alert / Sound (separate messages)
# ================================
def play_alarm(reason="sleep"):
    system = platform.system()

    if reason == "phone":
        # Phone-use specific warning
        message = "Driving karte waqt phone ka istemal mat kijiye. Do not use phone while driving."
    else:
        # Sleep / eyes closed / yawning / head-down warning
        message = "Driving karte waqt mat soiye. Do not sleep while driving."

    try:
        if system == "Darwin":  # macOS
            subprocess.Popen(["say", message])
        elif system == "Windows":
            import winsound
            winsound.Beep(2500, 700)
            print("ALERT:", message)
        else:
            print("ALERT:", message)
    except Exception as e:
        print("Error playing alarm:", e)

# ================================
# Thresholds and counters
# ================================
# Eyes
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 30        # ~1+ sec eyes closed

# Yawn
MAR_THRESHOLD = 0.80
YAWN_CONSEC_FRAMES = 25

# Head down – stricter
PITCH_DOWN_THRESHOLD = 40.0   # strong down tilt
HEAD_CONSEC_FRAMES = 35
FACE_CENTER_Y_MIN = 0.65      # face center must be lower in frame

eye_counter = 0
yawn_counter = 0
head_counter = 0

LAST_ALERT_TIME = 0
ALERT_COOLDOWN = 5            # seconds

# ================================
# Dlib models
# ================================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
MOUTH_IDX = list(range(48, 60))

# ================================
# Camera
# ================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    faces = detector(gray, 0)

    # ============================
    # Placeholder for phone detection
    # ============================
    # TODO: Replace this with your own phone detection logic
    phone_using = False

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])

        # -------- Eyes --------
        left_eye = shape_np[LEFT_EYE_IDX]
        right_eye = shape_np[RIGHT_EYE_IDX]

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if ear < EAR_THRESHOLD:
            eye_counter += 1
        else:
            eye_counter = 0

        # -------- Mouth / Yawn --------
        mouth = shape_np[MOUTH_IDX]
        mar = mouth_aspect_ratio(mouth)

        for (x, y) in mouth:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        if mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = 0

        # -------- Head Pose --------
        pitch = get_head_pose(shape_np, frame.shape)

        face_center_y = (face.top() + face.bottom()) / 2.0
        face_center_norm = face_center_y / float(h)  # 0 = top, 1 = bottom

        if pitch > PITCH_DOWN_THRESHOLD and face_center_norm > FACE_CENTER_Y_MIN:
            head_counter += 1
        else:
            head_counter = 0

        # -------- Drowsiness decision --------
        drowsy = False

        if eye_counter >= EAR_CONSEC_FRAMES:
            drowsy = True
        if yawn_counter >= YAWN_CONSEC_FRAMES:
            drowsy = True
        if head_counter >= HEAD_CONSEC_FRAMES:
            drowsy = True

        current_time = time.time()

        # ============================
        # Phone-use alert
        # ============================
        if phone_using:
            if current_time - LAST_ALERT_TIME > ALERT_COOLDOWN:
                play_alarm(reason="phone")
                LAST_ALERT_TIME = current_time

        # ============================
        # Sleep / drowsiness alert
        # ============================
        if drowsy:
            cv2.putText(frame, "DROWSINESS ALERT!!!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if current_time - LAST_ALERT_TIME > ALERT_COOLDOWN:
                play_alarm(reason="sleep")
                LAST_ALERT_TIME = current_time

        # Debug info
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FaceY: {face_center_norm:.2f}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0.0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Advanced Drowsiness Detector - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
        break

cap.release()
cv2.destroyAllWindows()
