import math
import numpy as np


class FMSAlgorithms:

    def __init__(self):
        # Indeksy MediaPipe Pose
        self.L_WRIST = 15
        self.R_WRIST = 16
        self.L_HIP = 23
        self.R_HIP = 24
        self.L_KNEE = 25
        self.R_KNEE = 26
        self.L_ANKLE = 27
        self.R_ANKLE = 28
        self.L_INDEX = 19
        self.R_INDEX = 20  # Do obliczania długości dłoni

    def calculate_distance_3d(self, p1, p2):
        return math.sqrt(
            (p2['x'] - p1['x']) ** 2 +
            (p2['y'] - p1['y']) ** 2 +
            (p2['z'] - p1['z']) ** 2
        )

    def calculate_hand_length(self, landmarks):
        left_len = self.calculate_distance_3d(landmarks[self.L_WRIST], landmarks[self.L_INDEX])
        right_len = self.calculate_distance_3d(landmarks[self.R_WRIST], landmarks[self.R_INDEX])
        return (left_len + right_len) / 2

    def calculate_angle(self, a, b, c):
        vec_ba = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
        vec_bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])

        norm_ba = np.linalg.norm(vec_ba)
        norm_bc = np.linalg.norm(vec_bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def analyze_shoulder_mobility(self, landmarks, hand_length):
        if hand_length == 0:
            return 1

        dist = self.calculate_distance_3d(landmarks[self.L_WRIST], landmarks[self.R_WRIST])

        # Logika punktacji FMS
        if dist <= hand_length * 4:
            return 3
        elif dist <= 1.5 * hand_length * 4:
            return 2
        else:
            return 1

    def analyze_aslr(self, landmarks, side="right"):
        if side == "right":
            raised_ankle = landmarks[self.R_ANKLE]
            stable_hip = landmarks[self.L_HIP]
            stable_knee = landmarks[self.L_KNEE]
        else:
            raised_ankle = landmarks[self.L_ANKLE]
            stable_hip = landmarks[self.R_HIP]
            stable_knee = landmarks[self.R_KNEE]

        dist_hip_to_knee = abs(stable_hip['z'] - stable_knee['z'])
        dist_hip_to_mid_thigh = dist_hip_to_knee / 2

        dist_hip_to_ankle_proj = abs(stable_hip['z'] - raised_ankle['z'])

        if dist_hip_to_ankle_proj <= dist_hip_to_mid_thigh*1.5:
            return 3
        elif dist_hip_to_ankle_proj <= dist_hip_to_knee*1.5:
            return 2
        else:
            return 1

    def check_aslr_compensation(self, landmarks, side="right"):
        check_side = "left" if side == "right" else "right"

        knee_idx = self.R_KNEE if check_side == "right" else self.L_KNEE
        hip_idx = self.R_HIP if check_side == "right" else self.L_HIP
        ankle_idx = self.R_ANKLE if check_side == "right" else self.L_ANKLE

        angle = self.calculate_angle(landmarks[hip_idx], landmarks[knee_idx], landmarks[ankle_idx])

        return abs(180 - angle) > 20