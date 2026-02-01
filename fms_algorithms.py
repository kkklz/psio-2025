import math
import numpy as np

class FMSAlgorithms:

    def __init__(self):
        self.L_WRIST = 15; self.R_WRIST = 16
        self.L_HIP = 23; self.R_HIP = 24
        self.L_KNEE = 25; self.R_KNEE = 26
        self.L_ANKLE = 27; self.R_ANKLE = 28

    def calculate_distance_3d(self, p1, p2):
        return math.sqrt(
            (p2['x'] - p1['x']) ** 2 +
            (p2['y'] - p1['y']) ** 2 +
            (p2['z'] - p1['z']) ** 2
        )

    def calculate_angle(self, a, b, c):
        vec_ba = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
        vec_bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])

        cosine_angle = np.dot(vec_ba, vec_bc) / (np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def analyze_shoulder_mobility(self, landmarks, hand_length):
        dist = self.calculate_distance_3d(landmarks[self.L_WRIST], landmarks[self.R_WRIST])
        if dist <= hand_length:
            return 3  # jak dłonie są mega blisko to masz 3 punkty
        elif dist <= 1.5 * hand_length:
            return 2  # jak są trochę dalej to dostajesz 2 punkty
        else:
            return 1  # jak jest duża przerwa to tylko 1 punkt

    def analyze_aslr(self, landmarks, side="right"):
        if side == "right":
            raised_ankle = landmarks[self.R_ANKLE]
            stable_hip = landmarks[self.L_HIP]
            stable_knee = landmarks[self.L_KNEE]
        else:
            raised_ankle = landmarks[self.L_ANKLE]
            stable_hip = landmarks[self.R_HIP]
            stable_knee = landmarks[self.R_KNEE]
        mid_thigh_x = (stable_hip['x'] + stable_knee['x']) / 2

        if raised_ankle['x'] < mid_thigh_x:
            return 3  # noga wysoko nad połową uda to max punktów
        elif raised_ankle['x'] < stable_knee['x']:
            return 2  # noga nad kolanem ale pod połową uda to 2 punkty
        else:
            return 1  # noga niziutko pod kolanem to tylko 1 punkt

    def check_aslr_compensation(self, landmarks, side="right"):
        knee_idx = self.R_KNEE if side == "right" else self.L_KNEE
        hip_idx = self.R_HIP if side == "right" else self.L_HIP
        ankle_idx = self.R_ANKLE if side == "right" else self.L_ANKLE
        angle = self.calculate_angle(landmarks[hip_idx], landmarks[knee_idx], landmarks[ankle_idx])

        # jak noga ugięta o więcej niż 10 stopni to znaczy że robisz błąd
        return abs(180 - angle) > 10