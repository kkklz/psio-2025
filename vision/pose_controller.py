import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- KONFIGURACJA ---
MODEL_PATH = '../pose_landmarker_full.task'
VIDEO_FRONT = './wypad_front.mp4'
VIDEO_SIDE = './wypad_side.mp4'
OUTPUT_CSV = 'dane_3d_class.csv'


class PoseController:
    def __init__(self, model_asset_path):
        """
        Inicjalizacja kontrolera. Tworzy dwie niezależne instancje Landmarkera
        dla uniknięcia konfliktów timestampów.
        """
        self.model_path = model_asset_path

        if not os.path.exists(model_asset_path):
            raise FileNotFoundError(f"Nie znaleziono modelu: {model_asset_path}")

        # Skróty do klas MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # --- FIX: OSOBNE OPCJE ---
        # Tworzymy osobne konfiguracje dla obu detektorów
        options_front = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1
        )

        options_side = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1
        )

        print("Inicjalizacja modeli MediaPipe...")
        self.landmarker_front = PoseLandmarker.create_from_options(options_front)
        self.landmarker_side = PoseLandmarker.create_from_options(options_side)

    def detect(self, frame_front, frame_side, timestamp_ms):
        """
        Przetwarza dwie klatki (front i bok) i zwraca połączone dane 3D.

        Zwraca krotkę: (lista_punktów_3d, wynik_front, wynik_side)
        Gdzie lista_punktów_3d to lista słowników: {'id': int, 'x': float, 'y': float, 'z': float}
        """
        # Konwersja BGR (OpenCV) -> RGB (MediaPipe) -> mp.Image
        mp_image_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB))
        mp_image_s = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB))

        # Detekcja
        result_f = self.landmarker_front.detect_for_video(mp_image_f, timestamp_ms)
        result_s = self.landmarker_side.detect_for_video(mp_image_s, timestamp_ms)

        points_3d = []

        # Jeśli wykryto postać na obu nagraniach, łączymy dane
        if result_f.pose_landmarks and result_s.pose_landmarks:
            landmarks_front = result_f.pose_landmarks[0]
            landmarks_side = result_s.pose_landmarks[0]

            for i in range(len(landmarks_front)):
                # Logika 3D:
                # X, Y -> z kamery Front
                # Z -> z kamery Side (oś X na obrazie bocznym to głębokość)

                # UWAGA: Jeśli na kamerze bocznej postać patrzy w lewo, oś X rośnie w głąb sceny.
                # Jeśli patrzy w prawo, może być konieczne odwrócenie: z = 1.0 - landmarks_side[i].x

                point_data = {
                    'id': i,
                    'x': landmarks_front[i].x,
                    'y': landmarks_front[i].y,
                    'z': landmarks_side[i].x
                }
                points_3d.append(point_data)

        return points_3d, result_f, result_s

    def draw_landmarks(self, image, detection_result):
        """
        Rysuje punkty na podanym obrazie (w miejscu).
        Zwraca obraz z naniesionymi punktami.
        """
        annotated_image = np.copy(image)
        pose_landmarks_list = detection_result.pose_landmarks

        # Jeśli nic nie wykryto, zwracamy oryginał
        if not pose_landmarks_list:
            return annotated_image

        height, width, _ = annotated_image.shape

        # Rysowanie punktów dla pierwszej osoby
        for idx, landmark in enumerate(pose_landmarks_list[0]):
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            # Rysujemy żółte kółko
            cv2.circle(annotated_image, (cx, cy), 4, (255, 255, 0), -1)

        return annotated_image

    def close(self):
        """Zamyka instancje MediaPipe i zwalnia zasoby."""
        self.landmarker_front.close()
        self.landmarker_side.close()