import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- KONFIGURACJA ---
MODEL_PATH = './pose_landmarker_full.task'
VIDEO_FRONT = './wyprost_front.mp4'
VIDEO_SIDE = './wyprost_side.mp4'
OUTPUT_CSV = 'dane_3d.csv'

# Skróty
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    if not pose_landmarks_list:
        return annotated_image

    height, width, _ = annotated_image.shape
    for idx, landmark in enumerate(pose_landmarks_list[0]):
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(annotated_image, (cx, cy), 4, (255, 255, 0), -1)
    return annotated_image


def main():
    if not os.path.exists(MODEL_PATH):
        print('Model not existing: ' + MODEL_PATH)
        return

    # --- FIX 1: OSOBNE OPCJE DLA KAŻDEGO DETEKTORA ---
    # Musimy utworzyć dwa niezależne obiekty opcji, żeby nie współdzieliły stanu w C++

    options_front = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    options_side = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    # Tworzymy detektory z OSOBNYCH opcji
    print("Inicjalizacja modeli...")
    landmarker_front = PoseLandmarker.create_from_options(options_front)
    landmarker_side = PoseLandmarker.create_from_options(options_side)

    cap_f = cv2.VideoCapture(VIDEO_FRONT)
    cap_s = cv2.VideoCapture(VIDEO_SIDE)

    if not cap_f.isOpened() or not cap_s.isOpened():
        print('Video not opened')
        return

    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_ms', 'landmark_index', 'x', 'y', 'z'])

        fps = cap_f.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        frame_duration_ms = 1000.0 / fps

        frame_index = 0

        # --- FIX 2: STARTUJEMY OD 100ms ---
        # Czasem timestamp 0 sprawia problemy, bezpieczniej zacząć od wartości dodatniej
        start_offset = 100

        print(f"Rozpoczynam (FPS: {fps:.2f})")

        while True:
            ret_f, frame_f = cap_f.read()
            ret_s, frame_s = cap_s.read()

            if not ret_f or not ret_s:
                break

            mp_image_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB))
            mp_image_s = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB))

            # Obliczamy czas ręcznie + offset
            current_timestamp_ms = int(start_offset + (frame_index * frame_duration_ms))

            try:
                # Teraz, gdy opcje są rozdzielone, wywołanie z tym samym timestampem
                # dla dwóch różnych obiektów nie spowoduje błędu
                result_f = landmarker_front.detect_for_video(mp_image_f, current_timestamp_ms)
                result_s = landmarker_side.detect_for_video(mp_image_s, current_timestamp_ms)

            except Exception as e:
                # Wypisujemy błąd, ale nie przerywamy pętli całkowicie, próbujemy następną klatkę
                print(f"Błąd w klatce {frame_index} (ts: {current_timestamp_ms}): {e}")
                frame_index += 1
                continue

            if result_f.pose_landmarks and result_s.pose_landmarks:
                landmarks_front = result_f.pose_landmarks[0]
                landmarks_side = result_s.pose_landmarks[0]

                for i in range(len(landmarks_front)):
                    x_val = landmarks_front[i].x
                    y_val = landmarks_front[i].y
                    z_val = landmarks_side[i].x

                    writer.writerow([current_timestamp_ms, i, x_val, y_val, z_val])

            # Wizualizacja
            vis_f = draw_landmarks_on_image(cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB), result_f)
            vis_s = draw_landmarks_on_image(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB), result_s)

            cv2.imshow('Kamera Front', cv2.cvtColor(vis_f, cv2.COLOR_RGB2BGR))
            cv2.imshow('Kamera Side', cv2.cvtColor(vis_s, cv2.COLOR_RGB2BGR))

            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    landmarker_front.close()
    landmarker_side.close()
    cap_f.release()
    cap_s.release()
    cv2.destroyAllWindows()
    print("Zakończono.")


if __name__ == '__main__':
    main()