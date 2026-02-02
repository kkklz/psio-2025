import cv2
import csv

from vision.camera import DualCamera
from vision.pose_controller import PoseController

# --- KONFIGURACJA ---
MODEL_PATH = './pose_landmarker_full.task'
VIDEO_FRONT = './wyprost_front.mp4'
VIDEO_SIDE = './wyprost_side.mp4'
OUTPUT_CSV = 'dane_3d_class.csv'


def main():
    # 1. Inicjalizacja kontrolera
    try:
        pose_controller = PoseController(MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Otwarcie plików wideo
    dcamera =  DualCamera(front_src=VIDEO_FRONT, side_src=VIDEO_SIDE)


    # 3. Przygotowanie pliku CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_ms', 'landmark_index', 'x', 'y', 'z'])

        # Pobieranie FPS
        fps = dcamera.front_cam.get_fps_count()
        frame_duration_ms = 1000.0 / fps

        frame_index = 0
        start_offset_ms = 100  # Bezpieczny start timestampu

        print(f"Rozpoczynam przetwarzanie (FPS: {fps:.2f})...")

        while True:
            # Pobranie klatek
            ret_f, frame_f, ret_s, frame_s = dcamera.read()

            if not ret_f or not ret_s:
                break

            # Obliczenie czasu
            current_timestamp_ms = int(start_offset_ms + (frame_index * frame_duration_ms))

            try:
                # --- UŻYCIE KLASY: DETEKCJA ---
                # Funkcja zwraca listę punktów 3D oraz surowe wyniki dla wizualizacji
                points_3d, result_f, result_s = pose_controller.detect(frame_f, frame_s, current_timestamp_ms)

                # Zapis do CSV
                if points_3d:
                    for p in points_3d:
                        writer.writerow([current_timestamp_ms, p['id'], p['x'], p['y'], p['z']])

                # --- UŻYCIE KLASY: RYSOWANIE ---
                vis_f = pose_controller.draw_landmarks(frame_f, result_f)
                vis_s = pose_controller.draw_landmarks(frame_s, result_s)

                # Wyświetlanie
                cv2.imshow('Kamera Front', vis_f)
                cv2.imshow('Kamera Side', vis_s)

            except Exception as e:
                print(f"Błąd w klatce {frame_index}: {e}")
                frame_index += 1
                continue

            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Sprzątanie
    pose_controller.close()
    dcamera.release()
    cv2.destroyAllWindows()
    print("Zakończono.")


if __name__ == '__main__':
    main()