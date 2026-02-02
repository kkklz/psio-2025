import cv2
import csv
import time
from vision.camera import DualCamera
from vision.pose_controller import PoseController
from fms_algorithms import FMSAlgorithms
from fms_scorers import ASLRScorer, ShoulderScorer
from state.state_controller import StateMachine, ExerciseType
from voice_recognition import VoiceRecognizer

# --- KONFIGURACJA ---
MODEL_PATH = './pose_landmarker_heavy.task'
VIDEO_FRONT = './wyprost_front.mp4'
VIDEO_SIDE = './wyprost_side.mp4'
RESULTS_CSV = 'wyniki_cwiczen.csv'
LANDMARKS_CSV = 'punkty_3d.csv'

# Wybór testu: 'ASLR' lub 'SHOULDER'
CURRENT_TEST = 'ASLR'


def main():
    voice_recognizer = None
    pose_controller = None
    dcamera = None
    try:
        pose_controller = PoseController(MODEL_PATH)
        fms_algo = FMSAlgorithms()
        voice_recognizer = VoiceRecognizer()
        voice_recognizer.start()

        if CURRENT_TEST == 'ASLR':
            state_machine = StateMachine(ExerciseType.ASLR)
            scorer = ASLRScorer()
        else:
            state_machine = StateMachine(ExerciseType.SHOULDER_MOBILITY)
            scorer = ShoulderScorer()

        dcamera = DualCamera(front_src=VIDEO_FRONT, side_src=VIDEO_SIDE)

        hand_length = 0.0
        calibration_frames = 0

        with open(RESULTS_CSV, mode='w', newline='') as results_file, \
             open(LANDMARKS_CSV, mode='w', newline='') as landmarks_file:

            results_writer = csv.writer(results_file)
            landmarks_writer = csv.writer(landmarks_file)

            results_writer.writerow(['timestamp', 'test_type', 'side', 'score', 'pain'])
            landmarks_writer.writerow(['timestamp', 'landmark_id', 'x', 'y', 'z'])

            print(f"Rozpoczynam analizę: {CURRENT_TEST}...")
            print("Powiedz 'start', aby rozpocząć, i 'stop', aby zakończyć.")

            frame_index = 0

            while True:
                command = voice_recognizer.get_command()
                state_machine.update(command)

                ret_f, frame_f, ret_s, frame_s = dcamera.read()
                if not ret_f or not ret_s:
                    break

                current_time = frame_index * 33  # ok. 30 fps

                points_3d, res_f, res_s = pose_controller.detect(frame_f, frame_s, current_time)

                vis_f = pose_controller.draw_landmarks(frame_f, res_f)
                vis_s = pose_controller.draw_landmarks(frame_s, res_s)

                if points_3d:
                    for point in points_3d:
                        landmarks_writer.writerow([current_time, point['id'], point['x'], point['y'], point['z']])

                if state_machine.is_exercising() and points_3d:
                    if calibration_frames < 10:
                        pass

                    if state_machine.is_shoulder_test():
                        if hand_length > 0:
                            score = fms_algo.analyze_shoulder_mobility(points_3d, hand_length)
                            scorer.add_attempt("right", score, pain_detected=False)
                            results_writer.writerow([current_time, CURRENT_TEST, "right", score, False])

                            feedback = "Dlonie niewystarczajaco blisko"
                            if score == 2:
                                feedback = "Dlonie powinny byc blizej"
                            if score == 3:
                                feedback = "Doskonale!"

                            cv2.putText(vis_f, f"Ocena: {feedback}", (40, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    elif state_machine.is_aslr_test():
                        # prawa
                        score_r = fms_algo.analyze_aslr(points_3d, side="right")
                        compensation_r = fms_algo.check_aslr_compensation(points_3d, side="right")
                        final_score_r = 1 if compensation_r else score_r
                        scorer.add_attempt("right", final_score_r, pain_detected=False)
                        results_writer.writerow([current_time, CURRENT_TEST, "right", final_score_r, False])
                        cv2.putText(vis_f, f"Ocena ASLR (prawa noga): {final_score_r}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # lewa
                        score_l = fms_algo.analyze_aslr(points_3d, side="left")
                        compensation_l = fms_algo.check_aslr_compensation(points_3d, side="left")
                        final_score_l = 1 if compensation_l else score_l
                        scorer.add_attempt("left", final_score_l, pain_detected=False)
                        results_writer.writerow([current_time, CURRENT_TEST, "left", final_score_l, False])
                        cv2.putText(vis_f, f"Ocena ASLR (lewa noga): {final_score_l}", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        if compensation_r and score_r != 1 and final_score_l == 1:
                            cv2.putText(vis_f, "Lewa noga musi byc prosta!", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if compensation_l and score_l != 1 and final_score_r == 1:
                            cv2.putText(vis_f, "Prawa noga musi byc prosta!!", (50, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Front', vis_f)
                cv2.imshow('Side', vis_s)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_index += 1

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
    finally:
        if voice_recognizer:
            voice_recognizer.stop()
        if pose_controller:
            pose_controller.close()
        if dcamera:
            dcamera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()