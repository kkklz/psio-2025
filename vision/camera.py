import cv2
import os

class Camera:
    def __init__(self, src=0, name="Camera"):
        """
        src:
            - int  -> kamera (0, 1, ...)
            - str  -> ścieżka do pliku .mp4

        W naszym przypadku testowym inicjalizujemy klase DuelCamera w ten sposob:
        cams = DualCamera(
            front_src="front.mp4",
            side_src="side.mp4"
        )
        """
        self.src = src
        self.name = name

        if isinstance(src, str):
            if not os.path.exists(src):
                raise ValueError(f"Plik wideo nie istnieje: {src}")

        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise ValueError(f"Nie można otworzyć źródła wideo: {src}")

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return None # wtedy koniec wideo

        return frame

    def release(self):
        if self.cap:
            self.cap.release()

    def show(self, frame):
        if frame is not None:
            cv2.imshow(self.name, frame)


class DualCamera:
    def __init__(self, front_src=0, side_src=1):
        self.front_cam = Camera(front_src, "Front Camera")
        self.side_cam = Camera(side_src, "Side Camera")

    def read(self):
        front_frame = self.front_cam.read()
        side_frame = self.side_cam.read()
        return front_frame, side_frame

    def release(self):
        self.front_cam.release()
        self.side_cam.release()

    def show(self, front_frame, side_frame):
        self.front_cam.show(front_frame)
        self.side_cam.show(side_frame)