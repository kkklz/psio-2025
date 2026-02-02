import speech_recognition as sr
import threading
import queue

class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()
        self.stop_listening = None
        self.thread = threading.Thread(target=self._listen)

    def _listen(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.stop_listening = self.recognizer.listen_in_background(
            self.microphone, self._audio_callback
        )

    def _audio_callback(self, recognizer, audio):
        try:
            command = recognizer.recognize_google(audio, language="pl-PL").lower()
            print(f"Rozpoznano komendę: {command}")
            self.command_queue.put(command)
        except sr.UnknownValueError:
            pass  # Ignoruj, jeśli nie można zrozumieć mowy
        except sr.RequestError as e:
            print(f"Błąd API Google Speech Recognition; {e}")

    def start(self):
        self.thread.start()

    def stop(self):
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
        self.thread.join()

    def get_command(self):
        if not self.command_queue.empty():
            return self.command_queue.get()
        return None