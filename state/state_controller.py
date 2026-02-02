from enum import Enum, auto

class AppState(Enum):
    IDLE = auto()
    EXERCISE = auto()
    FINISHED = auto()

class ExerciseType(Enum):
    SHOULDER_MOBILITY = auto()
    ASLR = auto()

class StateMachine:
    def __init__(self, exercise: ExerciseType):
        self.state = AppState.IDLE
        self.exercise = exercise

    def update(self, command: str | None):
        if not command:
            return

        command = command.lower()

        if self.state == AppState.IDLE:
            if "start" in command:
                self.state = AppState.EXERCISE
        elif self.state == AppState.EXERCISE:
            if "stop" in command:
                self.state = AppState.FINISHED
        elif self.state == AppState.FINISHED:
            if "start" in command:
                self.state = AppState.EXERCISE

    def is_idle(self):
        return self.state == AppState.IDLE

    def is_exercising(self):
        return self.state == AppState.EXERCISE

    def is_finished(self):
        return self.state == AppState.FINISHED

    def is_shoulder_test(self):
        return self.exercise == ExerciseType.SHOULDER_MOBILITY

    def is_aslr_test(self):
        return self.exercise == ExerciseType.ASLR

    def reset(self):
        self.state = AppState.IDLE