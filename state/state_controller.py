from enum import Enum, auto


class AppState(Enum):
    IDLE = auto()        # czekamy na "start"
    EXERCISE = auto()    # wykonywanie ćwiczenia
    FINISHED = auto()    # zakończone -> wyniki


class ExerciseType(Enum):
    SHOULDER_MOBILITY = auto()
    LUNGE = auto()


class StateMachine:
    def __init__(self, exercise: ExerciseType):
        self.state = AppState.IDLE

        # to ustawiamy przy tworzeniu nowej instancji klasy w main.py, mozemy przekazywac cwiczenie jako np. parametr startowy
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

        # to mozna usunac, nie wiem czy chcemy zeby mozna bylo zrestestowac program bez fizycznego resetowania programu
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

    def is_lunge_test(self):
        return self.exercise == ExerciseType.LUNGE

    def reset(self):
        self.state = AppState.IDLE