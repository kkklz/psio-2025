class FMSBaseScorer:

    def __init__(self, name):
        self.name = name
        self.results = {"left": [], "right": []}
        self.pain = {"left": False, "right": False}

    def add_attempt(self, side, score, pain_detected=False):

        if len(self.results[side]) < 3:
            self.results[side].append(score)

        if pain_detected:
            self.pain[side] = True

    def get_side_score(self, side):
        if self.pain[side]:
            return 0
        if not self.results[side]:
            return 1
        return max(self.results[side])

    def get_final_score(self):
        left_best = self.get_side_score("left")
        right_best = self.get_side_score("right")
        return min(left_best, right_best)


class ShoulderScorer(FMSBaseScorer):
    def __init__(self):
        super().__init__("Shoulder Mobility")


class ASLRScorer(FMSBaseScorer):
    def __init__(self):
        super().__init__("Active Straight Leg Raise")