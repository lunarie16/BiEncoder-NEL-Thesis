class CalculateMetrics:
    def __init__(self, tp: int, fp: int, fn: int):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.recall = 0
        self.precision = 0
        self.f1_score = 0
        self.accuracy = 0

    def calc_precision(self):
        try:
            self.precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            self.precision = 0
        return self.precision

    def calc_recall(self):
        try:
            self.recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            self.recall = 0
        return self.recall

    def calc_f1_score(self):
        try:
            self.f1_score = 2 * ((self.precision * self.recall) / (self.precision + self.recall))
        except ZeroDivisionError:
            self.f1_score = 0
        return self.f1_score

    def get_all(self):
        self.calc_precision()
        self.calc_recall()
        self.calc_f1_score()
        return self.precision, self.recall, self.f1_score
