from src.development_mode_use_case import DevelopmentReleaseModeUseCase
from src.task1 import MoodPredictionModel


class StubModel(object):

    training_review: str
    expected_sentiment: int

    def __init__(self):
        self.fit_was_called = False

    def fit(self, training_review, expected_sentiment):
        self.fit_was_called = True
        self.training_review = training_review[0]
        self.expected_sentiment = expected_sentiment[0]


def test_use_case_calls_model():
    c = DevelopmentReleaseModeUseCase(StubModel())

    model = StubModel()
    c.train(model, ["A DAMN GOOD MOVIE! I give it a 10/10."], [1])
    assert model.fit_was_called
    assert model.training_review == "A DAMN GOOD MOVIE! I give it a 10/10."
    assert model.expected_sentiment == 1
