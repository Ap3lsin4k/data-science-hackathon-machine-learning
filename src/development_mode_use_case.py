import numpy as np

from src.task1_model import MoodPredictionModel


class DevelopmentReleaseModeUseCase(object):

    def __init__(self, model=MoodPredictionModel()):
        pass

    def train(self, model, training_review, training_sentiment):
        self.model = model
        model.fit(training_review, training_sentiment)
        pass

    def predict(self, testing_reviews):
        return self.model.predict(testing_reviews)

    def compute_accuracy(self, param, param1):
        return np.mean(param == param1)

    def development_mode_halve_data_compute_accuracy_on_training_data(self, training_reviews_path):
        pass
