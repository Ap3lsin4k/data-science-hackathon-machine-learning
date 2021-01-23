import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.task1_model import MoodPredictionModel


class DevelopmentReleaseModeUseCase(object):

    def __init__(self, model=MoodPredictionModel()):
        self.model = model

    def compute_accuracy_using_testing_data_from_train_csv(self):
        train_csv = pd.read_csv("../src/model/data/train.csv", index_col='id')

        numpy_array = train_csv.to_numpy()
        review_train = numpy_array[:, 0] # aka X_train
        self.sentiment_train = numpy_array[:, 1].astype('int') # aka Y_train

        X_train, X_test, Y_train, Y_test = train_test_split(
            review_train, self.sentiment_train, test_size=0.9, random_state=42)


        self.train(self.model, X_train, Y_train)
        self.compute_accuracy(self.predict(X_test), Y_test)

    def train(self, model, training_review, training_sentiment):
        self.model = model
        model.deprecated_fit(training_review, training_sentiment)

    def fit(self):
        self.model.auto_fit()

    def predict(self, testing_reviews):
        return self.model.predict(testing_reviews)

    def compute_accuracy(self, param, param1):
        print(np.mean(param == param1))
        return np.mean(param == param1)

    def development_mode_halve_data_compute_accuracy_on_training_data(self, training_reviews_path):
        pass
