from datetime import datetime

import pandas as pd

from src.model.task1_model import MoodPredictionModel


def present(sentiment_column, submission_csv_path="model/data/newsubmission.csv"):
    submission = pd.read_csv("../src/model/data/defaultsubmission.csv")
    submission['sentiment'] = sentiment_column
    submission.to_csv(submission_csv_path, index=False)


def get_new_submission_path_with_version():
    return datetime.now().strftime("model/data/newsubmission %d %H;%M;%S.csv")  # day hour:minutes:seconds


def control(model):
    #model.fit()
    predicted_array_of_sentiments = model.predict()
    present(predicted_array_of_sentiments, get_new_submission_path_with_version())
