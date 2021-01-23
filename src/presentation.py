from datetime import datetime

import pandas as pd

from src.task1 import MoodPredictionModel


def present(sentiment_column, submission_csv_path="model/data/newsubmission.csv"):
    submission = pd.read_csv("../src/model/data/defaultsubmission.csv")
    submission['sentiment'] = sentiment_column
    submission.to_csv(submission_csv_path, index=False)


def get_new_submission_path_with_version():
    return datetime.now().strftime("model/data/newsubmission %d %H;%M;%S.csv")  # day hour:minutes:seconds


def control():
    # Res
    mood = MoodPredictionModel()

    test = pd.read_csv("E:/dstesttask1/test.csv", index_col='id')

    mood.predict(test['review'].values)
    present(mood.predicted, get_new_submission_path_with_version())
