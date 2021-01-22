import pandas as pd

from src.task1 import MoodGuessUseCase


def present(sentiment):
    submission = pd.read_csv("E:/dstesttask1/defaultsubmission.csv")
    submission['sentiment'] = sentiment
    submission.to_csv("E:/dstesttask1/submission.csv", index=False)


def control():

    # Res
    mood = MoodGuessUseCase()

    test = pd.read_csv("E:/dstesttask1/test.csv", index_col='id')

    mood.predict(test['review'].values)
    present(mood.predicted)