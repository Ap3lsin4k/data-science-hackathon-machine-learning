import pandas as pd

from src.task1 import MoodGuessUseCase


def present(param):
    submission = pd.read_csv("D:/projects/ds/data-science-hackathon-machine-learning/submission.csv")
    submission['sentiment'] = param
    submission.to_csv("E:/dstesttask1/submission.csv", index=False)


def control():
    sdata = pd.read_csv("E:/dstesttask1/test.csv", index_col='id')
    # test.csv
    import numpy as np
    numpy_array = sdata.to_numpy()
    reviews_test = numpy_array[:, 0] # 0 is reviews

    # Res
    mood = MoodGuessUseCase()

    mood.predict(reviews_test)
    present(mood.predicted)