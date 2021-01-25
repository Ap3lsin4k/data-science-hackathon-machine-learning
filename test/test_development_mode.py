import numpy as np

from src.development_mode_use_case import DevelopmentReleaseModeUseCase


def test_nothing():
    c = DevelopmentReleaseModeUseCase()


    assert c.compute_accuracy(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1])) == 0.75
    assert c.compute_accuracy(np.array([0, 0, 1, 1, 0]), np.array([0, 0, 1, 1, 1])) == 4/5
    assert c.compute_accuracy(np.array([0, 0, 1, 1, 0, 0, 0]), np.array([0, 1, 0, 0, 1, 1, 1])) == 1/7

    # train = pd.read_csv("E:/dstesttask1/train.csv", index_col='id')
    # numpy_array = train_csv.to_numpy()
    # review_train = numpy_array[:, 0] # aka X_train
    # self.sentiment_train = numpy_array[:, 1].astype('int') # aka Y_train

    # X_train, X_test, Y_train, Y_test = train_test_split(
    # X, Y, test_size=0.4, random_state=42)
    c.development_mode_halve_data_compute_accuracy_on_training_data("../test/model/data/dummytrain.csv")
