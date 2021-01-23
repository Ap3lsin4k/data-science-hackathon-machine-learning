from src.development_mode_use_case import DevelopmentReleaseModeUseCase
from src.model.disperse_estimation import de
from src.task1 import *

# takes a long time to teach the neural network
# todo use dummy instead of the model to have
mood = DevelopmentReleaseModeUseCase(MoodPredictionModel())
mood.train(MoodPredictionModel(), )

# TODO two modes: development mode for training and showing the accuracy, then showing what tests went wrong As a develper of ML I want to , so that I can
# TODO Use case doesn't know about algorithm we are using, and we can easily change between the two how to check that this works: I don't have to run neural network to check that use case works
# second mode: release mode
# TODO for different algorithms find the intersection of failing data to be able to analyse it

def test_really_short_positive_comment():
    assert mood.predict(['The film does a WONDERFUL job in creating a very "spooky atmosphere". THIS film is a MUST!'])[0] == 1


def test_really_short_negative_comment_contains_bad():
    assert mood.predict(["it's bad."])[0] == 0


def test_really_short_negative_comment_contains_rubbish():
    assert mood.predict(["Absolute rubbish."])[0] == 0


def test_something():

    assert de([30, 30]) == 0
    assert de([29, 30]) != 0
    # assert de([0, 30]) == 1
    # assert de([4, 317]) == .9750778816199377
    # assert de([4, 317, 480])  == 0.7397612136993055
    # assert de([4, 317, 480, 495, 537, 613, 643, 667, 704])  == 0.4167387645666981
    # assert de([4, 317, 480, 495, 537, 613, 643, 667, 704, 737]) == 0.4019249037181562
    assert de([4, 317 - 4, 480 - 317, 495 - 480, 537 - 495, 613 - 537, 643 - 613, 667 - 643, 704 - 667, 737 - 704, 754 - 737 + 4]) == 0.5173128473494533




