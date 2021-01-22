from src.task1 import *

mood = MoodGuessUseCase()

# TODO Use case doesn't know about algorithm we are using, and we can easily change between the two
# TODO have ability to pickle neural network
# TODO two modes: development mode for training and showing the accuracy, then showing what tests went wrong As a develper of ML I want to , so that I can
# TODO second mode: release mode
# TODO for different algorithms find the intersection of failing data to be able to analyse it

def test_really_short_positive_comment():
    assert mood.predict(['The film does a WONDERFUL job in creating a very "spooky atmosphere". THIS film is a MUST!'])[0] == 1


def test_really_short_negative_comment_contains_bad():
    assert mood.predict(["it's bad."])[0] == 0


def test_really_short_negative_comment_contains_rubbish():
    assert mood.is_positive(["Absolute rubbish."])[0] == 0


def test_something():
    import math
    def square(a):
        b = a.copy()
        for i in range(len(b)):
            b[i] *= b[i]
        return b

    def avg(A):
        return sum(A) / len(A)

    def de(A):
        print('avg(',(square(A)), ')=  ',avg(square(A)), '-', avg(A)**2)
        return math.sqrt(avg(square(A)) - avg(A) ** 2) / avg(A)

    assert de([30, 30]) == 0
    assert de([29, 30]) != 0
    # assert de([0, 30]) == 1
    # assert de([4, 317]) == .9750778816199377
    # assert de([4, 317, 480])  == 0.7397612136993055
    # assert de([4, 317, 480, 495, 537, 613, 643, 667, 704])  == 0.4167387645666981
    # assert de([4, 317, 480, 495, 537, 613, 643, 667, 704, 737]) == 0.4019249037181562
    assert de([4, 317-4, 480-317, 495-480, 537-495, 613-537, 643-613, 667-643, 704-667, 737-704, 754-737+4]) == 0.5173128473494533




