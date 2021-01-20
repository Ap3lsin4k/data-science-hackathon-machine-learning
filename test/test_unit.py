from src.task1 import *

mood = MoodGuessUseCase()


def test_really_short_positive_comment():
    assert mood.predict(['The film does a WONDERFUL job in creating a very "spooky atmosphere". THIS film is a MUST!'])[0] == 1


def test_really_short_negative_comment_contains_bad():
    assert mood.predict(["it's bad."])[0] == 0


def test_really_short_negative_comment_contains_rubbish():
    assert mood.is_positive(["Absolute rubbish."])[0] == 0


def test_something():

    pass