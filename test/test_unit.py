from src.task1 import *

def test_really_short_positive_comment():
    mood = MoodGuessUseCase()
    assert mood.is_positive('The film does a WONDERFUL job in creating a very "spooky atmosphere". THIS film is a MUST!') == 1


def test_really_short_negative_comment_contains_bad():
    mood = MoodGuessUseCase()
    assert mood.is_positive("""it's bad.""") == 0


def test_really_short_negative_comment_contains_rubbish():
    mood = MoodGuessUseCase()
    assert mood.is_positive("Absolute rubbish.") == 0


def test_clean_from_br