import pandas
import pytest

from src.presentation import present, control






def test_remove():
    import os
    os.remove("filename.txt")

def test_path():
    with open("../src/filename.txt", "w") as f:
        pass


def test_open_csv():
 #   control()
    present()
    with open("E:/dstesttask1/submission.csv") as f:
        assert len(f.read()) == 68903

    submission = pandas.read_csv("E:/dstesttask1/submission.csv")
    assert submission['sentiment'].value_counts() == 9999