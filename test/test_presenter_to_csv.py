import pandas
import pytest

from src.task1 import present, control


def get_submission():
    pass

def test_nothing():
#    with pytest.raises(FileNotFoundError):
#        open("D:/projects/ds/data-science-hackathon-machine-learning/mysubmission.csv")
    present([1, 1, 0])
    with open("D:/projects/ds/data-science-hackathon-machine-learning/mysubmission.csv") as f:
        assert len(f.read()) == 68903





def test_open_csv():
    control()
    with open("E:/dstesttask1/submission.csv") as f:
        assert len(f.read()) == 68903

    submission = pandas.read_csv("E:/dstesttask1/submission.csv")
    assert submission['sentiment'].value_counts() == 9999