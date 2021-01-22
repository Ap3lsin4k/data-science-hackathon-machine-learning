import os.path


def test_default_submission_csv_file_exists():
    assert os.path.isfile("../src/model/data/defaultsubmission.csv")
    assert os.path.isfile("../src/model/data/test.csv")
    assert os.path.isfile("../src/model/data/train.csv")


def test_default_submission_csv_is_not_corrupted():
    with open("E:/dstesttask1/defaultsubmission.csv") as f:
        assert len(f.read()) == 68903