def test_default_submission_csv_is_not_corrupt():
    with open("E:/dstesttask1/defaultsubmission.csv") as f:
        assert len(f.read()) == 68903