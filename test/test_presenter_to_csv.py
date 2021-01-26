import pandas
import pytest

from src.presentation import present, control

import os

from datetime import datetime
import tempfile


def test_release_new_submission_csv_with_unique_name():
    # create a temporary directory
    with tempfile.TemporaryDirectory() as directory:
        newsubmissionpath = directory + "released submission -1 00.00.00.csv"

        assert not os.path.isfile(newsubmissionpath)
        present([1, 1, 0, 1, 0]*2000, newsubmissionpath)
        assert os.path.isfile(newsubmissionpath)