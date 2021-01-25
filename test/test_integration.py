import pandas as pd
import pytest

from src.heavy_model import HeavyModel
from src.presentation import control


@pytest.mark.skip("Run this when you want to kill some time")
def test_integration_whole_system():
    train = pd.read_csv("../kaggle/input/train.csv", index_col='id')

    control(HeavyModel(train_csv=train))
