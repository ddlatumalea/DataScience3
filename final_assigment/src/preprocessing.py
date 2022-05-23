"""Preprocesses the dataset in a training set and a testing set.
It uses a Stratified splitter to keep the samples balanced."""
from pathlib import Path
import pandas as pd

from utils.paths import get_data_dir

from splitters.splitters import StratifiedSplitter


if __name__ == '__main__':
    data_dir = get_data_dir()

    DATA = 'parkinsons.data'
    DATA_PATH = str(Path(data_dir, DATA))
    X_TRAIN_PATH = str(Path(data_dir, 'X_train.pkl'))
    X_TEST_PATH = str(Path(data_dir, 'X_test.pkl'))
    Y_TRAIN_PATH = str(Path(data_dir, 'y_train.pkl'))
    Y_TEST_PATH = str(Path(data_dir, 'y_test.pkl'))

    df = pd.read_csv(DATA_PATH, sep=',').set_index('name')

    splitter = StratifiedSplitter(df)
    X_train, X_test, y_train, y_test = splitter.split("status")

    X_train.to_pickle(X_TRAIN_PATH)
    X_test.to_pickle(X_TEST_PATH)
    y_train.to_pickle(Y_TRAIN_PATH)
    y_test.to_pickle(Y_TEST_PATH)
    