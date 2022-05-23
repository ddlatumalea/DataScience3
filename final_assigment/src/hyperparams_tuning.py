"""Module that hypertunes the model."""

import pickle
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils.paths import get_model_saves_dir, get_data_dir
from model_tuning.utils import get_random_grid, randomized_search

if __name__ == '__main__':
    data_dir = get_data_dir()
    model_saves_dir = get_model_saves_dir()

    X_train = pd.read_pickle(str(Path(data_dir, 'X_train.pkl')))
    y_train = pd.read_pickle(str(Path(data_dir, 'y_train.pkl')))
    X_test = pd.read_pickle(str(Path(data_dir, 'X_test.pkl')))
    y_test = pd.read_pickle(str(Path(data_dir, 'y_test.pkl')))

    # get the data without correlation
    no_corr_cols = ['MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'RPDE', 'DFA', 'spread2','D2', 'PPE']
    X_no_corr_train = X_train[no_corr_cols]
    X_no_corr_test = X_test[no_corr_cols]

    params = get_random_grid()
    rf = RandomForestClassifier()
    model = randomized_search(rf, params, 'recall', X_no_corr_train, y_train)

    best_model = model.best_estimator_
    ypred = best_model.predict(X_no_corr_test)

    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))

    pickle.dump(best_model, open(str(Path(model_saves_dir, 'best_model.sav')), 'wb'))
