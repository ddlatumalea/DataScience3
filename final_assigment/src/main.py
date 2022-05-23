"""Main file to create different model results.
It evaluates models and different data representations."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from models.models import SimpleModels, SmoteModels
from parsers.parsers import Transform
from utils.paths import get_data_dir, get_results_dir

models = ['lr', 'gb', 'svm', 'tree', 'knn', 'rf', 'grad_b', 'ada_b']


if __name__ == '__main__':
    X_FILE = 'X_train.pkl'
    Y_FILE = 'y_train.pkl'
    X_PATH = str(Path(get_data_dir(), X_FILE))
    Y_PATH = str(Path(get_data_dir(), Y_FILE))

    RES_FILE = 'results_basemodels_del.csv'
    RES_PATH = str(Path(get_results_dir(), RES_FILE))

    X, y = pd.read_pickle(X_PATH), pd.read_pickle(Y_PATH)
    y = y.values

    cv = StratifiedKFold(n_splits=5)
    score_collection = {}

    no_corr_cols = ['MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'RPDE', 'DFA', 'spread2','D2', 'PPE']
    pca_transformed_cols = ['MDVP:Shimmer(dB)', 'MDVP:Fo(Hz)', 'DFA', 'RPDE', 'spread2',
       'MDVP:Flo(Hz)', 'D2']
    pca_raw_cols = ['MDVP:Shimmer(dB)', 'MDVP:Fo(Hz)', 'DFA', 'MDVP:Jitter(Abs)',
        'spread2', 'MDVP:Fhi(Hz)', 'D2']

    X_raw = X.values
    X_no_corr = X[no_corr_cols].values
    X_transformed_pca = Transform().box_cox_transform(X=X[pca_transformed_cols])
    X_pca = X[pca_raw_cols].values

    simple_models = SimpleModels(model_names=models)
    smote_models = SmoteModels(model_names=models)

    # simple models
    datasets = [X_raw, X_no_corr, X_transformed_pca, X_pca]
    data_types_simple = ['raw data -- simple', 'no corr -- simple',
                        'pca transformed -- simple', 'pca -- simple']
    for name, data_X in zip(data_types_simple, datasets):
        scores = simple_models.cross_val_score(data_X, y, cv)
        score_collection[name] = scores

    # smote models
    data_types_smote = ['raw data -- smote', 'no corr -- smote',
                        'pca transformed -- smote', 'pca -- smote']
    for name, data_X in zip(data_types_smote, datasets):
        scores = smote_models.cross_val_score(data_X, y, cv)
        score_collection[name] = scores


    result = pd.concat(score_collection, names=['data type'])
    print(result)

    result.to_csv(RES_PATH)
