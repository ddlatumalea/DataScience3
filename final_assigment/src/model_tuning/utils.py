"""Some functions that are necessary for hyperparameter tuning."""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def get_random_grid():
    """Returns a grid with the configuration for randomized search."""
    n_estimators = np.arange(0, 2000, 100)
    max_features = ['sqrt', 'log2', None]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    return random_grid

def randomized_search(estimator, param_distributions, scoring, X, y):
    """Uses randomized search to find the best parameter combination of the model.
    
    Keyword arguments:
    estimator -- the model
    param_distributions -- the search space to use for random search
    scoring -- the scoring metric that must be optimized
    X -- the training features
    y -- the training labels
    """
    random_model = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions, scoring=scoring, n_iter=100, cv=5, verbose=1, n_jobs=-1)
    random_model.fit(X, y)
    return random_model
