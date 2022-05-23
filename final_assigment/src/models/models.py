"""Module to create multiple models that can be used
for fitting on data and getting scores."""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


class Models(ABC):
    """Abstract class to create several model pipelines.
    It also has concrete implementations. It is only necessary to implements
    the create_pipelines method."""
    mapping = {
        'lr': LogisticRegression,
        'gb': GaussianNB,
        'svm': SVC,
        'tree': DecisionTreeClassifier,
        'knn': KNeighborsClassifier,
        'rf': RandomForestClassifier,
        'grad_b': GradientBoostingClassifier,
        'ada_b': AdaBoostClassifier
    }

    def __init__(self, model_names: list) -> None:
        super().__init__()
        self.pipelines = self.create_pipelines(model_names)

    @abstractmethod
    def create_pipelines(self, keywords: list) -> dict:
        """Create the model pipelines.
        
        Keyword arguments:
        keywords -- a list with arguments to choose the model from.
        """
        pass

    def cross_val_score(self, X, y, cv):
        """Return a DataFrame containing the cross validation scores.
        
        Keyword arguments:
        X -- the training features
        y -- the training labels
        cv -- cross validator
        """
        scores = {}
        scoring = ['balanced_accuracy', 'precision', 'recall',
                   'roc_auc', 'accuracy', 'matthews_corrcoef']
        for key, model in self.pipelines.items():
            scores[key] = cross_validate(
                model, X=X, y=y, cv=cv, scoring=scoring)

        res = self.scores_to_df(scores)

        cols = [f'test_{col}' for col in scoring]
        res = res[cols]

        return res.sort_values(by=['test_matthews_corrcoef', 'test_recall', 'test_precision'],
                               axis=0, ascending=False)

    def scores_to_df(self, scores):
        """Convert the scores to a pandas DataFrame.
        
        Keyword arguments:
        scores: a dictionary of values scores.
        """
        index = []
        data = {}

        for key, score in scores.items():
            for score_key in score.keys():
                data[score_key] = []
            break

        for key, score in scores.items():
            index.append(key)
            for score_key, score_val in score.items():
                data[score_key].append(score_val.mean())

        return pd.DataFrame(data=data, index=pd.Series(index, name='model'))


class SimpleModels(Models):
    """Provides a dictionary of models that can be used for calculations.

    Keyword arguments:
    keywords -- identifier str of a model

    Valid keywords are: lr, gb, svm, tree, knn, rf, grad_b, ada_b
    """

    def __init__(self, model_names: list) -> None:
        super().__init__(model_names)

    def create_pipelines(self, keywords: list) -> dict:
        """Create the model pipelines using StandardScaler()
        
        Keyword arguments:
        keywords -- a list with arguments to choose the model from.
        """
        mapping = Models.mapping

        pipelines = {}

        for key in keywords:
            if key not in mapping:
                raise KeyError(
                    'Expects one of values: {}, but got: {}'.format(mapping.keys(), key))

            pipelines[key] = make_pipeline(StandardScaler(), mapping[key]())

        return pipelines


class SmoteModels(Models):
    """Provides a dictionary of models that can be used for calculations with a SMOTE pipeline.
    It upsampled the minority class and downsamples the majority class.

    Used for imbalanced datasets.

    Keyword arguments:
    keywords -- identifier str of a model

    Valid keywords are: lr, gb, svm, tree, knn, rf, grad_b, ada_b"""
    def __init__(self, model_names: list) -> None:
        super().__init__(model_names)

    def create_pipelines(self, keywords: list):
        """Create the model pipelines using SMOTE.
        
        Keyword arguments:
        keywords -- a list with arguments to choose the model from.
        """
        mapping = Models.mapping

        pipelines = {}

        over = SMOTE()
        under = RandomUnderSampler()

        for key in keywords:
            if key not in mapping:
                raise KeyError(
                    'Expects one of values: {}, but got: {}'.format(
                        mapping.keys(), key))

            pipelines[key] = Pipeline(steps=[(
                'over', over), ('under', under),
                ('scaler', StandardScaler()),
                (key, mapping[key]())])

        return pipelines
