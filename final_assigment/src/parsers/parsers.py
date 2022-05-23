"""Module for transforming data."""
from scipy.stats import boxcox
import pandas as pd

class Transform:
    """Transformer class to transform data."""
    def __init__(self) -> None:
        pass

    def box_cox_transform(self, X):
        """Aply box cox transformation to transform data to gaussian distribution.
        
        Keyword arguments:
        X -- training features.
        """
        data = X.copy()
        for col in data.columns:
            if all(data[col] > 0):
                transformation, _ = boxcox(data[col])
                data[col] = transformation
            
        return data
