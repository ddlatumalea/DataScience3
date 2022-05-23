"""Module for splitting data."""
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split


class Splitter(ABC):
	"""Abstract class to split data into several files.
	
	Keyword arguments:
	df -- a pandas DataFrame containing the data.
	"""
	def __init__(self, df: pd.DataFrame) -> None:
		super().__init__()
		self.df = df

	@abstractmethod
	def split(target: str):
		"""Split the data into several files.
		
		Keyword arguments:
		target -- the name of the y label
		"""
		pass

class StratifiedSplitter(Splitter):
	"""The splitter uses Stratified splitting to make sure the 
	files will keep the same class ratios.
	Keyword arguments:
	df -- a pandas DataFrame containing the data.
	"""

	def __init__(self, df: pd.DataFrame) -> None:
		super().__init__(df)

	def split(self, target: str):
		"""Split the data into several files.
		
		Keyword arguments:
		target -- the name of the y label
		"""
		X_train, X_test, y_train, y_test = self.stratified_split(self.df, target)

		return X_train, X_test, y_train, y_test

	def stratified_split(self, df, target):
		"""Split the data in a stratified manner and return the datasets.
		
		Keyword arguments:
		df -- the data to split
		target -- the name of the y label
		"""
		X = df.drop(columns=target)
		y = df[target]

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

		return X_train, X_test, y_train, y_test