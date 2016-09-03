from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SimpleFeaturesExtractor(BaseEstimator, TransformerMixin):
  """Extrae atributos en base a la lista de tuplas (nombre, funcion) 'extractors' """
  def __init__(self, extractors):
  	self.extractors = extractors

  def fit(self, x, y=None):
  	return self

  def transform(self, mails):
  	return np.array([[ e[1](mail) for e in self.extractors ] for mail in mails ])

  def get_feature_names(self):
  	return [ e[0] for e in self.extractors ]

def count_spaces(txt):
	return txt.count(" ")
