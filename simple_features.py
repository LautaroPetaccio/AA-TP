import email
from nltk import tokenize
import email_text_retrieval as er
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

# Returns the number of blank spaces in the email body
def count_spaces(message):
	return er.retrieve_payload_text(message).count(" ")

# Returns the body length
def body_length(message):
  return len(er.retrieve_payload_text(message))

# Returns true if the message has a HTML content type
def has_html(message):
  return has_content_type(message, 'html')

# Returns true if the message has a image content type
def has_image(message):
  return has_content_type(message, 'image')

# Returns true if the message has te content type specified
def has_content_type(message, content_type):
  if content_type in message.get_content_type():
    return True
  for message in message.message.get_payload():
    if has_content_type(message, content_type):
      return True
  return False

# Returns the number of sentences in a mail body
def number_of_sentences(message):
  sentences = tokenize.sent_tokenize(er.retrieve_payload_text(message))
  return len(sentences)