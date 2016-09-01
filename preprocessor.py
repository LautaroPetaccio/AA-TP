import email
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from HTMLParser import HTMLParser

# HTML tag stripper from http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
class MLStripper(HTMLParser):
  def __init__(self):
    self.reset()
    self.fed = []
  def handle_data(self, d):
    self.fed.append(d)
  def get_data(self):
    return ''.join(self.fed)

def strip_tags(html):
  s = MLStripper()
  s.feed(html)
  return s.get_data()

# Use custom tokenizer and lemmatizer
class LemmaTokenizer(object):
  def __init__(self):
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def retrieve_message_text(message_text):
  message_text = message_text.encode('ascii', 'ignore')
  message = email.message_from_string(message_text)
  return retrieve_payload_text(message)

def retrieve_payload_text(message):
  payload = ""
  # Has the email payload multiple contents?
  # This is the recursive case
  if "multipart" in message.get_content_type():
    for payload_message in message.get_payload():
      # This conditional avoids emails wrongly parsed
      if type(payload_message) is str:
        payload = payload + payload_message
      else:
        payload = payload + retrieve_payload_text(payload_message)
  else:
    # Base case
    # The content type must be text
    if "text" not in message.get_content_type():
      return ""

    # Gets the text in the message
    payload = message.get_payload()
    if "html" in message.get_content_type():
      # Strips the HTML content
      payload = strip_tags(message.get_payload())

  return payload

def tfidf_matrix(messages, custom_tokenizer):
  # TODO: try bi-grams
  if not custom_tokenizer:
    vectorizer = CountVectorizer(min_df=1)  
  else:
    # Using the nltk lemmatizer and tokenizer includes punctuation marks
    vectorizer = CountVectorizer(min_df=1, tokenizer=LemmaTokenizer())
  counts = vectorizer.fit_transform(messages).toarray()
  # print counts
  transformer = TfidfTransformer()
  transformer.fit_transform(counts).toarray()
