# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import email
import argparse
from nltk.tokenize import word_tokenize
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

def retrieve_payload(message):
  payload = None
  # Has the email payload multiple contents?
  # This is the recursive case
  if "multipart" in message.get_content_type():
    for payload_message in message.get_payload():
      if payload == None:
        payload = retrieve_payload(payload_message)
      else:
        payload = payload + retrieve_payload(payload_message)
  else:
    # Base case
    # The content type must be text
    if "text" not in message.get_content_type():
      if args.tokenize:
        return []
      else:
        return ""
    
    # Gets the text in the message
    payload = message.get_payload()

    if args.strip and ("html" in message.get_content_type()):
      # Strips the HTML content
      payload = strip_tags(message.get_payload())
    if args.tokenize:
      # Tokenizes the text
      payload = word_tokenize(payload)
      # Lemmatizes and removes stopwords
      if args.lemmatize:
        payload = map(lambda word: wnl.lemmatize(word), payload)
      if args.stopwords:
        payload = filter(lambda word: word.lower() not in stoplist, payload)
  return payload

parser = argparse.ArgumentParser()
parser.add_argument("type", help="Type of the email, spam or ham")
parser.add_argument("number", type=int, help="Email index")
parser.add_argument("-s", "--strip", action="store_true", help="Strip HTML into text")
parser.add_argument("-t", "--tokenize", action="store_true", help="Tokenize text")
parser.add_argument("-l", "--lemmatize", action="store_true", help="Lemmatize text if it was tokenized")
parser.add_argument("-r", "--stopwords", action="store_true", help="Remove stopwords if tokenized")

args = parser.parse_args()

# Initialize the WordNetLemmatizer
wnl = WordNetLemmatizer()

# Initialize the stopwords lists
stoplist = stopwords.words('english')

# Leo los mails (poner los paths correctos).
ham_txt = json.load(open('../dataset/ham_txt.json'))
spam_txt = json.load(open('../dataset/spam_txt.json'))

# We can use the email parser from:
# https://docs.python.org/2/library/email.parser.html#parser-class-api
# The email parser parses the email and returns a Message instance
# https://docs.python.org/2/library/email.message.html#email.message.Message
if args.type == "ham":
  msg = email.message_from_string(ham_txt[args.number])
else:
  msg = email.message_from_string(spam_txt[args.number])

print "Email number: " + str(args.number)
print "------------------------------------------------------"
print "Email subjet: " + msg.get("subject")
print "------------------------------------------------------"
print "Payload:"
print retrieve_payload(msg)
print "------------------------------------------------------"
print "Content type: " + msg.get_content_type()
print "------------------------------------------------------"
print "Message keys: "
print msg.keys()
print "------------------------------------------------------"