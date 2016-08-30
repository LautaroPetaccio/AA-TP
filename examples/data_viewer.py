# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import email
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("type", help="Type of the email, spam or ham")
parser.add_argument("number", type=int, help="Email index")
parser.add_argument("-s", "--strip", action="store_true", help="Strip HTML into text")
args = parser.parse_args()

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
if args.strip and ("html" in msg.get_content_type()):
  print strip_tags(msg.get_payload())
else:
  print msg.get_payload()
print "------------------------------------------------------"
print "Content type: " + msg.get_content_type()
print "------------------------------------------------------"
print "Message keys: "
print msg.keys()
print "------------------------------------------------------"