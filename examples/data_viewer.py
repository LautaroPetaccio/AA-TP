# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import email
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("type", help="Type of the email, spam or ham")
parser.add_argument("number", type=int, help="Email index")
args = parser.parse_args()



# Leo los mails (poner los paths correctos).
ham_txt = json.load(open('../dataset/ham_txt.json'))
spam_txt = json.load(open('../dataset/spam_txt.json'))

# We can use the email parser from:
# https://docs.python.org/2/library/email.parser.html#parser-class-api
# The email parser parses the email and returns a Message instance
# https://docs.python.org/2/library/email.message.html#email.message.Message

msg = email.message_from_string(ham_txt[0])
print "Email number: " + str(args.number)
print "------------------------------------------------------"
print "Email subjet: " + msg.get("subject")
print "------------------------------------------------------"
print "Payload:"
print msg.get_payload()
print "------------------------------------------------------"
print "Content type: " + msg.get_content_type()
print "------------------------------------------------------"
print "Message keys: "
print msg.keys()
print "------------------------------------------------------"