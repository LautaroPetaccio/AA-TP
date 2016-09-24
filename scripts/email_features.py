import email
import itertools

from HTMLParser import HTMLParser


def parse_mails(plain_mails, label):
    return map(email.message_from_string, plain_mails)


class MLStripper(HTMLParser):
    """
    HTML tag stripper from
    http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python

    """

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


def retrieve_content_type_list(message):
    """Returns a list with all the content types present in this message"""
    if message.is_multipart():
        return list(itertools.chain.from_iterable(
            [retrieve_content_type_list(payload_message)
             for payload_message in message.get_payload()]))

    return [message.get_content_type()]
