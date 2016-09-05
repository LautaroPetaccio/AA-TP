import numpy as np
import pandas as pd
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SimpleFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Extract attributes given a list of tuples(name, callable)
    """

    def __init__(self, extractors):
        self.extractors = extractors

    def fit(self, x, y=None):
        return self

    def transform(self, mails):
        features = pd.DataFrame()
        for name, extractor in self.extractors:
            features[name] = mails.apply(extractor, axis=1)

        return features

    def get_feature_names(self):
        return [e[0] for e in self.extractors]


class SentimentsStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, messages):
        sid = SentimentIntensityAnalyzer()
        sentiment_analysis_result = []
        for message in messages:
            body_sentences = tokenize.sent_tokenize(message)
            sentences_stats = map(
                lambda sentence: sid.polarity_scores(sentence), body_sentences)
            stats = {'neg': 0, 'neu': 0, 'pos': 0}
            for sentence_stat in sentences_stats:
                stats['neg'] += sentence_stat['neg']
                stats['neu'] += sentence_stat['neu']
                stats['pos'] += sentence_stat['pos']
            if len(sentences_stats) != 0:
                stats['neg'] /= len(sentences_stats)
                stats['neu'] /= len(sentences_stats)
                stats['pos'] /= len(sentences_stats)
            sentiment_analysis_result.append(stats)
        return sentiment_analysis_result


def count_spaces(mail):
    """Returns the number of blank spaces in the email body"""
    return mail['body'].count(" ")


def body_length(mail):
    """Returns the body length"""
    return len(mail['body'])


def has_html(mail):
    """Returns 1 if the mail has a HTML content type and 0 if not"""
    return int(has_content_type(mail, 'html'))


def has_image(mail):
    """Returns 1 if the mail has a image content type and 0 if not"""
    return int(has_content_type(mail, 'image'))


def has_content_type(mail, content_type):
    """
    Returns true if any of the mails content types is equal to content_type
    """
    return any([content_type in mct for mct in mail['content_types']])


def number_of_sentences(mail):
    """Returns the number of sentences in a mail body"""
    sentences = tokenize.sent_tokenize(mail['body'])
    return len(sentences)
