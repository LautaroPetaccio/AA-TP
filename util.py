import time
import io
import json
import email
import random
import os

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

import email_text_retrieval as er


def load_file_and_get_mails(filename, subset_size=None):
    print 'Loading data from %s' % filename

    t0 = time.time()
    with io.open(filename, encoding='utf8') as f:
        data = json.load(f)
    duration = time.time() - t0

    print "Done in %fs" % duration

    data_size = len(data)
    data_mb_size = sum(len(d.encode('utf-8')) for d in data) / 1e6
    print "Loaded %d(%0.3fMB) mails" % (data_size, data_mb_size)

    if subset_size is not None:
        subset_size = min(data_size, subset_size)
        data = [data[i] for i in
                np.random.choice(subset_size, subset_size, replace=False)]

    print 'Parsing mails'

    t0 = time.time()
    mails = [email.message_from_string(d.encode('ascii', 'ignore'))
             for d in data]
    duration = time.time() - t0

    print "Done in %fs" % duration

    print "Parsed %d mails" % len(mails)

    return mails


def load_data(subset_size=None, test_size=0.20, spam_proportion=0.5):
    if subset_size is not None:
        ham_size = int(subset_size * (1 - spam_proportion))
        spam_size = int(subset_size * spam_proportion)
    else:
        ham_size = None
        spam_size = None

    ham_mails = load_file_and_get_mails(
        'dataset/ham_dev.json', subset_size=ham_size)
    ham_size = len(ham_mails)

    spam_mails = load_file_and_get_mails(
        'dataset/spam_dev.json', subset_size=spam_size)
    spam_size = len(spam_mails)

    mails = ham_mails + spam_mails

    print 'Generating Pandas DataFrame'

    t0 = time.time()
    df = pd.DataFrame({
        'subject': [m.get('subject') if m.get('subject') is not None else ''
                    for m in mails],
        'body': [er.retrieve_payload_text(m) for m in mails],
        'content_types': [er.retrieve_content_type_list(m) for m in mails],
        'label': ['ham'] * ham_size + ['spam'] * spam_size
    })

    duration = time.time() - t0

    print "Done in %fs" % duration

    print 'Splitting into Training and Test Set'

    train_set, test_set = train_test_split(df, test_size=test_size)
    train_size = len(train_set)
    test_size = len(test_set)

    duration = time.time() - t0

    print "Done in %fs" % duration

    train_size = len(train_set)
    train_ham_size = sum(train_set['label'] == 'ham')
    train_ham_proportion = float(train_ham_size) / float(train_size)
    train_spam_size = sum(train_set['label'] == 'spam')
    train_spam_proportion = float(train_spam_size) / float(train_size)

    test_size = len(test_set)
    test_ham_size = sum(test_set['label'] == 'ham')
    test_ham_proportion = float(test_ham_size) / float(test_size)
    test_spam_size = sum(test_set['label'] == 'spam')
    test_spam_proportion = float(test_spam_size) / float(test_size)

    print "Train Set: %d samples - Ham: %d(%0.2f%%) Spam: %d(%0.2f%%)" % \
        (train_size, train_ham_size, train_ham_proportion,
         train_spam_size, train_spam_proportion)
    print "Test Set:  %d samples - Ham: %d(%0.2f%%) Spam: %d(%0.2f%%)" % \
        (test_size, test_ham_size, test_ham_proportion,
         test_spam_size, test_spam_proportion)

    return train_set, test_set


def save_model(name, model):
    directory = 'models/%s/%s' % (name, time.strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(directory):
        os.makedirs(directory)

    print "Saving Model %s to disk" % treat_descr

    t0 = time.time()
    joblib.dump(model, '%s/model.pkl' % directory, compress=True)
    duration = time.time() - t0

    print "Done in %fs" % duration
    print "Saved at %s/model.pkl" % directory


class ColumnSelectorExtractor(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step.
    This class should be used to select a column from a pandas data frame.
    """

    def __init__(self, column):
        if isinstance(column, str):
            self.column = column
        else:
            raise ValueError("Invalid type for column")

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame[self.column]

    def get_feature_names(self):
        return [self.column]


class SubjectAndBodyMergerExtractor(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step.
    This class should be used to select a column from a pandas data frame.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame['subject'] + ' ' + data_frame['body']

    def get_feature_names(self):
        return ['subject_and_body']
