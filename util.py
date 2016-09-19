import time
import io
import json
import email
import os

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

import email_text_retrieval as er

def print_time(func):
    t0 = time.time()
    res = func()
    duration = time.time() - t0
    print 'Done in %fs' % duration
    return res

def load_file_and_get_mails(label, subset_size=None):
    filename = 'dataset/%s_dev.json' % label
    print 'Loading %s data from %s' % (label, filename)
    with io.open(filename, encoding='ascii') as file:
        plain_mails = print_time(lambda: [plain_mail.encode('ascii', 'ignore') for plain_mail in json.load(file)])

    print 'Loaded %d(%0.3fMB) %s mails' % (len(plain_mails), sum(map(len, plain_mails)) / 1e6, label)

    if subset_size is not None:
        subset_size = min(data_size, subset_size)
        plain_data = [plain_data[i] for i in np.random.choice(subset_size, subset_size, replace=False)]

    print 'Parsing %s mails' % label
    mails = print_time(lambda: map(email.message_from_string, plain_mails))
    print 'Parsed %d %s mails' % (len(mails), label)

    return mails

def create_dataframe_from_mails(mails, labels):
    return pd.DataFrame({
        'subject': [m.get('subject') if m.get('subject') is not None else '' for m in mails],
        'body': [er.retrieve_payload_text(m) for m in mails],
        'content_types': [er.retrieve_content_type_list(m) for m in mails],
        'label': labels
    }, columns=['content_types', 'subject', 'body', 'label'])

def print_sets_summarys(train_set, test_set):
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

    print 'Train Set: %d samples - ham: %d(%0.2f%%) spam: %d(%0.2f%%)' % \
        (train_size, train_ham_size, train_ham_proportion,
         train_spam_size, train_spam_proportion)
    print 'Test Set:  %d samples - ham: %d(%0.2f%%) spam: %d(%0.2f%%)' % \
        (test_size, test_ham_size, test_ham_proportion,
         test_spam_size, test_spam_proportion)


def load_raw_data(test_size=0.20, subset_size=None, spam_proportion=0.5, random_state=None):
    if subset_size is not None:
        ham_size = int(subset_size * (1 - spam_proportion))
        spam_size = int(subset_size * spam_proportion)
    else:
        ham_size = None
        spam_size = None

    ham_mails = load_file_and_get_mails('ham', subset_size=ham_size)
    print ''

    spam_mails = load_file_and_get_mails('spam', subset_size=spam_size)
    print ''

    print 'Generating pandas DataFrame'
    df = print_time(lambda: create_dataframe_from_mails(ham_mails + spam_mails, ['ham'] * len(ham_mails) + ['spam'] * len(spam_mails)))
    print ''

    print 'Splitting into Train Set and Test Set'
    train_set, test_set = print_time(lambda: train_test_split(df, test_size=test_size, random_state=random_state))
    print ''

    print 'Resetting indexes'
    train_set, test_set = print_time(lambda: (train_set.reset_index(drop=True), test_set.reset_index(drop=True)))

    return train_set, test_set


def save_model(name, model):
    if not os.path.exists('models'):
        os.makedirs('models')

    print 'Saving model %s to disk' % treat_descr
    t0 = time.time()
    joblib.dump(model, 'models/%s.pkl' % name, compress=True)
    duration = time.time() - t0
    print 'Done in %fs' % duration

    print 'Saved at models/%s.pkl' % name


class ColumnSelectorExtractor(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step.
    This class should be used to select a column from a pandas data frame.
    """

    def __init__(self, column):
        if isinstance(column, str):
            self.column = column
        else:
            raise ValueError('Invalid type for column')

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
