import time
import io
import json
import os

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.externals import joblib

import email_features as ef


def measure_time(func):
    t0 = time.time()
    res = func()
    duration = time.time() - t0
    return res, duration


def print_time(func):
    res, duration = measure_time(func)
    print 'Done in %fs' % duration
    return res


def load_file_and_get_mails(label, subset_size=None):
    filename = 'dataset/%s_dev.json' % label
    print 'Loading %s data from %s' % (label, filename)
    with io.open(filename, encoding='ascii') as file:
        plain_mails = print_time(lambda: [plain_mail.encode(
            'ascii', 'ignore') for plain_mail in json.load(file)])

    print 'Loaded %d(%0.3fMB) %s mails' % (len(plain_mails), sum(map(len, plain_mails)) / 1e6, label)

    if subset_size is not None:
        subset_size = min(data_size, subset_size)
        plain_data = [plain_data[i] for i in np.random.choice(
            subset_size, subset_size, replace=False)]

    return plain_mails


def parse_mails(plain_mails, label):
    print 'Parsing %s mails' % label
    mails = print_time(lambda: ef.parse_mails(plain_mails))
    print 'Parsed %d %s mails' % (len(mails), label)

    return mails


def get_mails(label, subset_size=None):
    return parse_mails(load_file_and_get_mails(label, subset_size=subset_size), label)


def create_dataframe_from_mails(ham_mails, spam_mails):
    mails = ham_mails + spam_mails
    labels = ['ham'] * len(ham_mails) + ['spam'] * len(spam_mails)
    return pd.DataFrame({
        'subject': [m.get('subject') if m.get('subject') is not None else '' for m in mails],
        'body': [ef.retrieve_payload_text(m) for m in mails],
        'content_types': [ef.retrieve_content_type_list(m) for m in mails],
        'label': labels
    }, columns=['content_types', 'subject', 'body', 'label'])


def load_raw_data(test_size=0.20, subset_size=None, spam_proportion=0.5, random_state=None):
    if subset_size is not None:
        ham_size = int(subset_size * (1 - spam_proportion))
        spam_size = int(subset_size * spam_proportion)
    else:
        ham_size = None
        spam_size = None

    ham_mails = get_mails('ham', subset_size=ham_size)
    print ''
    spam_mails = get_mails('spam', subset_size=spam_size)
    print ''

    print 'Generating pandas DataFrame'
    df = print_time(lambda: create_dataframe_from_mails(ham_mails, spam_mails))
    print ''

    print 'Splitting into Train Set and Test Set'
    train_set, test_set = print_time(lambda: train_test_split(
        df, test_size=test_size, random_state=random_state))
    print ''

    print 'Resetting indexes'
    train_set, test_set = print_time(
        lambda: (train_set.reset_index(drop=True), test_set.reset_index(drop=True)))

    return train_set, test_set


def load_processed_data():
    # Por reproducibilidad, y para no perder tiempo armando el DataFrame cada vez, lo bajamos a
    # un archivo la priemra vez que lo corrimos, ya seprando 20% para reportar como valor final
    # de nuestro modelo
    train_set_name = 'dataset/train_set.pkl'
    test_set_name = 'dataset/test_set.pkl'
    train_set = None
    test_set = None

    if os.path.isfile(train_set_name) and os.path.isfile(test_set_name):
        print 'Loading Train Set'
        train_set = print_time(lambda: joblib.load(train_set_name))
        print ''

        print 'Loading Test Set'
        test_set = print_time(lambda: joblib.load(test_set_name))
    else:
        train_set, test_set = load_raw_data()
        print ''

        print 'Saving Train Set'
        print_time(lambda: joblib.dump(
            train_set, train_set_name, compress=True))
        print ''

        print 'Saving Test Set'
        print_time(lambda: joblib.dump(test_set, test_set_name, compress=True))

    return train_set, test_set


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


def evaluate_and_meassure(X, pipelines, results_prefix, cv=10, n_jobs=1, verbose=0, force_run=False):
    if not os.path.exists('results'):
        os.makedirs('results')

    scores_name = 'results/%s_scores.pkl' % results_prefix
    cv_times_name = 'results/%s_cv_times.pkl' % results_prefix

    scores = {}
    cv_times = {}

    loaded_scores = {}
    loaded_cv_times = {}

    if not force_run and os.path.isfile(scores_name):
        print 'Loading previous scores'
        loaded_scores = print_time(lambda: joblib.load(scores_name))

    if not force_run and os.path.isfile(cv_times_name):
        print 'Loading previous cv_times'
        loaded_cv_times = print_time(lambda: joblib.load(cv_times_name))

    print ''

    i = 0
    pipelines_count = len(pipelines)

    for name, _, pipeline in pipelines:
        i = i + 1

        if name in loaded_scores:
            print 'Loaded from previous run %d-Fold CV for pipeline %s(%d/%d)' % (cv, name, i, pipelines_count)
            model_scores = loaded_scores[name]
            model_cv_time = loaded_cv_times[name]
        else:
            print 'Running %d-Fold CV for pipeline %s(%d/%d)' % (cv, name, i, pipelines_count)
            model_scores, model_cv_time = measure_time(lambda: cross_val_score(
                pipeline, X, X.label, cv=cv, n_jobs=n_jobs, verbose=verbose))
            print 'Done in %fs' % model_cv_time

        scores[name] = model_scores
        cv_times[name] = model_cv_time
        joblib.dump(scores, scores_name, compress=True)
        joblib.dump(cv_times, cv_times_name, compress=True)

        print 'CV scores mean: %f std: %f' % (np.mean(model_scores), np.std(model_scores))
        print ''

    return scores, cv_times


def save_model(name, folder, model):
    if not os.path.exists(folder):
        os.makedirs(folder)

    print 'Saving model %s to disk' % name
    print_time(lambda: joblib.dump(model, '%s/%s.pkl' %
                                   (folder, name), compress=True))
    print 'Saved at %s/%s.pkl' % (folder, name)
