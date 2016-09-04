import json
import os
from time import time, strftime
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib

def load_data(subset_size=None, spam_proportion=0.5):
    ham_txt = json.load(open('dataset/ham_dev.json'))
    spam_txt = json.load(open('dataset/spam_dev.json'))
    
    ham_size = len(ham_txt)
    spam_size = len(spam_txt)

    if subset_size is not None:
    	subset_size = min(ham_size + spam_size, subset_size)
    	ham_size = int(subset_size * (1 - spam_proportion))
    	spam_size = int(subset_size * spam_proportion)

        ham_txt = np.random.choice(ham_txt, size=ham_size, replace=False)
        spam_txt = np.random.choice(spam_txt, size=spam_size, replace=False)
    
    data = np.array(ham_txt + spam_txt, dtype=object)
    labels = np.array(['ham' for _ in range(len(ham_txt))] + ['spam' for _ in range(len(spam_txt))], dtype=object)

    data_size = len(data)
    data_mb_size = sum(len(m.encode('utf-8')) for m in data) / 1e6

    ham_proportion = float(ham_size) / float(data_size) * 100
    spam_proportion = float(spam_size) / float(data_size) * 100

    print "Dataset: %d samples(%0.3fMB) - Ham: %d(%0.2f%%) Spam: %d(%0.2f%%)" % (data_size, data_mb_size, ham_size, ham_proportion, spam_size, spam_proportion)
    return data, labels

def extract_features(feature_extractor, feature_extractor_descr, data):
    print "Extracting features from the dataset using a %s" % feature_extractor_descr

    t0 = time()
    X = feature_extractor.fit_transform(data)
    duration = time() - t0

    print "Done in %fs" % duration
    print "Set: %d samples %d features" % X.shape   
    print ""
    
    return X


def cross_validate(clf, treat_descr, X, y, cv_folds=10, n_jobs=8):
    print "Running %d-Fold Cross Validation for %s" % (cv_folds, treat_descr)

    t0 = time()
    cv_scores = cross_val_score(clf, X, y, cv=cv_folds, n_jobs=n_jobs)
    cv_time = time() - t0
    
    print "Done in %fs" % cv_time
    print "CV Score: mean %f std %f" % (np.mean(cv_scores), np.std(cv_scores))

    return cv_scores

def run_ml_pipeline(feature_extractor_tuple, clf_tuple, data, labels, cv_folds=10, n_jobs=8):
    feature_extractor, feature_extractor_descr = feature_extractor_tuple
    clf, clf_descr = clf_tuple
    treat_descr = '%s-%s' % (feature_extractor_descr, clf_descr)
    run_start = strftime("%Y%m%d-%H%M%S")

    print "Running ML Pipeline for %s(%s)" % (treat_descr, run_start)

    X = extract_features(feature_extractor, feature_extractor_descr, data)
    cross_validate(clf, clf_descr, X, labels, cv_folds=cv_folds, n_jobs=n_jobs)

    directory = 'results/%s/%s' % (treat_descr, run_start)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print "Saving Trained Extractor and Model %s" % treat_descr
    joblib.dump(feature_extractor, '%s/extractor.pkl' % directory, compress=True)
    joblib.dump(clf_tuple[0], '%s/classifier.pkl' % directory, compress=True)
    