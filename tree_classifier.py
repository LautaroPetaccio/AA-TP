import json
import numpy as np
import pandas as pd
import preprocessor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

# Leo los mails (poner los paths correctos).
ham_txt = json.load(open('dataset/ham_txt.json'))
spam_txt = json.load(open('dataset/spam_txt.json'))

df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])

print "Pre-processing mails"
df['proccessed_text'] = map(lambda x: preprocessor.retrieve_message_text(x), df['text'])

print "Getting the tfidf matrix"
# Get tfidf matrix
df['tfidf'] = preprocessor.tfidf_matrix(df['proccessed_text'], True)

print "Getting the mails length"
# Get mails length
df['len'] = map(len, df.text)

df['class'] = ['ham'] * len(ham_txt) + ['spam'] * len(spam_txt)

# Elijo mi clasificador.
clf = DecisionTreeClassifier()

# Preparo data para clasificar
X = df[['tfidf', 'len']].values
y = df['class']

print "Starting cross validation test"

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)