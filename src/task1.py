class MoodGuessUseCase:
    def is_positive(self, param):
        return 0
#hello
import pandas as pd

import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("E:/train.csv", index_col='id')

# test.csv
import numpy as np

numpy_array = data.to_numpy()
X = numpy_array[:, 0]
Y = numpy_array[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42)
 #don't split if the given data base is already a split


text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
from sklearn.model_selection import GridSearchCV
parameters = {'clf__alpha': (1e-2, 1e-3),
            'tfidf__use_idf': (True, False),
            'vect__ngram_range': [(1, 1), (1, 2)],
 }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
Y_train=Y_train.astype('int')
gs_clf = gs_clf.fit(X_train, Y_train)
predicted = gs_clf.predict(X_test)
print(np.mean(predicted == Y_test))
gs_clf.best_params_
print(gs_clf.best_score_)
#gs_clf.best_params_

#Y_train = Y_train.astype('int')
#text_clf = text_clf.fit(X_train, Y_train)

#predicted = text_clf.predict(X_test)
#print(np.mean(predicted == Y_test))
