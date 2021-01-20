import pandas as pd

import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

# test.csv
import numpy as np

class MoodGuessUseCase:
    def __init__(self):
        numpy_array = data.to_numpy()
        docs_test = numpy_array[:, 0]
        self.mood_test = numpy_array[:, 1].astype('int')

#        X_train, X_test, Y_train, Y_test = train_test_split(
 #           X, Y, test_size=0.4, random_state=42)
        # don't split if the given data base is already a split

        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        self.text_clf = text_clf.fit(docs_test, self.mood_test)

    def is_positive(self, param):
        return 1

    def predict(self, docs_test):
        self.predicted = self.text_clf.predict(docs_test)
        return self.predicted

    def mean(self):
        print(np.mean(self.predicted == self.mood_test))

    def loads(self):
        pass

data = pd.read_csv("E:/dstesttask1/train.csv", index_col='id')

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
#gs_clf.best_params_
#print(gs_clf.best_score_)
#gs_clf.best_params_

#Y_train = Y_train.astype('int')
#text_clf = text_clf.fit(X_train, Y_train)

#predicted = text_clf.predict(X_test)
#print(np.mean(predicted == Y_test))
data = pd.read_csv("D:/projects/ds/data-science-hackathon-machine-learning/train.csv",index_col='id')



def present(param):
    submission = pd.read_csv("D:/projects/ds/data-science-hackathon-machine-learning/submission.csv")
    submission['sentiment'] = param
    submission.to_csv("E:/dstesttask1/submission.csv", index=False)


def control():
    sdata = pd.read_csv("E:/dstesttask1/test.csv", index_col='id')
    # test.csv
    import numpy as np
    numpy_array = sdata.to_numpy()
    reviews_test = numpy_array[:, 0] # 0 is reviews

    mood = MoodGuessUseCase()

    mood.predict(reviews_test)
    present(mood.predicted)
