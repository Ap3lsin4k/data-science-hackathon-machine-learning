import pandas as pd

import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
# test.csv
from sklearn.model_selection import GridSearchCV


class MoodGuessUseCase:
    def __init__(self):
        train_csv = pd.read_csv("E:/dstesttask1/train.csv", index_col='id')
        numpy_array = train_csv.to_numpy()
        review_train = numpy_array[:, 0] # aka X_train
        self.sentiment_train = numpy_array[:, 1].astype('int') # aka Y_train

        #X_train, X_test, Y_train, Y_test = train_test_split(
        #X, Y, test_size=0.4, random_state=42)

        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        parameters = {'clf__alpha': (1e-2, 1e-3),
                      'tfidf__use_idf': (True, False),
                      'vect__ngram_range': [(1, 1), (1, 2)],
                      }

        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        self.gs_clf = gs_clf.fit(review_train, self.sentiment_train)

    def predict(self, reviews):
        self.predicted = self.gs_clf.predict(reviews)
        return self.predicted

    def print_accuracy(self):
        print("Accuracy mean:", np.mean(self.predicted == self.sentiment_train))

    def loads(self):
        pass


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

    # Res
    mood = MoodGuessUseCase()

    mood.predict(reviews_test)
    present(mood.predicted)
