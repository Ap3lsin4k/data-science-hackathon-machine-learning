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



class FirstAlgorithm():
    def __init__(self):
        data = pd.read_csv("data/train.csv", index_col='id')

        numpy_array = data.to_numpy()
        X = numpy_array[:, 0]
        Y = numpy_array[:, 1]

        self.X_train, self.X_test, Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.95, random_state=42)
        # don't split if the given data base is already a split
        self.Y_train = Y_train.astype('int')

    def fit(self):

        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

        self.text_clf = text_clf.fit(self.X_train, self.Y_train)

    def predict(self):
        predicted = self.text_clf.predict(self.X_test)
        return predicted

    def print_prediction(self):
        print(np.mean(fa.predict() == self.Y_test))
    # TODO print what is left unknown

FirstAlgorithm()
fa = FirstAlgorithm()
fa.fit()
fa.print_prediction()
