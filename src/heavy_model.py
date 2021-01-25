import numpy
from numpy.core._multiarray_umath import ndarray
from sklearn.linear_model import SGDClassifier


class MoodGuessUseCase:
    def is_positive(self, param):
        return 1

import pandas as pd

import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


documents = []

from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import nltk
nltk.download('wordnet')
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
#nltk.download("stopwords")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# test.csv
import numpy as np

def clean(X_input):
    stemmer = WordNetLemmatizer()

    for sen in range(0, len(X_input)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X_input[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        #
        #    document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

    #    documents.append(document)

   # return np.array(documents)
    # predicted = text_clf2.predict(X_test)
    # np.mean(predicted == Y_test)


class HeavyModel():

    text_clf: Pipeline
    reviews_train: ndarray

    def __init__(self, train_csv):
        self.train_csv = train_csv
        self.sentiment_train = []

    def fit(self):
        numpy_array = self.train_csv.to_numpy()
        self.reviews_train = clean(numpy_array[:, 0])
        sentiment_train = numpy_array[:, 1]
        self.sentiment_train = sentiment_train.astype('int')

        self.text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ]).fit(self.reviews_train, sentiment_train)

#X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.4, random_state=42)
 # don't split if the given data base is already a split

    def predict(self, reviews_train):
        predicted = self.text_clf.predict(clean(reviews_train))
        print(np.mean(predicted == self.sentiment_train))
        return predicted
