# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import pickle
import sys

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import re
import nltk

from src.model_wrapper import ClassifierWrapper

#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

train_csv = pd.read_csv('kaggle/input/text-classification-int20h/train.csv', index_col = 'id')
test = pd.read_csv('kaggle/input/text-classification-int20h/test.csv', index_col = 'id')

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


class SmartStableCV:
    gs_clf: GridSearchCV

    def __init__(self):
        self.gs_clf = "string"
        if os.path.isfile("kaggle/working/gs_classifier.pickle"):
            print("Using precomputed classifier from \"kaggle/working/gs_classifier.pickle\" ")
            self.load()
        else:
            print("Teaching on training reviews")
            self.fit()
            print("Dumping")
            self.dump()

    def fit(self):
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

    def dump(self):
        pickle_out = open("kaggle/working/gs_classifier.pickle", "wb")
        pickle.dump(self.gs_clf, pickle_out)
        pickle_out.close()

    def load(self):
        with open("kaggle/working/gs_classifier.pickle", "rb") as pickle_in:
            self.gs_clf = pickle.load(pickle_in)
            print("Gsclf", self.gs_clf)



def present(sentiment_column, submission_csv_path="model/data/newsubmission.csv"):
    submission = pd.read_csv("../src/model/data/defaultsubmission.csv")
    submission['sentiment'] = sentiment_column
    submission.to_csv(submission_csv_path, index=False)

def control():
    sdata = test
    # test.csv
    import numpy as np
    numpy_array = sdata.to_numpy()
    reviews_test = numpy_array[:, 0] # 0 is reviews

    # Res
    print("construction")
    model = SmartStableCV()

    print("predicting...")
    #mood.dump()


    present(model.predict(reviews_test))

    predicted = model.predict(pd.read_csv("kaggle/input/text-classification-int20h/TRAINing Reviewsset").to_numpy()[:, 0])
    print(predicted, len(predicted))
    true_sentiment = pd.read_csv("kaggle/input/text-classification-int20h/TRAINing Reviewsset")['sentiment'].map({
    "positive":1,
    "negative":0
    }).to_numpy()

    predicted = np.array(predicted).astype(int)
    print(predicted, len(predicted))

    error = 0
    for i in range(len(predicted)):
        if predicted[i] != true_sentiment[i]:
            error += 1

    print("er", error, error/len(predicted)*100,"%",  error/len(true_sentiment)*100)
print(pd.read_csv("kaggle/input/text-classification-int20h/TRAINing Reviewsset")['sentiment'].map({
    "positive":1,
    "negative":0
    }).to_numpy(), len (pd.read_csv("kaggle/input/text-classification-int20h/TRAINing Reviewsset")['sentiment'].map({
    "positive":1,
    "negative":0
    }).to_numpy()))

control()