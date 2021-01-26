# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import pickle
import sys
from datetime import datetime

#import estimator as estimator
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

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV


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
from nltk.stem import WordNetLemmatizer


class SmartStableCV:
    gs_clf: GridSearchCV

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
            document = document.lower()
            document = document.split()

            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)

            # Lemmatization
            #document = ' '.join(document)
        return document
    def __init__(self, train_csv_processed):
        self.gs_clf = "string"
        if os.path.isfile("kaggle/working/gs_classifier.pickle"):
            print("Using precomputed classifier from \"kaggle/working/gs_classifier.pickle\" ")
            self.load()
        else:
            print("Teaching on training reviews")
            self.fit(train_csv_processed)
            print("Dumping")
            #self.dump()

    def fit(self, train_csv):



        numpy_array = train_csv.to_numpy()
        review_train = (numpy_array[:, 0]) # aka X_train
        self.sentiment_train = numpy_array[:, 1].astype('int') # aka Y_train

        parameters = { 'C' : [0.01, 0.05, 0.125, 0.17, 0.2, 0.25,0.30, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]
                    }
        gs_clf = GridSearchCV(LinearSVC(), parameters, scoring='f1', n_jobs=-1)
        self.tfidf=TfidfVectorizer(min_df=5, max_df =0.7, ngram_range=(1,5))
        review_train=self.tfidf.fit_transform(train_csv['processed'])
        self.gs_clf = gs_clf.fit(review_train, self.sentiment_train)


    def predict(self, reviews):
        #tfidf=TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1,2))
        reviews= self.tfidf.transform(reviews)
        self.predicted = self.gs_clf.best_estimator_.predict(reviews)
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

stop_words = set(stopwords.words('english'))


def process(text):

    res = re.sub(r'\s+', ' ', text, flags=re.I)
    res = re.sub(r'^br\s+', '', text)
    res = re.sub('<.*?>', ' ', text)
    res = re.sub('\W', ' ', res)
    res = re.sub('\s+[a-zA-Z]\s+', ' ', res)
    res = re.sub('\s+', ' ', res)
    word_tokens = word_tokenize(res)
    filtered_res = " ".join([w for w in word_tokens if w not in stop_words])
    return filtered_res


def present(sentiment_column, submission_csv_path="model/data/newsubmission.csv"):
    submission = pd.read_csv("../src/model/data/defaultsubmission.csv")
    submission['sentiment'] = sentiment_column
    submission.to_csv(submission_csv_path, index=False)

def get_new_submission_path_with_version():
    return datetime.now().strftime("kaggle/working/newsubmission %d %H;%M;%S.csv")  # day hour:minutes:seconds

def control():
    train_csv = pd.read_csv('kaggle/input/text-classification-int20h/train.csv', index_col='id')
    test_csv = pd.read_csv('kaggle/input/text-classification-int20h/test.csv', index_col='id')
    test_csv['processed'] = test_csv['review'].apply(lambda x: process(x))
    train_csv['processed'] = train_csv['review'].apply(lambda x: process(x))

    # test.csv
    import numpy as np
    numpy_array = test_csv.to_numpy()
    reviews_test = numpy_array[:, 0] # 0 is reviews

    # Res
    print("construction")
    model = SmartStableCV(train_csv)

    print("predicting...")
    #mood.dump()


    present(model.predict(test_csv['processed']), get_new_submission_path_with_version())

    predicted = model.predict(pd.read_csv("kaggle/input/text-classification-int20h/TRAINing Reviewsset").to_numpy()[:, 0])
    print("Predicted on training reviews", predicted, len(predicted))
    true_sentiment = pd.read_csv("kaggle/input/text-classification-int20h/TRAINing Reviewsset")['sentiment'].map({
    "positive":1,
    "negative":0
    }).to_numpy()

    predicted = np.array(predicted).astype(int)
    print(predicted, len(predicted))
    print("npmean", np.mean(predicted == true_sentiment))

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