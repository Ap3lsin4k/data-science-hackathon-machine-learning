import pandas as pd
import re

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

# Data cleaning
def clean(s):
    s = s.lower()
    s = re.sub('<br />','',s)
    
    return s


class MoodGuessUseCase:
    
    def __init__(self):
        train_csv = pd.read_csv("src/model/data/train.csv", index_col='id')
        train_csv['review'] = train_csv['review'].apply(clean)

        numpy_array = train_csv.to_numpy()
        review_train = numpy_array[:, 0] # aka X_train
        self.sentiment_train = numpy_array[:, 1].astype('int') # aka Y_train

        # This doesn't seem to be right, but it's works with "predict" and "print_accuracy" function
        global X_test
        global Y_test
        
        X_train, X_test, Y_train, Y_test = train_test_split(
        review_train, self.sentiment_train, test_size=0.4, random_state=42)

        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
                             ])
        parameters = {'clf__alpha': (1e-2, 1e-3),
                      'tfidf__use_idf': (True, False),
                      'vect__ngram_range': [(1, 1), (1, 2)],
                      }

        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        self.gs_clf = gs_clf.fit(X_train, Y_train)

    def predict(self, reviews):
        self.predicted = self.gs_clf.predict(X_test)
        return self.predicted

    def print_accuracy(self):
        print("Accuracy mean:", np.mean(self.predicted == Y_test))

    def loads(self):
        pass


def present(param):
    submission = pd.read_csv("src/model/data/submission.csv")
    submission['review'] = submission['review'].apply(clean)

    submission['sentiment'] = param
    submission.to_csv("src/model/data/submission.csv", index=False)


def control():
    sdata = pd.read_csv("src/model/data/test.csv", index_col='id')
    sdata['review'] = sdata['review'].apply(clean)

    # test.csv
    import numpy as np
    numpy_array = sdata.to_numpy()
    reviews_test = numpy_array[:, 0] # 0 is reviews

    # Res
    mood = MoodGuessUseCase()

    mood.predict(reviews_test)
    present(mood.predicted)
