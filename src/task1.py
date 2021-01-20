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

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

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