class MoodGuessUseCase:
    def is_positive(self, param):
        return 1


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

data = pd.read_csv("D:/projects/ds/data-science-hackathon-machine-learning/train.csv",index_col='id')

# test.csv
import numpy as np

numpy_array = data.to_numpy()
X = numpy_array[:, 0]
Y = numpy_array[:, 1]


X_train, X_test, Y_train, Y_test = train_test_split(
     X, Y, test_size=0.4, random_state=42)
 # don't split if the given data base is already a split


text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
Y=Y.astype('int')
text_clf = text_clf.fit(X, Y)


predicted = text_clf.predict(X)
print(np.mean(predicted == Y))
# predicted = text_clf2.predict(X_test)
# np.mean(predicted == Y_test)
