import numpy
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

data = pd.read_csv("D:/projects/ds/data-science-hackathon-machine-learning/train.csv",index_col='id')

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

        documents.append(document)

    return np.array(documents)
    # predicted = text_clf2.predict(X_test)
    # np.mean(predicted == Y_test)


numpy_array = data.to_numpy()
X = clean(numpy_array[:, 0])
Y = numpy_array[:, 1]


#X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.4, random_state=42)
 # don't split if the given data base is already a split


text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB()),
])

Y=Y.astype('int')
text_clf = text_clf.fit(X, Y)




predicted = text_clf.predict(clean(X))
print(np.mean(predicted == Y))

