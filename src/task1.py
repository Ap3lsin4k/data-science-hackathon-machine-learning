import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
# test.csv
from sklearn.model_selection import GridSearchCV


class MoodGuessUseCase:
    def __init__(self):
        train = pd.read_csv("E:/dstesttask1/train.csv", index_col='id')
        #numpy_array = train_csv.to_numpy()
        #review_train = numpy_array[:, 0] # aka X_train
        #self.sentiment_train = numpy_array[:, 1].astype('int') # aka Y_train

        #X_train, X_test, Y_train, Y_test = train_test_split(
        #X, Y, test_size=0.4, random_state=42)

        train['sentiment'].value_counts()
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(min_df=5)
        train_tfidf = self.tfidf.fit_transform(train['review'])

        self.est = MultinomialNB().fit(train_tfidf, train['sentiment'].values)

        #self.gs_clf = gs_clf.fit(review_train, self.sentiment_train)

    def predict(self, reviews):
        reviews_test = self.tfidf.transform(reviews)

        self.predicted = self.est.predict(reviews_test)
        return self.predicted

#    def print_accuracy(self):
#        print("Accuracy mean:", np.mean(self.predicted == self.sentiment_train))


