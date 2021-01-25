import pandas as pd

import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

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
            X, Y, test_size=0.4, random_state=42)
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
        print(np.mean(self.predict() == self.Y_test))
    
    def confusion_matrix(self):
        return confusion_matrix(fa.predict(),self.Y_test.astype(int))

    def pred_data(self,data):
        predicted = self.text_clf.predict(data)
        return predicted
        
FirstAlgorithm()
fa = FirstAlgorithm()
fa.fit()
fa.print_prediction()

errors_false_negative = []
errors_false_positive = []

#saving errors
for x,y_true in zip(fa.X_test,fa.Y_test):
    y_pred = fa.pred_data([x])
    
    if y_pred != y_true:
        if y_pred == 1:
            # y_pred = 1 and y_true = 0
            errors_false_positive.append(x)
        else:
            # y_pred = 0 and y_true = 1  
            errors_false_negative.append(x)
            
errors_false_negative = pd.DataFrame(errors_false_negative)
errors_false_positive = pd.DataFrame(errors_false_positive)

errors_false_positive.to_csv('../../kaggle/working/negative_wrongly_predicted_as_positive.csv', index=False)
errors_false_negative.to_csv('../../kaggle/working/positive_wrongly_predicted_as_negative.csv', index=False)