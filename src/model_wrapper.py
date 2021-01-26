import os
import pickle


class ClassifierWrapper():
    def __init__(self, model):
        if os.path.isfile("kaggle/working/gs_classifier.pickle"):
            print("Teaching on training reviews")
            self.model = model
            #self.model.fit()
            print("Dumping")
            self.dump()
        else:
            print("Using precomputed classifier from \"kaggle/working/gs_classifier.pickle\" ")
            self.load()

    def predict(self, param):
        self.model.predict(param)

    def dump(self):
        pickle_out = open("kaggle/working/gs_classifier.pickle", "wb")
        pickle.dump(self.model, pickle_out)
        pickle_out.close()

    def load(self):
        pickle_in = open("kaggle/working/gs_classifier.pickle", "rb")
        self.model = pickle.load(pickle_in)
        print("Loaded")