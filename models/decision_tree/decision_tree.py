import numpy as np
import math
from .nodo  import Nodo

class DecisionTree:
    def __init__(self, name="DecisionTree"):
        self.name    = name
        self.root    = None
        self.classes = None

    # FIT
    def fit(self, X, y):
        return self.Train(X, y)

    # PREDICT
    def predict(self, X):
        Y = []
        for x in X:
            node = self.root
            while node.label is None:
                if x[node.index] < node.boundary:
                    node = node.lt_child
                else:
                    node = node.ge_child
            y = node.label
            Y.append(y)
        return np.array(Y).reshape(-1,1)
    
    # SCORE
    def score(self, X, y):
        prediction = self.predict(X)
        accuracy   = np.sum(prediction == y) / len(y)
        return accuracy
    
    # CLASS PROB
    def class_prob(self, X):
        predictions = []
        for x in X:
            node = self.root
            while node.label is None:
                if x[node.index] < node.boundary:
                    node = node.lt_child
                else:
                    node = node.ge_child
            y = node.prob
            predictions.append(y)
        return np.array(predictions)
    

    # Funciones de Utilidad
    def get_classes(self, y):
        return np.unique(y)

    # Funciones de training    
    def Train(self, X, y):
        y = y.reshape(-1)
        self.classes = self.get_classes(y)
        self.root = Nodo(X,y,self.classes)
        return None


