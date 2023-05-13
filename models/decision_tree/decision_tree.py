import numpy as np
import math
from .nodo  import Nodo

class DecisionTree:
    def __init__(self, name="DecisionTree", max_depth=None):
        self.name      = name
        self.root      = None
        self.classes   = None
        self.max_depth = max_depth

    # FIT
    def fit(self, X, y):
        return self.Train(X, y)

    # PREDICT
    def predict(self, X):
        Y = []
        # Para cada X encontrar el label del nodo hoja
        for x in X:
            # Inicializar en nodo raiz
            node = self.root
            # Recorrer hasta nodo hoja
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
        # Para cada X encontrar las probabilidades del nodo hoja
        for x in X:
            # Inicializar en nodo raiz
            node = self.root
            # Recorrer hasta nodo hoja
            while node.label is None:
                if x[node.index] < node.boundary:
                    node = node.lt_child
                else:
                    node = node.ge_child
            # Guardar Probabilidad de nodo hoja
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
        # Crear nodo raiz (con metodo recursivo)
        self.root = Nodo(X,y,self.classes, max_depth=self.max_depth, depth=0)
        return None


