import numpy as np
import math
from .simple_svm import SimpleSoftSVM


class MultipleSoftSVM:
    def __init__(self, epochs, c, alpha, name="SimpleSoftSVM", epsilon=1e-8):
        self.models  = None
        self.classes = None
        self.epochs  = epochs
        self.c       = c
        self.alpha   = alpha
        self.name    = name
        self.epsilon = epsilon

    # FIT
    def fit(self, X, y):
        return self.Train(X, y)

    # PREDICT
    def predict(self, X):
        predictions = []
        for m in self.models:
            m_pred = m.Hiperplano(X) #(m.predict(X)+1)/2            
            predictions.append(m_pred.reshape(-1,1))
        predictions = np.concatenate(predictions, axis=1)
        cls_index   = np.argmax(predictions, axis=1)
        return np.array([self.classes[i] for i in cls_index]).reshape(-1, 1)
    
    # SCORE
    def score(self, X, y):
        prediction = self.predict(X)
        accuracy   = np.sum(prediction == y) / len(y)
        return accuracy
    
    # CLASS PROB
    def class_prob(self, X):
        # TODO: Plat scaling &  MLE (Maximum Likelihood estimator)
        predictions = []
        for m in self.models:
            m_pred = (m.prob(X)+1)/2
            predictions.append(m_pred)
        predictions = np.concatenate(predictions, axis=1)
        added_pred  = np.sum(predictions, axis=1)
        prob        = (predictions / added_pred[:,None])
        return prob
    
    # Funciones de Utilidad
    def get_classes(self, y):
        return np.unique(y)

    # Funciones de training    
    def Train(self, X, y):
        self.classes = self.get_classes(y)
        self.models = []
        all_losses = []
        for cls in self.classes:
            y_prime = ( (y == cls).astype(int)*2 -1) # 1: Pertenece, -1: No pertenece
            model   = SimpleSoftSVM(self.epochs, self.c, self.alpha, epsilon=self.epsilon)
            losses = model.Train(X, y_prime)
            self.models.append(model)
            all_losses.append(losses)
        return all_losses

    # Funciones de Display
    def Display(self, all_losses):
        #TODO
        pass
