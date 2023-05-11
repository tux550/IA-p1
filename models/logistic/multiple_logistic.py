import numpy as np
import math
import matplotlib.pyplot as plt
from .simple_logistic import SimpleLogisticRegression

class MultipleLogisticRegression:
    def __init__(self, epochs, alpha, name="MultipleLogisticRegression", epsilon=1e-8):
        self.models  = None
        self.classes = None
        self.epochs  = epochs
        self.alpha   = alpha
        self.epsilon = epsilon
        self.name    = name

    # FIT
    def fit(self, X, y):
        return self.Train(X, y)

    # PREDICT
    def predict(self, X):
        X = self.add_bias(X)
        predictions = []
        for m in self.models:
            m_pred = m.Sigmoid(X)
            predictions.append(m_pred)
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
        X = self.add_bias(X)
        predictions = []
        for m in self.models:
            m_pred = m.Sigmoid(X)
            predictions.append(m_pred)
        predictions = np.concatenate(predictions,axis=1)
        added_pred  = np.sum(predictions, axis=1)
        prob        = (predictions / added_pred[:,None])
        return prob
    

    # Funciones de Utilidad
    def get_classes(self, y):
        return np.unique(y)
    
    def add_bias(self, X):
        return np.insert(X, 0, 1, axis = 1) # adding the bias to X as first column

    # Funciones de training    
    def Train(self, X, y):
        self.classes = self.get_classes(y)
        self.models = []
        all_losses = []
        for cls in self.classes:
            y_prime = (y == cls).astype(int)
            model   = SimpleLogisticRegression(self.epochs, self.alpha, self.epsilon)
            losses = model.Train(X, y_prime)
            self.models.append(model)
            all_losses.append(losses)
        return all_losses

    # Funciones de Display
    def Display(self, all_losses):
        fig, axs = plt.subplots(len(self.classes), 1, figsize=(8, 6*len(self.classes)))
        for i, cls in enumerate(self.classes):
            axs[i].plot(all_losses[i])
            axs[i].set_title(f'Loss for class {cls}')
            axs[i].set_xlabel('Iteration')
            axs[i].set_ylabel('Loss')
        plt.tight_layout(pad=5.0)
        plt.show()

