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
        # Add bias to X
        X = self.add_bias(X)
        # Get prediction for each class
        predictions = []
        for m in self.models:
            m_pred = m.Sigmoid(X)
            predictions.append(m_pred)
        # Get class with biggest probability for each point
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
        # Add bias to X
        X = self.add_bias(X)
        # Get probabilities for each class
        predictions = []
        for m in self.models:
            m_pred = m.Sigmoid(X)
            predictions.append(m_pred)
        # Softmax proabilities of classes for each point
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
        # Entrenar un SimpleLogistic por cada clase
        for cls in self.classes:
            # Transformar Y a un arreglo de OvR para la clase "cls"
            y_prime = (y == cls).astype(int)
            # Train model
            model   = SimpleLogisticRegression(self.epochs, self.alpha, self.epsilon)
            losses = model.Train(X, y_prime)
            self.models.append(model)
            all_losses.append(losses)
        return all_losses

    # Funciones de Display
    def Display(self, all_losses, title='Loss for all classes', save=False, show=True, save_name=None):
        fig, ax = plt.subplots(figsize=(8,6))
        colors = ['blue', 'red', 'green']
        for i, cls in enumerate(self.classes):
            ax.plot(all_losses[i], color=colors[i], label=f'Class {cls}')
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.tight_layout()
        if show:
            plt.show()
        if save and save_name:
            plt.savefig(save_name)


