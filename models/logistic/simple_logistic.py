import numpy as np
import math

class SimpleLogisticRegression:
    def __init__(self, epochs, alpha, name="SimpleLogisticModel", epsilon=1e-8):
        self.w       = None
        self.epochs  = epochs
        self.alpha   = alpha
        self.epsilon = epsilon

    # FIT
    def fit(self, X, y):
        return self.Train(X, y)

    # PREDICT
    def predict(self, X):
        X          = self.add_bias(X)
        prediction = self.Sigmoid(X)
        rounded    = np.round(prediction).astype(int)
        prediction = np.where(rounded > 0.5, 1, 0)
        return prediction
    
    # SCORE
    def score(self, X, y):
        prediction = self.predict(X)
        accuracy   = np.sum(prediction == y) / len(y)
        return accuracy

    # Funciones de Utilidad
    def gen_w(self, N):
        w = [np.random.rand() for i in range(0, N)]
        w = np.array(w)
        w = w.reshape((N, 1))
        return w    

    def gen_bucket(self, X, y):
        n = len(X)
        bucket_size = int(n*0.1)
        sample_rows = np.random.choice(X.shape[0], size = bucket_size, replace = False)
        cX = X[sample_rows, :]
        cy = y[sample_rows, :]
        return cX, cy
    
    def add_bias(self, X):
        return np.insert(X, 0, 1, axis = 1) # adding the bias to X as first column

    # Funciones de Modelo
    def Hiperplano(self, X):
        Hiperplano =  np.dot(X, self.w)
        return Hiperplano   

    def Sigmoid(self, X):
        hiperplano  = self.Hiperplano(X)
        sigmoid     = 1 / (1 + np.exp(-hiperplano))
        return sigmoid

    def Loss(self, X, y):
        n           = len(X)
        sigmoid     = self.Sigmoid(X)
        LeftSum     = y * np.log(sigmoid + self.epsilon)
        RightSum    = (1 - y) * np.log((1 - sigmoid) + self.epsilon)
        Loss        = - (1 / n) * np.sum(LeftSum+RightSum)
        return Loss

    def Derivatives(self, X, y):
        y_pred      = self.Sigmoid(X)
        y_rest      = y - y_pred
        n           = len(y)
        dw  = (1/n) * np.matmul(np.transpose(y_rest), -X)
        return dw

    # Funciones de training
    def Update(self, X, y):
        dw     = self.Derivatives(X, y)
        self.w = self.w - (self.alpha * (dw.T))
    
    def Train(self, X, y):
        X      = self.add_bias(X)
        self.w = self.gen_w(X.shape[1])
        Losses = []
        for _ in range(self.epochs):
            cX, cy = self.gen_bucket(X, y)
            loss   = self.Loss(cX, cy)
            Losses.append(loss)
            self.Update(cX, cy)
        return Losses

    # Funciones de Display
    def Display(self, losses):
        #TODO
        pass
