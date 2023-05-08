import numpy as np

class SimpleSoftSVM:
    def __init__(self, epochs, c, alpha, name="SimpleSoftSVM", epsilon=1e-8):
        self.w       = None
        self.bias    = None
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
        val = self.Hiperplano(X)
        y_pred = np.sign(val) 
        return y_pred.reshape(-1,1)
    
    # SCORE
    def score(self, X, y):
        prediction = self.predict(X)
        accuracy   = np.sum(prediction == y) / len(y)
        return accuracy

    # Prob de y=1
    def prob(self, X):
        val = self.Hiperplano(X)
        prob = self.Sigmoid(val) 
        return prob.reshape(-1,1)

    # Funciones de training    
    def Train(self, X, y):
        self.w    = np.random.rand(X.shape[1])
        self.bias = np.random.random()
        y = y.reshape(-1)
        Losses = []
        for _ in range(self.epochs):
            loss = self.Loss(X, y)
            Losses.append(loss)
            for idx, x_i in enumerate(X):
                self.Update(x_i, y[idx])
        return Losses
    
    # Funciones de Modelo
    def Hiperplano(self, x):
        Hiperplano = np.dot(x, self.w) + self.bias
        return Hiperplano
    
    def Loss(self, x, y):
        distances = y * (self.Hiperplano(x))
        err       = np.maximum(0, 1 - distances)
        loss      = (0.5 * np.linalg.norm(self.w)**2) + (self.c * np.sum(err))
        return loss
    
    def Derivatives(self, x, y):
        yh  = y * self.Hiperplano(x)
        if (yh > 1):
            dw = self.w
            db = 0
        else:
            dw = self.w - np.dot(x,y) * self.c
            db = - y * self.c
        return dw, db

    def Update(self, x, y):
        dw , db = self.Derivatives(x,y)
        self.w    = self.w - self.alpha * dw
        self.bias = self.bias - self.alpha * db
    
    # Funciones Extra
    def Sigmoid(self, y):
        sigmoid     = 1 / (1 + np.exp(y))
        return sigmoid