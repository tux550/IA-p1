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


    # CLASS PROB
    def class_prob(self, X):
        # TODO: Plat scaling &  MLE (Maximum Likelihood estimator)
        p        = self.prob(X)
        prob_mat = np.concatenate([1-p,p], axis=1)
        return prob_mat
    

    # Prob de y=1
    def prob(self, X):
        val = self.Hiperplano(X)
        prob = self.Sigmoid(val) 
        return prob.reshape(-1,1)

    # Funciones de training    
    def Train(self, X, y):
        self.w    = np.random.rand(X.shape[1]) # (20,)
        self.bias = np.random.random() 
        y = y.reshape(-1) # (n,)
        Losses = []
        for ep in range(self.epochs):
            loss = self.Loss(X, y)
            if ep % 1000 == 0:
                print(f'Epoch {ep}, loss {loss}')
            Losses.append(loss)
            self.Update(X, y)
            """
            for idx, x_i in enumerate(X):
                self.Update(x_i, y[idx])
            """
        return Losses
    
    # Funciones de Modelo
    def Hiperplano(self, x):
        Hiperplano = x @ self.w + self.bias
        return Hiperplano
    
    def Loss(self, x, y):
        distances = y * (self.Hiperplano(x))
        err       = np.maximum(0, 1 - distances)
        loss      = (0.5 * np.dot(self.w, self.w)) + (self.c * np.sum(err))
        return loss
    
    def Derivatives(self, x, y):
        dw = self.w
        db = 0
        for x_i, y_i in zip(x,y):
            pred = y_i*self.Hiperplano(x_i)
            if pred < 1:
                dw = dw - self.c*y_i*x_i
                db = db - self.c*y_i
        return dw, db

        """
        yh  = y * self.Hiperplano(x)
        if (yh > 1):
            dw = self.w
            db = 0
        else:
            dw = self.w - x * y * self.c
            #dw = self.w - np.dot(x,y) * self.c
            db = -y * self.c
        return dw, db
        """

    def Update(self, X, y):
        dw , db = self.Derivatives(X,y)
        self.w    = self.w    - self.alpha * dw
        self.bias = self.bias - self.alpha * db

    """
    def Update(self, x, y):
        dw , db = self.Derivatives(x,y)
        self.w    = self.w    - self.alpha * dw
        self.bias = self.bias - self.alpha * db
    """


    # Funciones Extra
    def Sigmoid(self, y):
        sigmoid     = 1 / (1 + np.exp(y))
        return sigmoid