import numpy as np
from sklearn.neighbors import KDTree

class KNN:
    def __init__(self, distance="minkowski", name="KNN", k=3):
        self.name     = name
        self.x_tree   = None
        self.y_train  = None
        self.classes  = None
        self.distance = distance
        self.k        = k

    # FIT
    def fit(self, X, y):
        return self.Train(X, y)

    # PREDICT
    def predict(self, X):
        predictions = []
        for x in X:
            x = x.reshape(1,-1)
            # Get classes of knn
            _, ind    = self.x_tree.query(x, k=self.k)
            pred_classes = np.array([self.y_train[i] for i in ind])
            # Vote
            prediction, _ = KNN.MostCommon(pred_classes, self.classes)
            predictions.append(prediction)
        return np.array(predictions).reshape(-1,1)
    
    # SCORE
    def score(self, X, y):
        prediction = self.predict(X)
        accuracy   = np.sum(prediction == y) / len(y)
        return accuracy
    
    # CLASS PROB
    def class_prob(self, X):
        probs = []
        for x in X:
            x = x.reshape(1,-1)
            # Get classes of knn
            _, ind    = self.x_tree.query(x, k=self.k)
            pred_classes = np.array([self.y_train[i] for i in ind])
            # Vote
            _, prob = KNN.MostCommon(pred_classes, self.classes)
            probs.append(prob)
        return np.array(probs)
    

    # Funciones de Utilidad
    def get_classes(self, y):
        return np.unique(y)

    def MostCommon(Y, classes):
        # return most common label
        values, counts = np.unique(Y, return_counts=True)
        total = counts.sum()
        cls_count = dict()
        for v,c in zip(values, counts):
            cls_count[v] = c
        probs = [cls_count[cls]/total if cls in cls_count else 0 for cls in classes]
        ind = np.argmax(counts)
        return values[ind], probs

    # Funciones de training    
    def Train(self, X, y):
        # Classes
        self.classes = self.get_classes(y)
        # Store train_dataset in KDTree
        self.x_tree  = KDTree(X, metric=self.distance) 
        self.y_train = y
        # Return None
        return None




