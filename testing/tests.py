import numpy as np
import random
from models import MultipleLogisticRegression, MultipleSoftSVM, KNN, DecisionTree
from .util import test_param, test_models


def test_logistic(X, y, db_name, seed=42):
    random.seed(seed)
    np.random.seed(seed) 
    print("--- TEST LOGISTIC ---")

    print("Testing parameter: epochs")
    def_args = {"alpha":1}
    epochs = [500,1000,1500,2000,2500,3000]
    test_param(MultipleLogisticRegression, X, y, "epochs", epochs, def_args,db_name)

    print("Testing parameter: alpha")
    def_args = {"epochs":3000}
    alphas = [0.0001,0.001,0.01,0.1,1]
    test_param(MultipleLogisticRegression, X, y, "alpha", alphas, def_args,db_name)

def test_svm(X,y,db_name,seed=42):
    random.seed(seed)
    np.random.seed(seed) 
    print("--- TEST SVM ---")

    print("Testing parameter: epochs")
    def_args = {"alpha":0.001, "c":20}
    epochs = [500,1000,1500,2000,2500,3000]
    test_param(MultipleSoftSVM, X, y, "epochs", epochs, def_args,db_name)

    print("Testing parameter: alpha")
    def_args = {"epochs":3000, "c":20}
    alphas = [0.0001,0.001,0.01]# Valores mayores producen error de overflow ,0.1,1]
    test_param(MultipleSoftSVM, X, y, "alpha", alphas, def_args,db_name)

    print("Testing parameter: c")
    def_args = {"epochs":3000, "alpha":0.001}
    cs =  [0.5,1,5,10,20,30]
    test_param(MultipleSoftSVM, X, y, "c", cs, def_args,db_name)



def test_knn(X,y,db_name,seed=42):
    random.seed(seed)
    np.random.seed(seed) 
    print("--- TEST KNN ---")

    print("Testing parameter: distance")
    def_args = {}
    distance = ["minkowski","cityblock","chebyshev"]
    test_param(KNN, X, y, "distance", distance, def_args,db_name)

def test_dt(X,y,db_name,seed=42):
    random.seed(seed)
    np.random.seed(seed) 
    print("--- TEST DECISION TREE ---")

    print("Testing parameter: max_depth")
    def_args = {}
    max_depth = [3,5,7,10,None]
    test_param(DecisionTree, X, y, "max_depth", max_depth, def_args,db_name)

def test_all(X,y,db_name,seed=42):
    test_knn(X,y,db_name,seed=seed)
    test_svm(X,y,db_name,seed=seed)
    test_logistic(X,y,db_name,seed=seed)
    test_dt(X,y,db_name,seed=seed)

def compare_models(X, y, db_name, seed=42):
    ls_models = [
        MultipleLogisticRegression(alpha=1, epochs=3000),
        MultipleSoftSVM(alpha=0.0001, c=1, epochs=3000),
        KNN(distance="cityblock"),
        DecisionTree(max_depth=None)
    ]
    random.seed(seed)
    np.random.seed(seed) 
    test_models(ls_models,X,y,db_name)
