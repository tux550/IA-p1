import numpy as np
import random
from models import MultipleLogisticRegression, MultipleSoftSVM, KNN, DecisionTree
from .util import test_param


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

    print("Testing parameter: c")
    def_args = {"epochs":300, "alpha":0.001}
    cs =  [0.5,1,5,10,20,30]
    test_param(MultipleSoftSVM, X, y, "c", cs, def_args,db_name)

    print("Testing parameter: alpha")
    def_args = {"epochs":300, "c":20}
    alphas = [0.0001,0.001,0.01]# Valores mayores producen error de overflow ,0.1,1]
    test_param(MultipleSoftSVM, X, y, "alpha", alphas, def_args,db_name)

    print("Testing parameter: epochs")
    def_args = {"alpha":0.001, "c":20}
    epochs = [100,150,200,250,300]
    test_param(MultipleSoftSVM, X, y, "epochs", epochs, def_args,db_name)

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

