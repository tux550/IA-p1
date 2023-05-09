import numpy as np
import random
from config import *
from models import MultipleLogisticRegression, DecisionTree, KNN, MultipleSoftSVM, SimpleSoftSVM
from load import load_dataset
from training import kfv_train, bootstrap_train, run_metrics

# VISUALIZACION PENDIENTE
#TODO: Display training
# METRICAS PENDIENTES
#TODO: Fix SVM Auc
#TODO: Knn con distancias distintas
#TODO: Confusion matrix (?)
# TEST
# TODO: Test models

# SEED FOR DETERMINISTIC RESULTS
# "Answer to the Ultimate Question of Life, the Universe, and Everything"
random.seed(42)
np.random.seed(42) 

# LOAD DATASET
x_train, x_test, y_train = load_dataset(N=20)



# MODEL CONFIG
# 1) MLR
epochs = 1000 #10000
alpha = 0.15
mlr = MultipleLogisticRegression(epochs=epochs, alpha=alpha)
# 2) DT
dt = DecisionTree()
# 3) KNN
knn = KNN()
# 4) SVM
svm = MultipleSoftSVM(epochs=1500, alpha=0.0001, c=10)



# TRAIN-TEST MODEL
#"""
print("Running ...") 
run_metrics(svm,x_train,y_train,n_splits=2, n_bootstraps=2)
run_metrics(knn, x_train, y_train)
run_metrics(dt, x_train, y_train)
run_metrics(mlr, x_train, y_train)
#"""

