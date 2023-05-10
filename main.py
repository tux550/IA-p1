import numpy as np
import random
from config import *
from testing import test_logistic, test_svm, test_knn, test_dt, test_all

from models import MultipleLogisticRegression, DecisionTree, KNN, MultipleSoftSVM, SimpleSoftSVM
from load import load_dataset
from training import kfv_train, bootstrap_train, run_metrics

# TEST
# TODO: Test models
# TODO: Test between models
# VISUALIZACION PENDIENTE
#TODO: Display training
# METRICAS PENDIENTES
#TODO: Confusion matrix (?)


# Extra
#TODO: Fix SVM/Reg Auc

# SEED FOR DETERMINISTIC RESULTS
# "Answer to the Ultimate Question of Life, the Universe, and Everything"
random.seed(42)
np.random.seed(42) 

# LOAD DATASET
x_train, x_test, y_train = load_dataset(N=20)

test_all(x_train, y_train)
exit()


test_dt(x_train, y_train)
test_knn(x_train, y_train)
test_logistic(x_train, y_train)
test_svm(x_train, y_train)
exit()












# MODEL CONFIG
# 1) MLR
mlr = MultipleLogisticRegression(epochs=1500, alpha = 0.15) #10000
# 2) DT
dt = DecisionTree()
# 3) KNN
knn = KNN()
# 4) SVM
svm = MultipleSoftSVM(epochs=200, alpha=0.0001, c=10)


# TEST SVM
import time
start = time.time()
kfv_train(mlr,x_train,y_train, n_splits=5, display=True)
end = time.time()
print(end - start)
exit()

# TRAIN-TEST MODEL
#"""
print("Running ...") 
run_metrics(mlr, x_train, y_train)
run_metrics(dt, x_train, y_train)
run_metrics(knn, x_train, y_train)
run_metrics(svm,x_train,y_train)
#"""

