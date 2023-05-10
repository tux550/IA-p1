import numpy as np
import random
from config import *
from testing import test_all
from load import load_dataset, load_iris


# VISUALIZACION PENDIENTE
#TODO: Display training (Promedio de loss: SVM y Regression logistica)

# TEST
# TODO: Test between models
#TODO: Confusion matrix (solo en el mejor de cada modelo)

# Extra
# TODO: Test batch size
#TODO: Fix SVM/Reg Auc

np.set_printoptions(precision = 5, floatmode="fixed", suppress = True)

# LOAD DATASET
x_iris, y_iris = load_iris()
x_train, x_test, y_train = load_dataset(N=20)


# TRAIN & TEST
test_all(x_iris, y_iris, "Iris", seed=42)
#test_all(x_train, y_train, "DB", seed=42)
