import numpy as np
import time
from config import *
from testing import test_all, test_logistic, compare_models
from load import load_dataset, load_iris
from training import *
from models import *

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
#x_iris, y_iris = load_iris()
x_train, x_test, y_train = load_dataset(N=20)


# TRAIN & TEST
test_all(x_train, y_train, "DB", seed=42)
compare_models(x_train, y_train, "DB", seed=42)

"""
Debug with iris
#test_all(x_iris, y_iris, "Iris", seed=42)
compare_models(x_iris, y_iris, "Iris", seed=42)
"""

"""
# Debug testing
np.random.seed(42)
start = time.time()
m = MultipleSoftSVM(epochs=5000, alpha=0.001, c=20)
bootstrap_train(m, x_train, y_train, n_bootstraps=1, display=True)
end = time.time()
print("TIME:",end - start)
"""