import numpy as np
import time
from config import *
from testing import test_all, test_logistic, test_svm, compare_models
from load import load_dataset, load_iris
from training import *
from models import *

np.set_printoptions(precision = 5, floatmode="fixed", suppress = True)

# LOAD DATASET
x_train, x_test, y_train = load_dataset(N=20)


# TRAIN & TEST
test_all(x_train, y_train, "DB", seed=42)
compare_models(x_train, y_train, "DB", seed=42)

#"""
#Debug with iris
#x_iris, y_iris = load_iris()
#test_all(x_iris, y_iris, "Iris", seed=42)
#compare_models(x_iris, y_iris, "Iris", seed=42)
#"""
