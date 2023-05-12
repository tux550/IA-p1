import numpy as np
import pandas as pd 
import time
import random
from config import *
from testing import test_all, test_logistic, test_svm, compare_models
from load import load_dataset, load_iris
from training import *
from models import *

np.set_printoptions(precision = 5, floatmode="fixed", suppress = True)

# LOAD DATASET
x_train, x_test, y_train = load_dataset(N=20)

# TESTING
test_all(x_train, y_train, "DB", seed=42)
compare_models(x_train, y_train, "DB", seed=42)

# SET SEED
random.seed(42)
np.random.seed(42) 

# BEST
dt = DecisionTree(max_depth=None)
dt.fit(x_train, y_train)
prediction  = dt.predict(x_test).astype(int)
linecounter = np.arange(len(prediction))
res = np.array([linecounter, prediction.reshape(-1)]).T
np.savetxt("output.csv",res,delimiter=" ",fmt='%d')

#"""
#Debug with iris
#x_iris, y_iris = load_iris()
#test_all(x_iris, y_iris, "Iris", seed=42)
#compare_models(x_iris, y_iris, "Iris", seed=42)
#"""
