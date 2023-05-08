import numpy as np
import random
from config import *
from models import MultipleLogisticRegression, DecisionTree
from load import load_dataset
from training import kfv_train, bootstrap_train

# MODELOS PENDIENTE
#TODO: KNN
#TODO: SVM
# VISUALIZACION PENDIENTE
#TODO: Display training
# METRICAS PENDIENTES
#TODO: Confusion matrix (?)

# SEED FOR DETERMINISTIC RESULTS
# "Answer to the Ultimate Question of Life, the Universe, and Everything"
random.seed(42)
np.random.seed(42) 

# LOAD DATASET
x_train, x_test, y_train = load_dataset()



# MODEL CONFIG
# 1) MLR
epochs = 1000 #10000
alpha = 0.15
mlr = MultipleLogisticRegression(epochs=epochs, alpha=alpha)
# 2) DT
dt = DecisionTree()

# TRAIN MODEL
kfv_train(dt, x_train, y_train, n_splits=10)
bootstrap_train(dt, x_train, y_train, n_bootstraps=50)

# Decision Tree example
"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
dx_train, dx_test, dy_train, dy_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
dt = DecisionTree()
res = dt.fit(dx_train, dy_train)
y_pred = dt.predict(dx_test)
cm = confusion_matrix(y_pred, dy_test)
print(cm)
print(dt.score(dx_test, dy_test))
print(dt.class_prob(dx_test))
"""
