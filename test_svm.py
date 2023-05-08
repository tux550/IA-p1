import numpy as np
import random
from config import *
from models import MultipleSoftSVM, SimpleSoftSVM
from load import load_dataset
from training import kfv_train, bootstrap_train, run_metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#random.seed(42)
#np.random.seed(42) 


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:75, :2]  # we only take the first two features.
y = iris.target[:75] # Only 0 and 1 class (100)

scaler  = MinMaxScaler()
X = scaler.fit_transform(X)
y = (y==1) * 2 -1
#print(X)
#print(y)


# SVM example
svm = SimpleSoftSVM(epochs=2000, alpha=0.0001, c=10)
dx_train, dx_test, dy_train, dy_test = train_test_split(X, y, test_size=0.2, random_state=43)
print(dy_train)

svm.fit(dx_train, dy_train)
y_pred = svm.predict(dx_test)
cm = confusion_matrix(y_pred, dy_test)
print(cm)
print(svm.w)
print(svm.bias)