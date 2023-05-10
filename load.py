import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from config import *

def load_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target.reshape(-1,1)
    scaler  = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

def load_dataset(N = N_FEATURES):
    training_df = pd.read_csv(cwd + "/dataset/training.csv")
    testing_df  = pd.read_csv(cwd + "/dataset/testing.csv")

    x_train = training_df.iloc[:, :N].values # (n,m)
    y_train = training_df.iloc[:, -1].values # (n,)
    x_test  = testing_df.iloc[:, :N].values  # (n,m)

    if N == 1:
        x_train = x_train.reshape(-1, 1)
        x_test  = x_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)    # (n,1)

    scaler  = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.fit_transform(x_test)
    return x_train, x_test, y_train