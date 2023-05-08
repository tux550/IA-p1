import numpy as np
def y2matrix(y):
    classes = np.unique(y)
    y_all = []
    for cls in classes:
        y_prime = (y == cls).astype(int)
        y_all.append(y_prime)
    return np.concatenate(y_all, axis=1)

def get_classes(y):
    return np.unique(y)