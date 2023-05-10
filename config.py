import warnings
import os
from rich.console import Console
from sklearn.exceptions import UndefinedMetricWarning

# Supress warning caused by no instance of a class existing in y_pred
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
#warnings.filterwarnings("ignore", message="elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison")

cwd = os.getcwd()

IMG_FOLDER = "imgs"
N_FEATURES = 20
EPSILON = 1e-8
N_CLASSES = 3
NORMAL = 1
SUSPECT = 2
PATHOLOGIC = 3