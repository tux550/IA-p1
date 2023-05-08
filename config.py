import warnings
import os
from rich.console import Console

warnings.filterwarnings("ignore", message="elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison")

cwd = os.getcwd()
console = Console()

N_FEATURES = 20
EPSILON = 1e-8
N_CLASSES = 3
NORMAL = 1
SUSPECT = 2
PATHOLOGIC = 3