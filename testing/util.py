import itertools
import numpy as np
from rich.console import Console
from rich.table import Table
from config import *
from training import run_metrics


def dict_product(dicts):
    return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))

def test_params(Model, X, y, dict_params):
    # Not used
    parameter_iterations = dict_product(dict_params)
    for model_args in parameter_iterations:
        m = Model(**model_args)
        run_metrics(m, X, y)


METHODS=["K-Fold Cross Validation", "Bootstrap"]
METRICS=["Accuracy","AUC","Precision","Recall","F1"]

def test_param(Model, X, y, p_name, p_ls, params_dict, db_name):    
    p_results = {}
    
    for p in p_ls:
        model_args         = params_dict
        model_args[p_name] = p
        print("Running:", model_args)
        m = Model(**model_args)
        metrics = run_metrics(m, X, y, display_metrics=False)
        p_results[p] = metrics

    for method in METHODS:
        title = f"{Model.__name__} Test Parameter: {p_name} - {method} ({db_name})"
        table = Table(title=title)
        table.add_column(f"{p_name}", justify="right", style="cyan", no_wrap=True)
        for metric in METRICS:
            table.add_column(f"{metric}", justify="center", style="green")
        for p in p_results:
            row_params = [str(p),]
            for metric in METRICS:
                res = p_results[p][method][metric]
                if type(res) in (np.float32, np.float64):
                    val = f"{res:.5f}"
                else:
                    val = f"{res[0]:.5f}"
                    for i in range(1,len(res)):
                        val += f" | {res[i]:.5f}"
                row_params.append(val)
            table.add_row(*row_params)
        console = Console(record=True)
        console.print(table)
        console.save_svg(f"{IMG_FOLDER}/console_{db_name}_{Model.__name__}_{method}_{p_name}.svg")


