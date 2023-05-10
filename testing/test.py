import itertools
from rich.console import Console
from rich.table import Table
from ..training import run_metrics


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

def test_param(Model, X, y, p_name, p_ls, params_dict):    
    p_results = {}
    
    for p in p_ls:
        model_args         = params_dict
        model_args[p_name] = p
        m = Model(**model_args)
        metrics = run_metrics(m, X, y)
        p_results[p] = metrics
    
    for method in METHODS:
        table = Table(title=f"Test Parameter: {p_name} - {method}")
        table.add_column(f"{p_name}", justify="right", style="cyan", no_wrap=True)
        for metric in METRICS:
            table.add_column(f"{metric}", justify="right", style="green")
        for p in p_results:
            row_params = [p,]
            for metric in METRICS:
                row_params.append(p_results[p][METHODS][METRICS])
            table.add_row(*row_params)
        console = Console()
        console.print(table)


