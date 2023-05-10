
from .kfv_training import kfv_train
from .bootstrap_training import bootstrap_train
from rich.console import Console
from rich.table import Table

console = Console()

def run_metrics(model, x_train, y_train, n_splits=10, n_bootstraps=50, display_metrics=True, return_metrics=True):
    k_accuracy, k_auc, k_precision, k_recall, k_f1 = kfv_train(model, x_train, y_train, n_splits=n_splits)
    b_accuracy, b_auc, b_precision, b_recall, b_f1 = bootstrap_train(model, x_train, y_train, n_bootstraps=n_bootstraps)

    # Metrics
    kfold_metrics     = {
        "Accuracy"  : k_accuracy,
        "AUC"       : k_auc,
        "Precision" : k_precision,
        "Recall"    : k_recall,
        "F1"        : k_f1
    }
    bootstrap_metrics = {
        "Accuracy"  : b_accuracy,
        "AUC"       : b_auc,
        "Precision" : b_precision,
        "Recall"    : b_recall,
        "F1"        : b_f1
    }
    metrics = {
        "K-Fold Cross Validation" : kfold_metrics,
        "Bootstrap"               :bootstrap_metrics
    }

    # Format
    k_accuracy = str( k_accuracy)
    k_auc = str( k_auc)
    k_precision = str( k_precision)
    k_recall = str( k_recall)
    k_f1 = str( k_f1)
    b_accuracy = str( b_accuracy)
    b_auc = str( b_auc)
    b_precision = str( b_precision)
    b_recall = str( b_recall)
    b_f1 = str( b_f1)

    # Display Metrics
    if display_metrics:
        table = Table(title=f"Metrics: {model.name}")

        table.add_column("Training Method", justify="right", style="cyan", no_wrap=True)
        table.add_column("Accuracy", justify="right", style="green")
        table.add_column("AUC", justify="right", style="green")
        table.add_column("Precision", justify="right", style="green")
        table.add_column("Recall", justify="right", style="green")
        table.add_column("F1", justify="right", style="green")

        table.add_row("K-Fold Cross-Validation", k_accuracy, k_auc, k_precision, k_recall, k_f1)
        table.add_row("Bootstrap", b_accuracy, b_auc, b_precision, b_recall, b_f1)

        console = Console()
        console.print(table)
    
    # Return metrics
    return metrics

