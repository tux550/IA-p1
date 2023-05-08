import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from .util import y2matrix

def bootstrap_train(model, X, y, n_bootstraps=50):
    # Init results
    ls_accuaracy = []
    ls_auc       = []
    ls_precision = []
    ls_recall    = []
    ls_f1        = []

    # Bootstrap Train-Test loop
    bX_train, bX_test, by_train, by_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for i in range(n_bootstraps):
        # Generate a bootstrap sample with replacement
        X_boot, y_boot = resample(bX_train, by_train, random_state = i)

        # Train model
        model.fit(X_boot, y_boot)

        # Prediction
        y_pred = model.predict(bX_test)
        # METRICS: Precision
        precision = precision_score(by_test,y_pred, average=None)
        ls_precision.append(precision)
        # METRICS: Recall
        recall = recall_score(by_test,y_pred, average=None)
        ls_recall.append(recall)
        # METRICS: F1
        f1 = f1_score(by_test,y_pred, average=None)
        ls_f1.append(f1)

        # Prediction accuracy
        accuracy_score = model.score(bX_test, by_test)
        ls_accuaracy.append(accuracy_score)

        # AUC
        y_matrix    = y2matrix(by_test)
        prob_matrix = model.class_prob(bX_test)
        auc_score = roc_auc_score(y_matrix, prob_matrix)
        ls_auc.append(auc_score)

    # mean scores
    mean_accuracy_score = sum(ls_accuaracy) / len(ls_accuaracy)
    mean_auc_score = sum(ls_auc) / len(ls_auc)
    ls_precision = np.array(ls_precision)
    ls_recall = np.array(ls_recall)
    ls_f1 = np.array(f1)
    mean_precision = ls_precision.sum(axis=0) / len(ls_precision)
    mean_recall = ls_recall.sum(axis=0) / len(ls_recall)
    mean_f1 = ls_f1.sum(axis=0) / len(ls_f1)

    # best scores
    #best_accuracy_score = max(ls_accuaracy)
    #best_auc_score = max(ls_auc)

    print("--- Bootstrap Results ---")
    print(f"Mean Accuracy: {mean_accuracy_score}")
    print(f"Mean AUC: {mean_auc_score}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1: {mean_f1}")
    
    


