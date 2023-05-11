import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from .util import y2matrix

def bootstrap_train(model, X, y, n_bootstraps=50, display=False, display_loss=False, save_loss=False, save_dir=None):
    # Init results
    ls_loss = []
    ls_accuaracy = []
    ls_auc       = []
    ls_precision = []
    ls_recall    = []
    ls_f1        = []

    # Bootstrap Train-Test loop
    rng = np.arange(len(y))
    
    for i in range(n_bootstraps):        
        # Generate a bootstrap sample with replacement
        ind = resample(rng, replace=True, n_samples=len(y), random_state = i)
        mask = np.zeros(len(rng),dtype=bool)
        mask[ind] = True

        bX_train = X[mask]
        by_train = y[mask]
        bX_test = X[~mask]
        by_test = y[~mask]
        
        if len(bX_test) == 0:
            continue

        # Train model
        loss = model.fit(bX_train, by_train)
        ls_loss.append(loss)

        # Prediction
        y_pred = model.predict(bX_test)
        # METRICS: Accuracy
        accuracy_score = balanced_accuracy_score(by_test,y_pred)
        ls_accuaracy.append(accuracy_score)
        # METRICS: Precision
        precision = precision_score(by_test,y_pred, average=None)
        ls_precision.append(precision)
        # METRICS: Recall
        recall = recall_score(by_test,y_pred, average=None)
        ls_recall.append(recall)
        # METRICS: F1
        f1 = f1_score(by_test,y_pred, average=None)
        ls_f1.append(ls_f1)

        # Score
        #accuracy_score = model.score(bX_test, by_test)
        #ls_accuaracy.append(accuracy_score)

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
    ls_f1 = np.array(ls_f1)
    mean_precision = ls_precision.sum(axis=0) / len(ls_precision)
    mean_recall = ls_recall.sum(axis=0) / len(ls_recall)
    mean_f1 = ls_f1.sum(axis=0) / len(ls_f1)

    if display:
        print("--- Bootstrap Results ---")
        print(f"Mean Accuracy: {mean_accuracy_score}")
        print(f"Mean AUC: {mean_auc_score}")
        print(f"Mean Precision: {mean_precision}")
        print(f"Mean Recall: {mean_recall}")
        print(f"Mean F1: {mean_f1}")

    if(loss): 
        ls_loss = np.array(ls_loss)
        ls_loss = np.mean(ls_loss,axis=0)
        if display_loss:
            model.Display(ls_loss, title=f"{model.name}: Average Loss (Bootstrap)", save=False, show=True, save_name=None)
        if(save_loss and save_dir):
            model.Display(ls_loss, title=f"{model.name}: Average Loss (Bootstrap)", save=True, show=False, save_name=f"{save_dir}/loss_{model.name}_bootstrap.png")


    return mean_accuracy_score, mean_auc_score, mean_precision, mean_recall, mean_f1


    
    


