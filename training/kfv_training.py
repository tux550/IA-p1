import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from .util import y2matrix

def kfv_train(model, X, y, n_splits=10, display=False, display_loss=False, save_loss=False, save_dir=None):
    # Init results
    ls_loss = []
    ls_accuaracy = []
    ls_auc       = []
    ls_precision = []
    ls_recall    = []
    ls_f1        = []

    # KFold
    kf             = KFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=42
                    )
    
    # KFold Train-Test loop
    for train_index, test_index in kf.split(X):
        # Split into training and testing
        cX_train, cX_test = X[train_index], X[test_index]
        cy_train, cy_test = y[train_index], y[test_index]
        
        # Train model
        loss = model.fit(cX_train, cy_train)
        ls_loss.append(loss)

        # Prediction
        y_pred = model.predict(cX_test) #print(np.unique(y_pred))
        # METRICS: Accuracy
        accuracy_score = balanced_accuracy_score(cy_test,y_pred)
        ls_accuaracy.append(accuracy_score)
        # METRICS: Precision
        precision = precision_score(cy_test,y_pred, average=None)
        ls_precision.append(precision)
        # METRICS: Recall
        recall = recall_score(cy_test,y_pred, average=None)
        ls_recall.append(recall)
        # METRICS: F1
        f1 = f1_score(cy_test,y_pred, average=None)
        ls_f1.append(f1)

        # Score
        #accuracy_score = model.score(cX_test, cy_test)
        #ls_accuaracy.append(accuracy_score)

        # AUC
        y_matrix    = y2matrix(cy_test)
        prob_matrix = model.class_prob(cX_test)
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

    # best scores
    #best_accuracy_score = max(ls_accuaracy)
    #best_auc_score = max(ls_auc)

    if display:
        print("--- K-Fold Cross-Validation Results ---")
        print(f"Mean Accuracy: {mean_accuracy_score}")
        print(f"Mean AUC: {mean_auc_score}")
        print(f"Mean Precision: {mean_precision}")
        print(f"Mean Recall: {mean_recall}")
        print(f"Mean F1: {mean_f1}")

    if(loss):
        # Average loss 
        ls_loss = np.array(ls_loss)
        ls_loss = np.mean(ls_loss,axis=0)
        if display_loss:
            model.Display(ls_loss, title=f"{model.name}: Average Loss (K-Fold)", save=False, show=True, save_name=None)
        if(save_loss and save_dir):
            model.Display(ls_loss, title=f"{model.name}: Average Loss (K-Fold)", save=True, show=False, save_name=f"{save_dir}/loss_{model.name}_kfold.png")

    return mean_accuracy_score, mean_auc_score, mean_precision, mean_recall, mean_f1
