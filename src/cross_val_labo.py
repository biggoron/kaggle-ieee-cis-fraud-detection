import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

class CrossValLabo(object):
    @staticmethod
    def optimize_hparams(model, param_set, X_train, Y_train, X_val, Y_val, save=False):
        clf = GridSearchCV(model, param_set, n_jobs=2, 
                    cv=[(
                            np.arange(0, Y_train.shape[0], 1),
                            np.arange(Y_train.shape[0], Y_train.shape[0] + Y_val.shape[0], 1))],
                    scoring='roc_auc',
                    verbose=2, refit=True)
        X = pd.concat((X_train, X_val))
        Y = pd.concat((Y_train, Y_val))
        clf.fit(X, Y)
        best_parameters_val, score_val, model = clf.best_params_, clf.best_score_, clf.best_estimator_
        preds_train = model.predict_proba(X_train)[:, 1]
        score_train = roc_auc_score(Y_train, preds_train)
        return best_parameters_val, score_val, score_train, model
        
