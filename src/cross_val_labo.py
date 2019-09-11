import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

class CrossValLabo(object):
    @staticmethod
    def optimize_hparams(model, params, param_set, X_train, Y_train, X_val, Y_val, save=False):
        clf = GridSearchCV(model, param_set, n_jobs=1, 
                    cv=[(
                            np.arange(0, Y_train.shape[0], 1),
                            np.arange(Y_train.shape[0], Y_train.shape[0] + Y_val.shape[0], 1))],
                    scoring='roc_auc',
                    verbose=2, refit=True)
        X = pd.concat((X_train, X_val))
        Y = pd.concat((Y_train, Y_val))
        clf.fit(X, Y)
        best_parameters_val = clf.best_params_
        params = {**params, **best_parameters_val}
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, Y_train)
        preds_train = model.predict_proba(X_train)[:, 1]
        score_train = roc_auc_score(Y_train, preds_train)
        preds_val = model.predict_proba(X_val)[:, 1]
        score_val = roc_auc_score(Y_val, preds_val)
        return best_parameters_val, score_val, score_train, model
