import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

class CrossValLabo(object):
    @staticmethod
    def optimize_hparams(model, param_set, X_train, Y_train, X_val, Y_val, save=False):
        clf = GridSearchCV(model, param_set, n_jobs=1, 
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
        
    @staticmethod
    def quick_lgbm(param_set, X_train, Y_train, X_val, Y_val):
        tr_data = lgb.Dataset(X_train, label=Y_train)
        vl_data = lgb.Dataset(X_val, label=Y_val)  

        mdl = lgb.LGBMRegressor(
            boosting_type='gbdt',
            objective='regression',
            metric='auc',
            learning_rate=0.01,
            num_leaves=2**4,
            max_depth=5,
            tree_learner='serial',
            colsample_bytree=0.80,
            subsample_freq=1,
            subsample=1,
            n_estimators=2**10,
            max_bin=255,
            verbose=-1,
            seed=1337,
            early_stopping_rounds=100,
            reg_alpha=0.3,
            reg_lamdba=0.243,
            n_jobs = -1)

        estimator = lgb.train(
            param_set,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )
