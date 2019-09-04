import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

def train_eval_model(features):
    features.to_csv('interm_features.csv')
    cols = list(features.columns)
    cols = [c for c in cols if c not in ['TransactionID', 'isFraud', 'TransactionDT']]
    X = features[cols]
    Y = features.isFraud
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)
    xgb_model = XGBClassifier()
    parameters = {
        'nthread':[2], #when use hyperthread, xgboost may become slower
        'objective':['binary:logistic'],
        'learning_rate': [0.6], # from 0.1
        'max_depth': [4], # opti 20 but slow
        'min_child_weight': [15], # from 4
        'silent': [1],
        'subsample': [0.8],
        'colsample_bytree': [0.8], # from 0.8
        'n_estimators': [200], #number of trees, change it to 1000 for better results
        'missing':[-999],
        'seed': [1337]}
    clf = GridSearchCV(xgb_model, parameters, n_jobs=4, 
                   cv=4,
                   scoring='roc_auc',
                   verbose=2, refit=True)
    clf.fit(X, Y)
    best_parameters, score, model = clf.best_params_, clf.best_score_, clf.best_estimator_
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(Y_test, preds)
    return model, auc
