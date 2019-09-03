from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score

def train_eval_model(features):
    features.to_csv('interm_features.csv')
    cols = list(features.columns)
    cols = [c for c in cols if c not in ['TransactionID', 'isFraud', 'TransactionDT']]
    X_train = features[features.TransactionDT < 13000000][cols]
    Y_train = features[features.TransactionDT < 13000000]["isFraud"]
    X_val = features[features.TransactionDT >= 13000000][cols]
    Y_val = features[features.TransactionDT >= 13000000]["isFraud"]
    params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 1.0,
    'n_estimators': 100}
    model = XGBClassifier(**params).fit(X_train, Y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(Y_val, preds)
    return model, auc
