import pandas as pd
import numpy as np

from features import train_date_sin_reg, card_type_dummies, card1_features, device_type_dummies

def preprocess(train=True, test=False):
    train_trs = pd.read_csv('data/train_transaction.csv')
    train_ids = pd.read_csv('data/train_identity.csv')
    used_cols = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'card1', 'card4', 'card6', 'ProductCD']
    train_trs = train_trs[used_cols + ['isFraud']]
    if test:
        test_trs = pd.read_csv('data/test_transaction.csv')
        test_ids = pd.read_csv('data/test_identity.csv')
        test_trs = test_trs[used_cols]
    train_df, test_df = None, None
    if train:
        train_df = preprocess_train(train_trs, train_ids)
    if test:
        train_df = train_df if train else pd.read_csv('train_features.csv')
        test_df = preprocess_test(test_trs, test_ids, train_df)
    return train_df, test_df

def preprocess_train(trs, ids, skip=[]):
    print('train')
    task_id = 'date_sin'
    print(task_id)
    to_merge = []
    if not task_id in skip:
        to_merge += [task_id]
        df = train_date_sin_reg(
            trs, "TransactionDT", "TransactionAmt", 5000).set_index("TransactionID")
        df.to_csv(f"tmp/output/{task_id}.csv")
    task_id = 'amount'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = trs[["TransactionID", "TransactionAmt"]].set_index("TransactionID")
        df["TransactionAmt"] = df.TransactionAmt.astype(np.float32)
        df.to_csv(f"tmp/output/{task_id}.csv")
        print(df.dtypes)
    task_id = 'product_cd_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = pd.get_dummies(
            trs[['TransactionID', 'ProductCD']].fillna('other'),
            columns=['ProductCD'],
            prefix='product_cd'
        ).set_index("TransactionID")
        df.to_csv(f"tmp/output/{task_id}.csv")
        print(df.dtypes)
    task_id = 'card1_features'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = card1_features(trs)
        df.to_csv(f"tmp/output/{task_id}.csv")
        print(df.dtypes)
    task_id = 'card_type_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = card_type_dummies(trs)
        df.to_csv(f"tmp/output/{task_id}.csv")
        print(df.dtypes)
    task_id = 'device_type_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = device_type_dummies(ids)
        df.to_csv(f"tmp/output/{task_id}.csv")
        print(df.dtypes)
    trs_new = trs[["TransactionID", "isFraud"]].copy().set_index("TransactionID")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/date_sin.csv'), on="TransactionID", how="left")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/amount.csv'), on="TransactionID", how="left")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/product_cd_dummies.csv'), on='TransactionID', how='left')
    trs_new = trs_new.merge(pd.read_csv('tmp/output/card_type_dummies.csv'), on='TransactionID', how='left')
    trs_new = trs_new.merge(pd.read_csv('tmp/output/device_type_dummies.csv'), on='TransactionID', how='left')
    trs_new = trs_new.merge(pd.read_csv('tmp/output/card1_features.csv'), on='TransactionID', how='left')
    trs_new = trs_new.merge(trs[['TransactionID', 'TransactionDT']], on='TransactionID', how="left")
    trs_new.to_csv('train_features.csv')
    return trs_new

def preprocess_test(trs, ids, train_df, skip=[]):
    task_id = 'date_sin'
    to_merge = []
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = train_date_sin_reg(
            trs, "TransactionDT", "TransactionAmt", 5000).set_index("TransactionID")
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    task_id = 'amount'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = trs[["TransactionID", "TransactionAmt"]].set_index("TransactionID")
        df["TransactionAmt"] = df.TransactionAmt.astype(np.float32)
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    task_id = 'product_cd_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = pd.get_dummies(
            trs[['TransactionID', 'ProductCD']].fillna('other'),
            columns=['ProductCD'],
            prefix='product_cd'
        ).set_index("TransactionID")
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    task_id = 'card1_features'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        card1_ref = train_df[['card1', 'card1_percent_fraud', 'card1_total_use']].drop_duplicates()
        per_med = card1_ref.card1_percent_fraud.median()
        use_med = card1_ref.card1_total_use.median()
        df = trs[["TransactionID", 'card1']]
        df = df.merge(card1_ref, on='card1', how='left').set_index("TransactionID")
        df['card1_total_use'] = df['card1_total_use'].fillna(use_med)
        df['card1_percent_fraud'] = df['card1_percent_fraud'].fillna(per_med)
        df.drop(columns=["card1"], inplace=True)
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    task_id = 'card_type_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = card_type_dummies(trs)
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    task_id = 'device_type_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = device_type_dummies(ids)
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    trs_new = trs[["TransactionID"]].copy().set_index("TransactionID")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/date_sin.csv'), on="TransactionID", how="left")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/amount.csv'), on="TransactionID", how="left")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/product_cd_dummies.csv'), on='TransactionID', how='left')
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/card_type_dummies.csv'), on='TransactionID', how='left')
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/device_type_dummies.csv'), on='TransactionID', how='left')
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/card1_features.csv'), on='TransactionID', how='left')
    trs_new.to_csv('train_features.csv')
    return trs_new
