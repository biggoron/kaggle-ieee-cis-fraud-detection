import pandas as pd
import numpy as np

from features import \
    get_date_sin_features, get_amount_features, card_type_dummies, device_type_dummies

def preprocess(train=True, test=False, val=True, skip=[]):
    train_trs = pd.read_csv('data/train_transaction.csv')
    train_ids = pd.read_csv('data/train_identity.csv')
    used_cols = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'card4', 'card6', 'ProductCD']
    train_trs = train_trs[used_cols + ['isFraud']]
    if test:
        test_trs = pd.read_csv('data/test_transaction.csv')
        test_ids = pd.read_csv('data/test_identity.csv')
        test_trs = test_trs[used_cols]
    train_df, test_df = None, None
    if val:
        test_trs = train_trs.loc[train_trs.TransactionDT >= 13000000, :]
        train_trs = train_trs.loc[train_trs.TransactionDT < 13000000, :]
        test_trs = test_trs[used_cols]
        test_ids = train_ids
    if train:
        train_df = preprocess_train(train_trs, train_ids, skip)
    if test:
        train_df = train_df if train else pd.read_csv('train_features.csv')
        test_df = preprocess_test(test_trs, test_ids, train_df, skip)
    return train_df, test_df

def preprocess_train(trs, ids, skip=[]):
    print('train')
    task_id = 'date_sin'
    print(task_id)
    to_merge = []
    if not task_id in skip:
        to_merge += [task_id]
        df = get_date_sin_features(
            trs, "TransactionDT", "TransactionAmt", 5000).set_index("TransactionID")
        df.to_csv(f"tmp/output/{task_id}.csv")
    task_id = 'amount'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = get_amount_features(trs)
        df.to_csv(f"tmp/output/{task_id}.csv")
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
    task_id = 'card_type_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = card_type_dummies(trs)
        df.to_csv(f"tmp/output/{task_id}.csv")
    task_id = 'device_type_dummies'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = device_type_dummies(trs, ids)
        df.to_csv(f"tmp/output/{task_id}.csv")
    trs_new = trs[["TransactionID", "isFraud"]].copy().set_index("TransactionID")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/date_sin.csv'), on="TransactionID", how="left")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/amount.csv'), on="TransactionID", how="left")
    #  trs_new = trs_new.merge(pd.read_csv('tmp/output/product_cd_dummies.csv'), on='TransactionID', how='left')
    #  trs_new = trs_new.merge(pd.read_csv('tmp/output/card_type_dummies.csv'), on='TransactionID', how='left')
    #  trs_new = trs_new.merge(pd.read_csv('tmp/output/device_type_dummies.csv'), on='TransactionID', how='left')
    trs_new.to_csv('train_features.csv')
    return trs_new

def preprocess_test(trs, ids, train_df, skip=[]):
    task_id = 'date_sin'
    to_merge = []
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = get_date_sin_features(
            trs, "TransactionDT", "TransactionAmt", 5000).set_index("TransactionID")
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    task_id = 'amount'
    print(task_id)
    if not task_id in skip:
        to_merge += [task_id]
        df = get_amount_features(trs)
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
        df = device_type_dummies(trs, ids)
        df.to_csv(f"tmp/output/test/{task_id}.csv")
    trs_new = trs[["TransactionID"]].copy().set_index("TransactionID")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/date_sin.csv'), on="TransactionID", how="left")
    trs_new = trs_new.merge(pd.read_csv('tmp/output/test/amount.csv'), on="TransactionID", how="left")
    #  trs_new = trs_new.merge(pd.read_csv('tmp/output/test/product_cd_dummies.csv'), on='TransactionID', how='left')
    #  trs_new = trs_new.merge(pd.read_csv('tmp/output/test/card_type_dummies.csv'), on='TransactionID', how='left')
    #  trs_new = trs_new.merge(pd.read_csv('tmp/output/test/device_type_dummies.csv'), on='TransactionID', how='left')
    trs_new.to_csv('train_features.csv')
    return trs_new
