import numpy as np
import pandas as pd

def card_type_dummies(trs):
    card_type_comp_dummies = pd.get_dummies(
        trs[['TransactionID', 'card4']].fillna('other'),
        columns=['card4'],
        prefix='card_comp').set_index("TransactionID")
    card_type_nat_dummies = pd.get_dummies(
        trs[['TransactionID', 'card6']].fillna('other'),
        columns=['card6'],
        prefix='card_type').set_index("TransactionID")
    card_type_dummies = card_type_nat_dummies.merge(card_type_comp_dummies, on='TransactionID', how='left')
    return card_type_dummies

def device_type_dummies(trs, ids):
    device_type_dummies = pd.get_dummies(
        ids[['TransactionID', 'DeviceType']].fillna('other'),
        columns=['DeviceType'],
        prefix='device_type').set_index("TransactionID")
    device_type_dummies.loc[:, 'device_not_recognized'] = 0
    total_ids = trs[['TransactionID']]
    total_ids = total_ids.merge(device_type_dummies, on='TransactionID', how='left')
    total_ids.loc[:, "device_not_recognized"] = total_ids.device_not_recognized.fillna(1)
    total_ids = total_ids.fillna(0).set_index("TransactionID")
    return total_ids
