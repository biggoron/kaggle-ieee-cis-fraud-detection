import numpy as np
import pandas as pd
from scipy.optimize import leastsq

from constants import FRAUD_FREQ, FRAUD_AMP, FRAUD_MEAN

def train_date_sin_reg(transactions, col_time, col_reg, agg_time):
    trs = transactions[["TransactionID", col_time, col_reg]].copy()
    m, M = trs[col_time].min(), trs[col_time].max()
    trs["transaction_agg_index"] = (trs[col_time] - m) // agg_time

    agg = trs.groupby("transaction_agg_index").agg({col_reg: "mean"})[col_reg]
    time = np.arange(m, M, agg_time)
        
    def opti_func_gen(time_span, values):
        optimize_func = \
            lambda x: FRAUD_AMP * np.sin(FRAUD_FREQ * time_span + x[0]) \
                      + FRAUD_MEAN - values
        return optimize_func

    first_guess_phase = 0
    cache = {}
    for i, j in enumerate(time):
        optimize_func = opti_func_gen(time[i : i+ 50], agg[i : i+50])
        est_phase = leastsq(
            optimize_func,
            [first_guess_phase])[0]
        cache[j] = est_phase

    phases_dict = sorted(cache.items())
    phases = pd.DataFrame({
        "transaction_agg_index": ([item[0] for item in phases_dict] - m) // agg_time,
        "phase": [item[1][0] for item in phases_dict],
    })
    print('here')
    phases.phase.iloc[50:] = phases.phase.shift(50)[50:]
    phases.phase.iloc[:50] = phases.phase.iloc[0]
    print('here')

    trs = trs.merge(phases, on=['transaction_agg_index'], how='left')
    trs["phase_sin"] = np.sin(FRAUD_FREQ * trs[col_time] + trs.phase).astype(np.float16)
    trs["phase_cos"] = np.cos(FRAUD_FREQ * trs[col_time] + trs.phase).astype(np.float16)
    trs.drop(
        columns=[col_time, "transaction_agg_index", col_reg, "phase"],
        inplace=True)
    return trs

def card1_features(trs):
    agg_per_fraud = (
        trs[['TransactionID', 'card1', 'isFraud']].groupby(['card1', 'isFraud'])
                                                  .agg('count')
                                                  .reset_index())
    no_fraud = (agg_per_fraud[agg_per_fraud.isFraud == 0].rename(columns={"TransactionID": "nb_no_fraud"})
                                                         .drop(columns=['isFraud']))
    fraud = (agg_per_fraud[agg_per_fraud.isFraud == 1].rename(columns={"TransactionID": "nb_fraud"})
                                                      .drop(columns=['isFraud']))
    agg_per_fraud = no_fraud.merge(fraud, on='card1', how='outer').fillna(0)
    agg_per_fraud['card1_total_use'] = (agg_per_fraud.nb_fraud + agg_per_fraud.nb_no_fraud).astype(np.int32)
    agg_per_fraud['card1_percent_fraud'] = (agg_per_fraud.nb_fraud / agg_per_fraud.card1_total_use).astype(np.float16)
    agg_per_fraud.loc[agg_per_fraud.card1_total_use == 1, 'card1_percent_fraud'] = 0.
    agg_per_fraud.drop(columns=['nb_no_fraud', 'nb_fraud'], inplace=True)
    card1_features = (trs[['TransactionID', 'card1']].merge(agg_per_fraud, on='card1', how='left')
                                                     .set_index('TransactionID'))
    return card1_features

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

def device_type_dummies(ids):
    device_type_dummies = pd.get_dummies(
        ids[['TransactionID', 'DeviceType']].fillna('other'),
        columns=['DeviceType'],
        prefix='device_type').set_index("TransactionID")
    return device_type_dummies
