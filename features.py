import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression

from constants import FRAUD_FREQ, FRAUD_AMP, FRAUD_MEAN

def get_date_sin_features(transactions, col_time, col_reg, agg_time):
    trs = transactions[["TransactionID", col_time, col_reg]].copy()
    m, M = trs[col_time].min(), trs[col_time].max()
    trs["transaction_agg_index"] = (trs[col_time] - m) // agg_time

    agg = trs.groupby("transaction_agg_index").agg({col_reg: "count"})[col_reg]
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
    phases.phase.iloc[50:] = phases.phase.shift(50)[50:]
    phases.phase.iloc[:50] = phases.phase.iloc[0]

    trs = trs.merge(phases, on=['transaction_agg_index'], how='left')
    trs.loc[:, "phase_sin"] = np.sin(FRAUD_FREQ * trs[col_time] + trs.phase).astype(np.float16)
    trs.loc[:, "phase_cos"] = np.cos(FRAUD_FREQ * trs[col_time] + trs.phase).astype(np.float16)
    trs.drop(
        columns=[col_time, "transaction_agg_index", col_reg, "phase"],
        inplace=True)
    return trs

def get_amount_features(trs):
    df = trs[["TransactionID", "TransactionAmt", "TransactionDT"]].copy().sort_values("TransactionDT")
    df.loc[:, "amt_scale"] = np.round(np.log(df.TransactionAmt) * 10).astype(np.int8)
    df.loc[:, "amt_mean"] = df.TransactionAmt.rolling(window=5000, center=True).mean()
    df.iloc[:2500].amt_mean = df.iloc[2500].loc["amt_mean"] 
    df.iloc[-2500:].amt_mean = df.iloc[-2501].loc["amt_mean"] 
    df.loc[:, "amt_diff"] = (df.TransactionAmt - df.amt_mean) / df.amt_mean
    df.loc[:, "amt_round_1"] = np.where(df.TransactionAmt.mod(1) == 0, 1, 0)
    df.loc[:, "amt_round_10"] = np.where(df.TransactionAmt.mod(10) == 0, 1, 0)
    df.loc[:, "amt_round_100"] = np.where(df.TransactionAmt.mod(100) == 0, 1, 0)
    df.drop(columns=["amt_mean", "TransactionDT", "TransactionAmt"], inplace=True)
    return df.set_index("TransactionID")

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
