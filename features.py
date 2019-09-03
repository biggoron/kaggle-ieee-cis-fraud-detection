import numpy as np
import pandas as pd
from scipy.optimize import leastsq

from constants import FRAUD_FREQ, FRAUD_AMP, FRAUD_MEAN

def train_date_sin_reg(transactions, col_time, col_reg, agg_time):
    trs = transactions[["TransactionID", col_time, col_reg]]
    m, M = trs[col_time].min(), trs[col_time].max()
    trs["transaction_agg_index"] = (trs[col_time] - m) // agg_time

    agg = trs.groupby("transaction_agg_index").agg({col_reg: "sum"})[col_reg]
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
    trs["phase_sin"] = np.sin(FRAUD_FREQ * trs.TransactionDT + trs.phase)
    trs["phase_cos"] = np.cos(FRAUD_FREQ * trs.TransactionDT + trs.phase)
    trs.drop(
        columns=["TransactionDT", "transaction_agg_index", "isFraud", "phase"],
        inplace=True)
    return trs
