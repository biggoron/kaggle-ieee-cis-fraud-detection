import inspect
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
from constants import FRAUD_FREQ, FRAUD_AMP, FRAUD_MEAN

from base_feature_engineer import BaseFeatureEngineer

class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, skip=[], exclude=[]):
        list_transformers = [(name, method) for name, method in inspect.getmembers(FeatureEngineer, predicate=inspect.isfunction)]
        super().__init__(list_transformers, skip, exclude)
        self.impute_missing = {
            "card_type_charge card": "card_type_other",
            "card_type_debit or credit": "card_type_other",
            }
        
    @staticmethod
    def _feat_0_trs_amt_sin(trs, phase):
        '''Computes the sin and cos arg in Transaction count oscillations.'''
        # TODO: replace all literals
        agg_time = 5000
        df = trs[['TransactionID', 'TransactionDT', 'TransactionAmt']].copy()
        m, M = df['TransactionDT'].min(), df['TransactionDT'].max()
        df["transaction_agg_index"] = (df['TransactionDT'] - m) // agg_time

        agg = df.groupby("transaction_agg_index").agg({'TransactionAmt': "count"})['TransactionAmt']
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

        df = df.merge(phases, on=['transaction_agg_index'], how='left')
        df.loc[:, "phase_sin"] = np.sin(FRAUD_FREQ * df['TransactionDT'] + df.phase).astype(np.float16)
        df.loc[:, "phase_cos"] = np.cos(FRAUD_FREQ * df['TransactionDT'] + df.phase).astype(np.float16)
        df.drop(
            columns=['TransactionDT', "transaction_agg_index", 'TransactionAmt', "phase"],
            inplace=True)
        return df.set_index("TransactionID")

    @staticmethod
    def _feat_1_trs_amt_scale(trs, phase):
        df = trs[["TransactionID", "TransactionAmt"]].copy()
        df.loc[:, "amt_scale"] = np.round(np.log(df.TransactionAmt) * 10).astype(np.int8)
        df.drop(columns=["TransactionAmt"], inplace=True)
        return df.set_index("TransactionID")
    
    @staticmethod
    def _feat_2_trs_amt_diff(trs, phase):
        df = trs[["TransactionID", "TransactionAmt", "TransactionDT"]].copy().sort_values("TransactionDT")
        df.loc[:, "amt_mean"] = df.TransactionAmt.rolling(window=5000, center=True).mean()
        df.iloc[:2500].amt_mean = df.iloc[2500].loc["amt_mean"] 
        df.iloc[-2500:].amt_mean = df.iloc[-2501].loc["amt_mean"] 
        df.loc[:, "amt_diff"] = (df.TransactionAmt - df.amt_mean) / df.amt_mean
        df.drop(columns=["amt_mean", "TransactionDT", "TransactionAmt"], inplace=True)
        return df.set_index("TransactionID")

    @staticmethod
    def _feat_3_trs_amt_round(trs, phase):
        df = trs[["TransactionID", "TransactionAmt"]].copy()
        df.loc[:, "amt_round_1"] = np.where(df.TransactionAmt.mod(1) == 0, 1, 0)
        df.loc[:, "amt_round_10"] = np.where(df.TransactionAmt.mod(10) == 0, 1, 0)
        df.loc[:, "amt_round_100"] = np.where(df.TransactionAmt.mod(100) == 0, 1, 0)
        df.drop(columns=["TransactionAmt"], inplace=True)
        return df.set_index("TransactionID")

    @staticmethod
    def _feat_4_trs_card_type_dum(trs, phase):
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


    #  @staticmethod
    #  def device_type_dummies(ids, phase):
        #  device_type_dummies = pd.get_dummies(
            #  ids[['TransactionID', 'DeviceType']].fillna('other'),
            #  columns=['DeviceType'],
            #  prefix='device_type').set_index("TransactionID")
        #  device_type_dummies.loc[:, 'device_not_recognized'] = 0
        #  total_ids = trs[['TransactionID']]
        #  total_ids = total_ids.merge(device_type_dummies, on='TransactionID', how='left')
        #  total_ids.loc[:, "device_not_recognized"] = total_ids.device_not_recognized.fillna(1)
        #  total_ids = total_ids.fillna(0).set_index("TransactionID")
        #  return total_ids
