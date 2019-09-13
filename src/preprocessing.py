import pandas as pd
import numpy as np
from constants import \
    FIRST_NAN_GROUP, SECOND_NAN_GROUP, THIRD_NAN_GROUP, \
    FOURTH_NAN_GROUP, FIFTH_NAN_GROUP, SIXTH_NAN_GROUP, \
    SEVENTH_NAN_GROUP, EIGTH_NAN_GROUP, NINTH_NAN_GROUP, \
    ELEVENTH_NAN_GROUP, TWELVTH_NAN_GROUP, \
    THIRTEENTH_NAN_GROUP , FOURTEENTH_NAN_GROUP, FIFTEENTH_NAN_GROUP , \
    EMAILS, US_EMAILS

from sklearn.preprocessing import OneHotEncoder

class Preprocessor():

    @staticmethod
    def remove_nan(df):
        df = Preprocessor.process_nan_first_group(df)
        df = Preprocessor.process_nan_second_group(df)
        df = Preprocessor.process_nan_third_group(df)
        df = Preprocessor.process_nan_fourth_group(df)
        df = Preprocessor.process_nan_fifth_group(df)
        df = Preprocessor.process_nan_sixth_group(df)
        df = Preprocessor.process_nan_seventh_group(df)
        df = Preprocessor.process_nan_eigth_group(df)
        df = Preprocessor.process_nan_ninth_group(df)
        df = Preprocessor.process_nan_eleventh_group(df)
        df = Preprocessor.process_nan_twelvth_group(df)
        df = Preprocessor.process_nan_thirteenth_group(df)
        df = Preprocessor.process_nan_fourteenth_group(df)
        df = Preprocessor.process_nan_fifteenth_group(df)
        df = Preprocessor.process_nan_card2to6(df)
        df = Preprocessor.process_nan_m1to9(df)
        df = Preprocessor.process_nan_d235(df)
        df = Preprocessor.process_nan_addr(df)
        df = Preprocessor.process_nan_email(df)
        df = Preprocessor.process_nan_dist(df)
        df = Preprocessor.process_nan_neg_num_id(df)
        df = Preprocessor.process_nan_pos_num_id(df)
        df = Preprocessor.process_nan_cat_id(df)
        df = Preprocessor.process_nan_id_05_06(df)
        df = Preprocessor.process_nan_device(df)
        bool_data = df.select_dtypes(include='bool')
        for col in bool_data.columns:
            df[col] = df[col].astype(np.int8)
        del bool_data
        return df

    def fit_transform_cat(self, df):
        self.cats = {}
        cat_data = df.select_dtypes(include='object')
        self.cats['cat_cols'] = cat_data.columns
        del cat_data

        # Low number of categories
        cat_few_values_cols = [col for col in self.cats['cat_cols'] if df[col].nunique() <= 7]
        self.cats['few_values'] = cat_few_values_cols

        # Big number of categories
        cat_many_values_cols = [col for col in self.cats['cat_cols'] if df[col].nunique() > 7]
        self.cats['many_values'] = cat_many_values_cols
        for col in cat_many_values_cols:
            print(col)
            if col in ['P_emaildomain_suffix', 'R_emaildomain_suffix', 'P_emaildomain_bin', 'R_emaildomain_bin']:
                continue
            subtable = df[['isFraud', col]].copy()
            mean = subtable.groupby(col).agg({'isFraud': 'mean'}).rename(columns={'isFraud': 'fraud_mean'})
            count = subtable.groupby(col).agg({'isFraud': 'count'}).rename(columns={'isFraud': 'sample_nb'})
            subtable = mean.merge(count, on=col, how='left').sort_values('fraud_mean')
            del mean
            del count
            subtable['category'] = pd.cut(subtable.sample_nb.cumsum(), 10, labels=False)
            m = subtable.category.min()
            M = subtable.category.max()
            cat_dict = subtable['category'].replace({
                **{v: 'usual' for v in list(
                    set(subtable.category.unique()).difference(set([m, M])))},
                m: 'low_fraud',
                M: 'high_fraud'}).to_dict()
            self.cats[col] = cat_dict
            self.cats[f'{col}_default'] = df[col].mode()[0]
            df[col] = df[col].replace(cat_dict)

        df = pd.get_dummies(df, columns=self.cats['cat_cols'], prefix=self.cats['cat_cols'])
        return df

    def transform_cat(self, df):
        for col in self.cats['many_values']:
            print(col)
            if col in ['P_emaildomain_suffix', 'R_emaildomain_suffix', 'P_emaildomain_bin', 'R_emaildomain_bin']:
                continue
            cat_dict = self.cats[col]
            missing_values = set(df[col].unique()).difference(set(list(cat_dict.keys())))
            default = self.cats[f'{col}_default']
            cat_dict = {**cat_dict, **{name: default for name in missing_values}}
            df[col] = df[col].replace(cat_dict)
        df = pd.get_dummies(df, columns=self.cats['cat_cols'], prefix=self.cats['cat_cols'])
        return df

    def transform_amount(self, df):
        df.loc[:, "amt_round_1"] = np.where(df.TransactionAmt.mod(1) == 0, 1, 0).astype(np.int8)
        df.loc[:, "amt_round_10"] = np.where(df.TransactionAmt.mod(10) == 0, 1, 0).astype(np.int8)
        df.loc[:, "amt_round_100"] = np.where(df.TransactionAmt.mod(100) == 0, 1, 0).astype(np.int8)
        return df

    def add_new_features(self, data): 
        data['uid'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)
        data['uid2'] = data['uid'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)
        data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)
        
        return data

    def transform(self, train, test):
        train = self.transform_amount(train)
        test = self.transform_amount(test)

        train = self.add_new_features(train)
        test = self.add_new_features(test)

        i_cols = ['card1','card2','card3','card5','uid','uid2','uid3']

        for col in i_cols:
            for agg_type in ['mean','std']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_df = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])
                #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
                temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})
                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()
                train[new_col_name] = train[col].map(temp_df)
                test[new_col_name]  = test[col].map(temp_df)

        train = train.replace(np.inf,999)
        test = test.replace(np.inf,999)

        train['TransactionAmt'] = np.log1p(train['TransactionAmt'])
        test['TransactionAmt'] = np.log1p(test['TransactionAmt'])

        for c in ['P_emaildomain', 'R_emaildomain']:
            train[c + '_bin'] = train[c].map(EMAILS)
            test[c + '_bin'] = test[c].map(EMAILS)

            train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
            test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

            train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in US_EMAILS else 'us')
            test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in US_EMAILS else 'us')

        p = 'P_emaildomain'
        r = 'R_emaildomain'
        uknown = 'email_not_provided'

        def setDomain(df):
            df[p] = df[p].fillna(uknown)
            df[r] = df[r].fillna(uknown)
            
            # Check if P_emaildomain matches R_emaildomain
            df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)

            df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])
            df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])
            
            return df
            
        train=setDomain(train)
        test=setDomain(test)

        return train, test

    @staticmethod
    def process_nan_first_group(df):
        df['first_group_nan'] = 1
        for col in FIRST_NAN_GROUP:
            df['first_group_nan'] = (df['first_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_second_group(df):
        df['second_group_nan'] = 1
        for col in SECOND_NAN_GROUP:
            df['second_group_nan'] = (df['second_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_third_group(df):
        df['third_group_nan'] = 1
        for col in THIRD_NAN_GROUP:
            df['third_group_nan'] = (df['third_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_fourth_group(df):
        df['fourth_group_nan'] = 1
        for col in FOURTH_NAN_GROUP:
            df['fourth_group_nan'] = (df['fourth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_fifth_group(df):
        df['fifth_group_nan'] = 1
        for col in FIFTH_NAN_GROUP:
            df['fifth_group_nan'] = (df['fifth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_sixth_group(df):
        df['sixth_group_nan'] = 1
        for col in SIXTH_NAN_GROUP:
            df['sixth_group_nan'] = (df['sixth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_seventh_group(df):
        df['seventh_group_nan'] = 1
        for col in SEVENTH_NAN_GROUP:
            df['seventh_group_nan'] = (df['seventh_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_eigth_group(df):
        df['eigth_group_nan'] = 1
        for col in EIGTH_NAN_GROUP:
            df['eigth_group_nan'] = (df['eigth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_ninth_group(df):
        df['ninth_group_nan'] = 1
        for col in NINTH_NAN_GROUP:
            df['ninth_group_nan'] = (df['ninth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_eleventh_group(df):
        df['eleventh_group_nan'] = 1
        for col in ELEVENTH_NAN_GROUP:
            df['eleventh_group_nan'] = (df['eleventh_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_twelvth_group(df):
        df['twelvth_group_nan'] = 1
        for col in TWELVTH_NAN_GROUP:
            df['twelvth_group_nan'] = (df['twelvth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_thirteenth_group(df):
        df['thirteenth_group_nan'] = 1
        for col in THIRTEENTH_NAN_GROUP:
            df['thirteenth_group_nan'] = (df['thirteenth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_fourteenth_group(df):
        df['fourteenth_group_nan'] = 1
        for col in FIFTEENTH_NAN_GROUP :
            df['fourteenth_group_nan'] = (df['fourteenth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_fifteenth_group(df):
        df['fifteenth_group_nan'] = 1
        for col in FOURTEENTH_NAN_GROUP :
            df['fifteenth_group_nan'] = (df['fifteenth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_card2to6(df):
        for i in range(2, 7):
            if i in [4, 6]:
                df[f'card_{i}'] = df[f'card{i}'].fillna('missing')
            else:
                df[f'no_card_{i}'] = df[f'card{i}'].isna()
                df[f'card{i}'] = df[f'card{i}'].fillna(-1)
        return df

    @staticmethod
    def process_nan_m1to9(df):
        for i in range(1, 10):
            df[f'M{i}'] = df[f'M{i}'].fillna('missing')
        return df

    @staticmethod
    def process_nan_addr(df):
        df['no_addr'] = 1
        for col in ['addr1', 'addr2']:
            df['no_addr'] = (df['no_addr'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_id_01_12(df):
        df['no_id_01_12'] = 1
        for col in ['id_01', 'id_12']:
            df['no_id_01_12'] = (df['no_id_01_12'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_d235(df):
        for col in ['D2', 'D3', 'D5']:
            df[f'no_{col}'] = df[col].isna()
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_dist(df):
        df['no_dist1'] = df['dist1'].isna()
        df['dist1'] = df['dist1'].fillna(-1)
        df['no_dist2'] = df['dist2'].isna()
        df['dist2'] = df['dist2'].fillna(-1)
        return df

    @staticmethod
    def process_nan_email(df):
        df['P_emaildomain'] = df['P_emaildomain'].fillna('missing')
        df['R_emaildomain'] = df['R_emaildomain'].fillna('missing')
        return df

    @staticmethod
    def process_nan_device(df):
        df['DeviceType'] = df['DeviceType'].fillna('missing')
        df['DeviceInfo'] = df['DeviceInfo'].fillna('missing')
        return df

    @staticmethod
    def process_nan_cat_id(df):
        for col in ['id_12', 'id_16', 'id_23', 'id_37', 'id_36', 'id_35', 'id_38', 'id_15', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_34', 'id_33']:
            df[col] = df[col].fillna('missing')
        return df

    @staticmethod
    def process_nan_id_05_06(df):
        df['missing_id_05_06'] = df.id_05.isna() & df.id_06.isna()
        df['id_05'] = df['id_05'].fillna(df['id_05'].mode()[0])
        df['id_06'] = df['id_06'].fillna(df['id_06'].mode()[0])
        df['missing_id_03_04'] = df.id_03.isna() & df.id_04.isna()
        df['id_03'] = df['id_03'].fillna(df['id_03'].mode()[0])
        df['id_04'] = df['id_04'].fillna(df['id_04'].mode()[0])
        df['missing_id_09_10'] = df.id_09.isna() & df.id_10.isna()
        df['id_09'] = df['id_09'].fillna(df['id_09'].mode()[0])
        df['id_10'] = df['id_10'].fillna(df['id_10'].mode()[0])
        df['missing_id_07_08'] = df.id_07.isna() & df.id_08.isna()
        df['id_07'] = df['id_07'].fillna(df['id_07'].mode()[0])
        df['id_08'] = df['id_08'].fillna(df['id_08'].mode()[0])
        return df

    @staticmethod
    def process_nan_pos_num_id(df):
        for col in [
            'id_02', 'id_11', 'id_13', 'id_17', 'id_18', 'id_19', 'id_20', 'id_32',
            'id_21', 'id_22', 'id_24', 'id_25', 'id_26', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
            df[f'missing_{col}'] = df[col].isna()
            df[col] = df[col].fillna(-1)
        return df

    @staticmethod
    def process_nan_neg_num_id(df):
        for col in ['id_01', 'id_14']:
            df[f'missing_{col}'] = df[col].isna()
            df[col] = df[col].fillna(df[col].mode()[0])
        return df
