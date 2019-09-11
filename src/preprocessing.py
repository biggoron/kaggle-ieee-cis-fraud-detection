from constants import \
    FIRST_NAN_GROUP, SECOND_NAN_GROUP, THIRD_NAN_GROUP, \
    FOURTH_NAN_GROUP, FIFTH_NAN_GROUP, SIXTH_NAN_GROUP, \
    SEVENTH_NAN_GROUP, EIGTH_NAN_GROUP, NINTH_NAN_GROUP

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
        df = Preprocessor.process_nan_card2to6(df)
        df = Preprocessor.process_nan_m1to9(df)
        df = Preprocessor.process_nan_d235(df)
        df = Preprocessor.process_nan_addr(df)
        df = Preprocessor.process_nan_email(df)
        df = Preprocessor.process_nan_dist1(df)
        df = Preprocessor.process_nan_id_01_12(df)
        df = Preprocessor.process_nan_device_type(df)
        return df

    @staticmethod
    def process_nan_first_group(df):
        df['first_group_nan'] = 1
        for col in FIRST_NAN_GROUP:
            df['first_group_nan'] = (df['first_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_second_group(df):
        df['second_group_nan'] = 1
        for col in SECOND_NAN_GROUP:
            df['second_group_nan'] = (df['second_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_third_group(df):
        df['third_group_nan'] = 1
        for col in THIRD_NAN_GROUP:
            df['third_group_nan'] = (df['third_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_fourth_group(df):
        df['fourth_group_nan'] = 1
        for col in FOURTH_NAN_GROUP:
            df['fourth_group_nan'] = (df['fourth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_fifth_group(df):
        df['fifth_group_nan'] = 1
        for col in FIFTH_NAN_GROUP:
            df['fifth_group_nan'] = (df['fifth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_sixth_group(df):
        df['sixth_group_nan'] = 1
        for col in SIXTH_NAN_GROUP:
            df['sixth_group_nan'] = (df['sixth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_seventh_group(df):
        df['seventh_group_nan'] = 1
        for col in SEVENTH_NAN_GROUP:
            df['seventh_group_nan'] = (df['seventh_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_eigth_group(df):
        df['eigth_group_nan'] = 1
        for col in EIGTH_NAN_GROUP:
            df['eigth_group_nan'] = (df['eigth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_ninth_group(df):
        df['ninth_group_nan'] = 1
        for col in NINTH_NAN_GROUP:
            df['ninth_group_nan'] = (df['ninth_group_nan'] & df[col].isna())
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    @staticmethod
    def process_nan_card2to6(df):
        for i in range(2, 7):
            df[f'no_card_{i}'] = df[f'card{i}'].isna()
            df[f'card{i}'] = df[f'card{i}'].fillna(df[f'card{i}'].mode()[0])
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
    def process_nan_dist1(df):
        df['no_dist1'] = df['dist1'].isna()
        df['dist1'] = df['dist1'].fillna(df['dist1'].mode()[0])
        return df

    @staticmethod
    def process_nan_email(df):
        df['P_emaildomain'] = df['P_emaildomain'].fillna('missing.missing')
        return df

    @staticmethod
    def process_nan_device_type(df):
        df['DeviceType'] = df['DeviceType'].fillna('missing')
        return df
