import pickle

class Packager(object):
    val_split_col = 'TransactionDT'
    index_col = 'TransactionID'
    label_col = 'isFraud'

    @classmethod
    def get_train_val_masks(cls, table, limit=13000000):
        df = table[[cls.index_col, cls.val_split_col]]
        val_mask = df[cls.val_split_col].ge(limit)
        train_mask = df[cls.val_split_col].lt(limit)
        return train_mask, val_mask

    @classmethod
    def split_feat_label(cls, data):
        tot_cols = set(list(data.columns))
        feat_cols = list(tot_cols.difference(set(cls.label_col)))
        return feat_cols, cls.label_col
