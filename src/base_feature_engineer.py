import pickle

import pandas as pd

class BaseFeatureEngineer(object):
    def __init__(self, list_transformers, skip=[], exclude=[]):
        list_transformers = [(name, method) for name, method in list_transformers if '_feat' in name]
        list_transformers = [
            (int(name.split('_')[2]), '_'.join(name.split('_')[3:]), method)
            for name, method in list_transformers]
        self.list_transformers = sorted(list_transformers)
        self.exclude = set(exclude)
        self.skip = set(skip).union(self.exclude)
        self.state={}

    def _load_state(self, filename='/home/dan/Projects/kaggle-ieee-cis-fraud-detection/tmp/features_state.pkl'):
        filehandler = open(filename,"r")
        self.state = pickle.load(filehandler)

    def _dump_state(self, filename='/home/dan/Projects/kaggle-ieee-cis-fraud-detection/tmp/features_state.pkl'):
        filehandler = open(filename,"wb")
        pickle.dump(self.state, filehandler)

    def transform(self, data, index_col, phase):
        for step, name, method in self.list_transformers:
            print(f"step {step}: {name}")
            if name in self.skip:
                print('skipping')
                continue
            transformed_data = method(data, phase)
            transformed_data.to_pickle(f"/home/dan/Projects/kaggle-ieee-cis-fraud-detection/tmp/{name}_{phase}.pkl")
        data = data[[index_col]].set_index(index_col)
        for _, name, _ in self.list_transformers:
            if name in self.exclude:
                continue
            df = pd.read_pickle(f"/home/dan/Projects/kaggle-ieee-cis-fraud-detection/tmp/{name}_{phase}.pkl")
            data = data.merge(df, on=index_col, how='left')
        if phase == 'train':
            state = {'columns': data.columns}
            self.state = {**self.state, **state}
            self._dump_state()
        if phase in ['val', 'test']:
            data = self.align_columns(data)
        return data

    def align_columns(self, data):
        ref_cols = self.state['columns']
        for col in ref_cols:
            if col not in data:
                print(f"missing {col}")
                data[self.impute_missing[col]] = 1
                data[col] = 0
        for col in data.columns:
            if col not in ref_cols:
                print(f"dropping {col}")
                data.drop(columns=[col], inplace=True)
        data = data[ref_cols]
        return data
