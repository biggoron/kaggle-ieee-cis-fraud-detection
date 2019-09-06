import os

import pandas as pd

class BaseDataLoader(object):
    def __init__(self, data_path, skip_convert=True):
        if skip_convert:
            print("Loaded tables' types are not converted")
        self.skip_convert = skip_convert
        self.data_path = data_path
        files = os.listdir(data_path)
        files = [f for f in files if f.split('.')[-1] in ['csv', 'pkl']]
        self.files = self._convert_ext(files)

    def _convert_ext(self, files):
        for f in files:
            name, ext = f.split('.')
            path = os.path.join(self.data_path, f)
            if ext == 'csv':
                df = pd.read_csv(path)
                df = self._convert_type(df, name)
                df.to_pickle(path.split('.')[0] + '.pkl')
                os.remove(path)
            elif ext == 'pkl':
                if not self.skip_convert:
                    df = pd.read_pickle(path)
                    df = self._convert_type(df, name)
                    df.to_pickle(os.path.join(self.data_path, 'converted', f))
        files = [f for f in files if f.split('.')[-1] in ['csv', 'pkl']]
        files = {f.split('.')[0]: pd.read_pickle(os.path.join(self.data_path, f)) for f in files}
        return files

    
    @staticmethod
    def _convert_type(df, name):
        '''To be overridden to change data type'''
        return df

if __name__ == '__main__':
    data = BaseDataLoader('data')
