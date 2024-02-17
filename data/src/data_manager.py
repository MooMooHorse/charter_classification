import os
import sys
import pandas as pd
from typing import List, Tuple
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname((cur_dir))))
from paths import data_dir, eval_output

class DataManager():
    def __init__(self):
        pass
    # load functions replaces column-names to lower cases (as in data files)
    def load_metadata(self, meta_data_fname = 'G and CCG.xlsx') -> pd.DataFrame:
        meta_data_path = os.path.join(data_dir, meta_data_fname)
        meta_data = pd.read_excel(meta_data_path, index_col=0)
        meta_data = meta_data.rename(columns = {'CIK': 'cik'})
        return meta_data
    def load_labels(self, label_fname = 'LabeledDataCCG.csv') -> pd.DataFrame:
        label_path = os.path.join(data_dir, label_fname)
        labels = pd.read_csv(label_path, index_col=0)
        labels = labels.rename(columns = {'Charter_ID': 'charter_id'})
        return labels
    
    def merge_tables(self):
        '''
        merge the data, label tables. 

        Notes
        -----
        v0: metadata will not be merged
        '''
        def _merge_data_metadata(key = 'cik', data_fname = 'ChartersPanelCCG.csv', meta_data_fname = 'G and CCG.xlsx'):
            '''
                data left outer join meta_data on key
            '''

            data_path = os.path.join(data_dir, data_fname)
            data = pd.read_csv(data_path, index_col=0)
            meta_data = self.load_metadata(meta_data_fname = meta_data_fname)
            return data, pd.merge(data, meta_data, on = key, how = 'left')
        def _merge_data_labels(data_df, key = 'charter_id', label_fname = 'LabeledDataCCG.csv'):
            '''
                data right outer join labels on key
            '''
            labels = self.load_labels(label_fname = label_fname)
            return pd.merge(data_df, labels, on = key, how = 'right')
        
        data_df, merged_df = _merge_data_metadata()
        return _merge_data_labels(data_df)
    
    def keep_wanted_columns(self, data_df, cols = ['cik', 'charter_id', 'text', 'Directors102b7']):
        '''
            keep only the wanted columns
        '''
        return data_df[cols]
    
    def drop_unlabeled(self, data_df, col = 'Directors102b7'):
        '''
            drop rows with nan in col
        '''
        return data_df.dropna(subset = [col])
    
    def store_processed_data(self, data_df, fname = 'processed_data.csv'):
        '''
            store the processed data
        '''
        data_df.to_csv(os.path.join(data_dir, fname))
        print(f'data stored at {os.path.join(data_dir, fname)}')
    def load_processed_data(self, fname = 'processed_data.csv'):
        '''
            load the processed data
        '''
        return pd.read_csv(os.path.join(data_dir, fname), index_col=0)
def test_1():
    dm = DataManager()
    data_df = dm.merge_tables()
    data_df = dm.drop_unlabeled(data_df)
    data_df = dm.keep_wanted_columns(data_df)
    # print num of rows and columns
    print(data_df.head(10))
    print(data_df.shape)
    dm.store_processed_data(data_df)

if __name__ == '__main__':
    test_1()