import os
import sys
import pandas as pd
from typing import List, Tuple
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname((cur_dir))))
from paths import data_dir, eval_output
from genai.itf import OpenAIITF
from data.src.data_manager import DataManager
class CorpusEvaluator():
    def __init__(self):
        pass
    def measure_length(self, csv_file_name = 'ChartersPanelCCG.csv'):
        '''
            extract 'text' field of the csv file and return the total length and average length (per row)
            Returns
            -------
            total_length: int
                accumulated length of all the text
            num_rows: int
                number of rows in the csv file
            avg_length: float
                average length of the text
        '''
        # data_path = os.path.join(data_dir, csv_file_name)
        # df = pd.read_csv(data_path)
        dm = DataManager()
        itf = OpenAIITF()
        df = dm.drop_unlabeled(dm.load_processed_data(), col = 'text')
        print(df.shape)
        text = df['text']
        distr = []
        for t in text:
            message = [{
                'role' : 'system',
                'content' : t
            }]
            num_tokens = itf.num_tokens_from_messages(message)
            distr.append(num_tokens)
        # plot the distribution of the length of the text
        # generate 2 plots, one for all distribution, the other for distribution of 0 - 128 k length
        import matplotlib.pyplot as plt

        plt.hist(distr, bins = 1000)
        plt.title('# tokens distribution')
        plt.xlabel('# tokens used per charter')
        plt.ylabel('Frequency')

        plt.savefig(os.path.join(eval_output, 'text_length_distribution.png'))
        plt.clf()
        plt.hist(distr, bins = 1000, range = (0, 128000))
        plt.title('# tokens distribution (0 - 128k)')
        plt.xlabel('# tokens used per charter')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(eval_output, 'text_length_distribution_0_128k.png'))
        total_length = sum(distr)
        return total_length, len(text), total_length/len(text)
    
    def measure_Y_N_ratio(self, col_name = ["Directors102b7"], csv_file_name = 'ChartersPanelCCG.csv'):
        '''
            extract 'Directors102b7' field of the csv file and return the ratio of Y and N
            Returns
            -------
            ratio: float
                ratio of Y and N
        '''
        # data_path = os.path.join(data_dir, csv_file_name)
        # df = pd.read_csv(data_path)
        dm = DataManager()
        df = dm.load_processed_data()
        col = df[col_name]
        return col.value_counts(normalize = True)
    
    def dump_k_row(self, k = 1, csv_file_name = 'ChartersPanelCCG.csv') -> Tuple[List[pd.Series], List[str]]:
        '''
            extract k rows from the csv file
            Returns
            -------
            rows: List[pandas.Series]
                k rows from the csv file
            text: List[str]
                text field of the row
        '''
        dm = DataManager()
        df = dm.load_processed_data()
        return df.head(k), df['text'].head(k).tolist()
    
def test_1():
    ce = CorpusEvaluator()
    # print(ce.measure_length())
    rows_sampled, texts = ce.dump_k_row(k = 20)
    charter_ids = rows_sampled['charter_id']
    Directors102b7 = rows_sampled['Directors102b7']
    for charter_id, text, directors in zip(charter_ids, texts, Directors102b7):
        with open(os.path.join(eval_output, f'{charter_id}_{directors}.txt'), 'w') as f:
            f.write(text)
    print(rows_sampled)
def test_2():
    ce = CorpusEvaluator()
    print(ce.measure_Y_N_ratio())

def test_3():
    ce = CorpusEvaluator()
    print(ce.measure_length())

if __name__ == '__main__':
    test_3()