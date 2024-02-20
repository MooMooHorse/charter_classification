import os
import sys
import pandas as pd
import json
import traceback

from typing import List, Tuple, Literal
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname((cur_dir))))
from paths import data_dir, eval_output, prompt_dir, openAI_dir
from data.src.data_manager import DataManager
from genai.itf import OpenAIITF
from utils.token_ops import truncate_string_by_tokens
from utils.completion import get_chat_completion

def _get_first_k_case_text_w_ground_truth(k: int = 1):
    dm = DataManager()
    df = dm.load_processed_data()
    text = df.head(k)['text'].tolist()
    director = df.head(k)['Directors102b7'].tolist()
    indices = df.head(k)['charter_id'].tolist()
    return text[0:k], director[0:k], indices[0:k]

def _get_l_to_r_case_text_w_ground_truth(l: int, r: int, csv_file_name = 'processed_data.csv'):
    '''
        extract l to r rows from the csv file.
        
        Returns
        -------
        text: List[str]
            a portion of charter text
        director: List[str]
             Y/N to the question "Does the charter exculpate directors from monetary liability 
             for breach of fiduciary duty of care (a "102b7" waiver)?" 
        indices: List[str]
            charter_id unique for each charter
    '''
    dm = DataManager()
    df = dm.load_processed_data(fname = csv_file_name)
    r = min(r, len(df))
    sliced_df = df.iloc[l:r+1]
    text = sliced_df['text'].tolist()
    director = sliced_df['Directors102b7'].tolist()
    indices = sliced_df['charter_id'].tolist()
    return text, director, indices


def _assemble_message_list(material, eval_prompt = 'baseline.prompt'):
    with open(os.path.join(prompt_dir, eval_prompt), 'r') as f:
        prompt = f.read()
    message_list = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": material
        }
    ]
    return message_list



def _store_reuslt(material, completion:str , ground_truth, index = '1'):
    '''
    The result stored should contains:
    1. raw material of charter
    2. completion result in format of 
        {
            "answer": <ANS> (Y/N/E),
            "reference": <REF> (part of text in charter),
            "confidence" <CONFIDENCE>
        }
        2.0: when <ANS> == 'E', the reference is the error message
    3. ground truth (against <ANS>)
    '''

    json_dump = {
        'material': material,
        'completion': json.loads(completion),
        'ground_truth': ground_truth
    }
    with open(os.path.join(eval_output, f'study_{index}.json'), 'w') as f:
        f.write(json.dumps(json_dump))

def study_1_case(eval_prompt = 'baseline.prompt'):
    material, ground_truth, _ = _get_first_k_case_text_w_ground_truth()
    material = material[0]
    ground_truth = ground_truth[0]
    messages = _assemble_message_list(material, eval_prompt)
    completion = get_chat_completion(messages)
    _store_reuslt(material, completion, ground_truth)
    return completion, ground_truth

def study_first_k_cases(k:int = 2, eval_prompt = 'baseline.prompt', start_row = 0):
    material, ground_truth, indices = _get_l_to_r_case_text_w_ground_truth(start_row, start_row + k - 1)
    itf = OpenAIITF()
    itf.initialize_env()
    for i in range(k):
        messages = _assemble_message_list(material[i], eval_prompt) 
        completion = get_chat_completion(messages, itf)
        _store_reuslt(material[i], completion, ground_truth[i], index = str(indices[i]))
    return completion, ground_truth

def _load_embedding(fname = "embeddings.json"):
        with open(os.path.join(eval_output, fname), 'r') as f:
            return np.array(json.load(f))

def _store_embedding(embeddings, fname = "embeddings.json"):
    with open(os.path.join(eval_output, fname), 'w') as f:
        f.write(json.dumps(embeddings.tolist()))

def embeddeing_and_cluster(texts:List[str], ids:List[str], num_clusters = 5, itf = None):
    import numpy as np
    from sklearn.cluster import KMeans
    import openai
    from scipy.spatial.distance import cdist
    
    
    if itf is None:
        itf = OpenAIITF()
        itf.initialize_env()
    
    errors = []
    try:
        # Generate embeddings for each text
        embeddings = itf.get_embeddings(texts)
    except Exception as e:
        # encapsulate the exception and store it in the log
        error_log = {
            'answer' : 'E',
            'error': str(e),
            'reference': traceback.format_exc()
        }
        errors.append(json.dumps(error_log))
    
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(embeddings)
    
    _store_embedding(embeddings)
    
    closest_texts_indices = np.argmin(cdist(embeddings, kmeans.cluster_centers_, 'euclidean'), axis=0)
    representative_texts = [texts[index] for index in closest_texts_indices]

    selected_ids = [ids[index] for index in closest_texts_indices]

    return representative_texts, selected_ids, errors

def evaluate_study(study_name = 'study_20'):
    evaluate_dir = os.path.join(eval_output, study_name)
    # iterate through the directory and evaluate the results
    num_errors = 0
    num_Y = 0
    hit = 0
    miss = 0
    miss_list = []
    ave_confidence = 0
    for file in os.listdir(evaluate_dir):
        with open(os.path.join(evaluate_dir, file), 'r') as f:
            data = json.load(f)
            completion = data['completion']['answer']
            ground_truth = data['ground_truth']
            if ground_truth == 'Y':
                num_Y += 1
            if completion == 'E':
                num_errors += 1
            elif completion == ground_truth:
                hit += 1
            else:
                # extract case id study_{id}.json
                miss_list.append(file.split('.')[0].split('_')[1])
                miss += 1
            if completion != 'E':
                ave_confidence += data['completion']['confidence']
    print(f'Errors: {num_errors}, Hit: {hit}, Miss: {miss}')
    print(f'Average confidence: {ave_confidence/(hit + miss)}')
    print(f'Y: {num_Y}, N: {hit + miss - num_Y}')
    print(f'Miss list: {miss_list}')


def apply_embedding_to_study_20():
    """
    Note
    ----
    v1.0 directly truncate the text to 8191 tokens, without splitting the text into multiple parts
    """
    evaluate_dir = os.path.join(eval_output, 'study_20')
    # enumberate the texts and extract their ids from filename study_{id}.json
    texts = []
    ids = []
    itf = OpenAIITF()
    itf.initialize_env()
    for file in os.listdir(evaluate_dir):
        with open(os.path.join(evaluate_dir, file), 'r') as f:
            data = json.load(f)
            completion = data['completion']['answer']
            ground_truth = data['ground_truth']
            if completion == ground_truth or completion == 'E':
                continue
            texts.append(truncate_string_by_tokens(data['material'], limit = 8000))
            ids.append(file.split('.')[0].split('_')[1])

    texts, ids, errors = embeddeing_and_cluster(texts, ids, num_clusters=5, itf = itf)
    print(ids)
    print(f''' there are {len(errors)} errors in the process of embedding and clustering''')
    print(f'''\033[1;31;40m{errors}\033[0m''')

def apply_difference_evaluation(study_dir = 'study_20', 
                                study_ids = ['1002037B20070619', '1002910N20170628', '1004434C20170614', '100726B20161026', '1002910E20170628'],
                                prompt_name = 'difference-evaluation.prompt'):
    itf = OpenAIITF()
    itf.initialize_env()
    for study_id in study_ids:
        with open(os.path.join(eval_output, study_dir, f'study_{study_id}.json'), 'r') as f:
            data = json.load(f)
            # data['material'] = truncate_string_by_tokens(data['material'], limit = 16000)
            messages = _assemble_message_list(str(data), eval_prompt = prompt_name) 
            completion = get_chat_completion(messages, itf, max_tokens = 2000)
            output_path = os.path.join(eval_output, f'{study_id}_diff.json')
            with open(output_path, 'w') as f:
                f.write(completion)


if __name__ == '__main__':
    # print(study_1_case())
    # study_first_k_cases(start_row=100, k=8)
    # evaluate_study_20()
    # apply_difference_evaluation()
    evaluate_study(study_name='baseline')
    # apply_difference_evaluation(study_dir='few-shots-CoT', study_ids=['104918C20120606', '730000E19831014', '900075D20161216', '1002637D20070510'])