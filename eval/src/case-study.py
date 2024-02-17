import os
import sys
import pandas as pd
import json
import traceback

from typing import List, Tuple
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname((cur_dir))))
from paths import data_dir, eval_output, prompt_dir, openAI_dir
from data.src.data_manager import DataManager
from genai.itf import OpenAIITF

def _get_first_k_case_text_w_ground_truth(k: int = 1):
    dm = DataManager()
    df = dm.load_processed_data()
    text = df.head(k)['text'].tolist()
    director = df.head(k)['Directors102b7'].tolist()
    indices = df.head(k)['charter_id'].tolist()
    return text[0:k], director[0:k], indices[0:k]

def _get_l_to_r_case_text_w_ground_truth(l: int, r: int):
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
    df = dm.load_processed_data()
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

def _get_chat_completion(messages:List, itf: None, max_tokens = 1000, model = 'gpt-3.5-turbo-0125') -> str:
    if itf is None:
        itf = OpenAIITF()
        itf.initialize_env()
    try:
        return itf.get_chat_completion_content(messages, max_tokens=1000, model='gpt-3.5-turbo-0125')
    except Exception as e:
        # encapsulate the exception and store it in the log
        error_log = {
            'answer' : 'E',
            'error': str(e),
            'reference': traceback.format_exc()
        }
        return json.dumps(error_log)

def _store_reuslt(material, completion, ground_truth, index = '1'):
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
    completion = _get_chat_completion(messages)
    _store_reuslt(material, completion, ground_truth)
    return completion, ground_truth

def study_first_k_cases(k:int = 2, eval_prompt = 'baseline.prompt', start_row = 0):
    material, ground_truth, indices = _get_l_to_r_case_text_w_ground_truth(start_row, start_row + k - 1)
    itf = OpenAIITF()
    itf.initialize_env()
    for i in range(k):
        messages = _assemble_message_list(material[i], eval_prompt) 
        completion = _get_chat_completion(messages, itf)
        _store_reuslt(material[i], completion, ground_truth[i], index = str(indices[i]))
    return completion, ground_truth

def evaluate_study_20():
    evaluate_dir = os.path.join(eval_output, 'study_20')
    # iterate through the directory and evaluate the results
    num_errors = 0
    hit = 0
    miss = 0
    ave_confidence = 0
    for file in os.listdir(evaluate_dir):
        with open(os.path.join(evaluate_dir, file), 'r') as f:
            data = json.load(f)
            completion = data['completion']['answer']
            ground_truth = data['ground_truth']
            if completion == 'E':
                num_errors += 1
            elif completion == ground_truth:
                hit += 1
            else:
                miss += 1
            if completion != 'E':
                ave_confidence += data['completion']['confidence']
    print(f'Errors: {num_errors}, Hit: {hit}, Miss: {miss}')
    print(f'Average confidence: {ave_confidence/(hit + miss)}')
if __name__ == '__main__':
    # print(study_1_case())
    # study_first_k_cases(start_row=100, k=8)
    evaluate_study_20()
    