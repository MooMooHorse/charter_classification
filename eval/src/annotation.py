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
from eval.src.case_study import _get_l_to_r_case_text_w_ground_truth, _assemble_message_list, _store_reuslt
from genai.itf import OpenAIITF
from utils.token_ops import truncate_string_by_tokens, num_tokens_from_messages
from utils.completion import get_chat_completion


version_evaluate_small_dataset = Literal['baseline', 'few-shots', 'few-shots-CoT', 'self-consistency']
model_evaluate_small_dataset = Literal['gpt-3.5-turbo-0125']

def get_token_limit(model_name: model_evaluate_small_dataset = 'gpt-3.5-turbo-0125'):
    if model_name == 'gpt-3.5-turbo-0125':
        return 16385
    else:
        raise ValueError(f"model {model_name} is not supported")

def get_prompt_name(version: version_evaluate_small_dataset = 'baseline'):
    if version == 'baseline':
        return 'baseline.prompt'
    elif version == 'few-shots':
        return 'few-shots.prompt'
    elif version == 'few-shots-CoT':
        return 'few-shots-CoT.prompt'
    elif version == 'self-consistency':
        raise ValueError(f"version {version} is not supported")
        # return 'self-consistency.prompt'
    else:
        raise ValueError(f"version {version} is not supported")

def iter_get_result(material, eval_prompt, itf, model_name: model_evaluate_small_dataset = 'gpt-3.5-turbo-0125', max_iter = 3):
    '''
    Now because material can be very long, we need to break it into smaller chunks and then get the completion.
    So the logic of return json is important:
        {
            "answer": <ANS>,
            "reference": <REF>,
            "confidence" <CONFIDENCE>
        }
    So we should OR the answer and the reference, and take the maximum of the confidence.
    '''
    def _merge_completion(result, new_completion):
        if result is None:
            return new_completion
        else:
            new_completion = json.loads(new_completion)
            if new_completion['answer'] == 'Y':
                result['answer'] = 'Y'
            result['reference'].append(new_completion['reference'])
            result['confidence'] = max(result['confidence'], new_completion['confidence'])
            return result
        
    limit = get_token_limit(model_name=model_name) - 1000
    num_prompt_tokens = num_tokens_from_messages([{"role": "system", "content": eval_prompt}])
    limit -= num_prompt_tokens
    material_start = 0
    completion = None
    while material_start < len(material) and max_iter > 0:
        messages = _assemble_message_list(material[material_start:], eval_prompt)
        new_completion = get_chat_completion(messages, itf)
        completion = _merge_completion(completion, new_completion)
        material_start = truncate_string_by_tokens(material[material_start:], limit) + material_start
        max_iter -= 1
    return completion

def evaluate_study_small_dataset(version: version_evaluate_small_dataset = 'baseline', start_row: int = 0, k: int = 1, model_name: model_evaluate_small_dataset = 'gpt-3.5-turbo-0125'):
    eval_dir = os.path.join(eval_output, version)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    eval_prompt_name = get_prompt_name(version)
    with open(os.path.join(prompt_dir, eval_prompt_name), 'r') as f:
        eval_prompt = f.read()

    material, ground_truth, indices = _get_l_to_r_case_text_w_ground_truth(start_row, start_row + k - 1, csv_file_name='small_dataset.csv')
    itf = OpenAIITF()
    itf.initialize_env()
    for i in range(k):
        if num_tokens_from_messages([{"role": "user", "content": material[i]}]) < get_token_limit(model_name=model_name):
            continue
        completion = iter_get_result(material[i], eval_prompt, itf, ground_truth[i], indices[i], model=model_name)
        _store_reuslt(material[i], completion, ground_truth[i], index = str(indices[i]))
        print(completion)

        break
    
def test_1():
    evaluate_study_small_dataset()