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
import tqdm


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
        return 'few-shots-CoT.prompt'
        # return 'self-consistency.prompt'
    else:
        raise ValueError(f"version {version} is not supported")

def iter_get_result(material, eval_prompt_name, itf, model_name: model_evaluate_small_dataset = 'gpt-3.5-turbo-0125', max_iter = 3, temperature = 0.7) -> dict | None:
    '''
    Now because material can be very long, we need to break it into smaller chunks and then get the completion.
    So we should OR the answer and the reference, and take the maximum of the confidence.
    Return
    ------
    result: dict
        {
            "answer": <ANS>,
            "reference": <REF>,
            "confidence" <CONFIDENCE>
        }
    '''
    def _merge_completion(result: dict | None, new_completion: str) -> dict:
        if result is None:
            # print(f'''\033[93m{new_completion}\033[0m''')
            return json.loads(new_completion)
        else:
            new_completion = json.loads(new_completion)
            # print(new_completion)
            if new_completion['answer'] == 'Y':
                result['answer'] = 'Y'
            result['reference'] = result['reference'] + new_completion['reference']
            result['confidence'] = max(result['confidence'], new_completion['confidence'])
            return result
    with open(os.path.join(prompt_dir, eval_prompt_name), 'r') as f:
        eval_prompt = f.read()
    limit = get_token_limit(model_name=model_name) - 1000
    num_prompt_tokens = num_tokens_from_messages([{"role": "system", "content": eval_prompt}])
    limit -= num_prompt_tokens
    material_end = 0
    material_start = 0
    completion = None
    while material_end < len(material) and max_iter > 0:
        material_end = len(truncate_string_by_tokens(material[material_end:], limit)) + material_end # this is the next start (IMPORTANT)
        messages = _assemble_message_list(material[material_start:material_end], eval_prompt_name)
        new_completion = get_chat_completion(messages, itf, temperature=temperature)
        completion = _merge_completion(completion, new_completion)
        material_start = material_end
        max_iter -= 1
    return completion

def evaluate_study_small_dataset(version: version_evaluate_small_dataset = 'baseline', 
                                 start_row: int = 0, k: int = 1, max_iter = 3,
                                 temperature = 0.7,model_name: model_evaluate_small_dataset = 'gpt-3.5-turbo-0125', self_consistency_rounds = 3):
    eval_dir = os.path.join(eval_output, version)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    eval_prompt_name = get_prompt_name(version)
    
    material, ground_truth, indices = _get_l_to_r_case_text_w_ground_truth(start_row, start_row + k - 1, csv_file_name='small_dataset.csv')
    itf = OpenAIITF()
    itf.initialize_env()
    for i in tqdm.tqdm(range(k), desc="Evaluating"):
        if version == 'self-consistency':
            completions_candidate = []
            for j in range(self_consistency_rounds):
                completion = iter_get_result(material[i], eval_prompt_name, itf, model_name=model_name, temperature=temperature, max_iter = max_iter)
                completions_candidate.append(completion)
            # print(completions_candidate)
            answers = [c['answer'] for c in completions_candidate]
            half_majority = self_consistency_rounds // 2
            ans = 'Y' if answers.count('Y') > half_majority else 'N'
            reference = [ref for c in completions_candidate for ref in c['reference'] if c['answer'] == ans]
            completion = {
                'answer': 'Y' if answers.count('Y') > half_majority else 'N',
                'reference': reference,
                'confidence': max([c['confidence'] for c in completions_candidate])
            }
        else:
            completion = iter_get_result(material[i], eval_prompt_name, itf, model_name=model_name, temperature=temperature)
        _store_reuslt(material[i], json.dumps(completion), ground_truth[i], index = str(indices[i]))
    
def test_1():
    evaluate_study_small_dataset(version='self-consistency', start_row=1, k=99, temperature=0.8, max_iter=2)

if __name__ == '__main__':
    test_1()