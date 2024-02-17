import os
cur_dir = os.path.abspath(os.path.dirname(__file__))
eval_dir = os.path.join(cur_dir, 'eval') # paths containing evaluation scripts
eval_src = os.path.join(eval_dir, 'src') # paths containing evaluation source code
eval_output = os.path.join(eval_dir, 'output') # paths containing evaluation output
plot_dir = os.path.join(cur_dir, 'plot') # paths containing plots & scripts
plot_src = os.path.join(plot_dir, 'src') # paths containing plot source code
plot_output = os.path.join(plot_dir, 'output') # paths containing plot output
doc_dir = os.path.join(cur_dir, 'docs') # paths containing documentation
data_dir = os.path.join(cur_dir, 'data') # paths containing data
data_src = os.path.join(data_dir, 'src') # paths containing data source code
openAI_dir = os.path.join(cur_dir, 'genai') # paths containing openai scripts
prompt_dir = os.path.join(openAI_dir, 'prompts') # paths containing openai prompt
openai_config_path = os.path.join(openAI_dir, 'config.json') # paths containing openai config
openai_log_dir = os.path.join(openAI_dir, 'logs') # paths containing openai logs