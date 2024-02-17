from typing import Dict, List
import os
import sys
cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname((cur_dir)))
from paths import openai_config_path
from genai.logger import OpenAI_Logger

class OpenAIITF():
    def __init__(self):
        self.logger = OpenAI_Logger()
        

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """
            Returns the number of tokens used by a list of messages.
            Source:
                https://platform.openai.com/docs/guides/text-generation/managing-tokens
        """
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    def get_chat_completion(self, messages, model="gpt-3.5-turbo", max_tokens=150, temperature=0.7, 
                            top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, 
                            stop=None):
        """
            Returns the completion of a chat given a list of messages.
            Source:
                https://platform.openai.com/docs/guides/text-generation/managing-tokens
        """
        from openai import OpenAI
        client = OpenAI()

        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
    
    def get_chat_completion_content(self, messages, model="gpt-3.5-turbo", max_tokens=150, temperature=0.7,
                                    top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
        """
            Returns the content of the completion of a chat given a list of messages.

            Side-effect
            -----------
            get_chat_completion() might throw an exception, it MUST be handled by the caller.

            This allows customized error handling to help the experimentations.
        """
        completion =  self.get_chat_completion(messages, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop)
        content = completion.choices[0].message.content
        self.logger.log_message(str(completion))
        self.logger.update_statistics(completion.usage.total_tokens, 
                                      completion.usage.completion_tokens, 
                                      completion.usage.prompt_tokens,
                                      is_called=1)
        return content
    
    def initialize_env(self):
        """
            Initializes the OpenAI environment.
            
            Side-effect
            -----------
            Sets the environment variable OPENAI_API_KEY.
        """
        import openai, json
        with open(openai_config_path, "r") as f:
            config = json.load(f)
        openai.api_key = config["api_key"]
        # set the environment variable
        os.environ["OPENAI_API_KEY"] = config["api_key"]

    
def test_1():
    itf = OpenAIITF()
    itf.initialize_env()
    messages = [
        {"role": "system", "content" : "Hello, how are you doing today?"}
    ]
    print(itf.get_chat_completion_content(messages))

if __name__ == '__main__':
    test_1()

