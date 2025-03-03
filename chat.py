from ollama import chat

import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

default_prompt = config['default_prompt']
example_question = config['example_question']

def chat(model_name, message=example_question):
    messages = [
    {
        'role': 'system',
        'content': default_prompt,
    },
    {
        'role': 'user',
        'content': message,
    },
    ]

    for part in chat(model_name, messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)
