from ollama import chat

import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

default_prompt = config['default_prompt']
example_question = config['example_question']

async def run_chat(model_name, message=example_question):
    ollama_url = os.getenv("OLLAMA_URL")

    client = ollama.AsyncClient(ollama_url)


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

    for part in await client.chat(model_name, messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)

