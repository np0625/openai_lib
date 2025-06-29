
from openai_lib import OpenAIClient, expand_yaml_template


import os

secret = os.environ['OPENAI_KEY']

client = OpenAIClient(secret)

test_batch_query = {
    'model': 'gpt-4o',
    'instructions': 'You have a wry, dry, erudite British sense of humor',
    'input': 'Tell me a funny story with a sharp punchline'
}

res = client.run_as_batch(test_batch_query, 'batch-test', {'purpose': 'testing'})
print(res)


