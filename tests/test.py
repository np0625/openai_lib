from openai_lib import OpenAIClient, expand_yaml_template
import os
import json
import urllib.request
import urllib.parse

secret = os.environ['OPENAI_KEY']

client = OpenAIClient(secret)

"""
# Test run_as_batch
print("*** *** *** Run as batch *** *** ***")
test_batch_query = {
    'model': 'gpt-4o',
    'instructions': 'You have a wry, dry, erudite British sense of humor',
    'input': 'Tell me a funny story with a sharp punchline'
}
res = client.run_as_batch(test_batch_query, 'batch-test', {'purpose': 'testing'})
print(res)
"""

def fun_caller(name, args):
    if name != 'get_publication_info':
        raise Exception(f"I know nothing about: {name}")

    args = json.loads(args)
    res = urllib.request.urlopen('https://docmetadata.transltr.io/publications?' + urllib.parse.urlencode(args)).read().decode('utf-8')
    return json.dumps(res)


# Test run_as_loop along with the YAML template reading
print("*** *** *** Load template *** *** ***")
q = expand_yaml_template('tests/tool-call.yaml', ('instructions','tools'))
print(q)

print("*** *** *** Run as loop *** *** ***")
orig_input = q['input']
del(q['input'])
res = client.run_as_loop(orig_input, q, fun_caller)
print(res)
