
import tiktoken
import openai
import json
import os
import sys
from openai_lib import OpenAIClient

# Read in the API key
secret = os.environ['OPENAI_KEY']
# print(secret)


client = OpenAIClient(secret)
models = client.get_model_list()
sorted_model_ids = sorted(m.id for m in models.data)
print(sorted_model_ids)

print(client.encoding_for_model_id('gpt-4o'))
print("upload file")
# retval = client.upload_file("junk/in2") # -- tested and worked
# print(retval) ->
# FileObject(id='file-Ltm8GK7Si7YWDyGL4HmwUh', bytes=753, created_at=1743883437, filename='in2', object='file',
#            purpose='user_data', status='processed', expires_at=None, status_details=None)

lis = client.list_files()
print(lis)
print(f"{type(lis.data)}")


# Create a map of encoding->model
# Note that all modern (text) models use 'o200k_base'
enc_map = {}
for model_id in sorted_model_ids:
    try:
        enc = tiktoken.encoding_for_model(model_id)
        # print(f"{model_id}: {enc}")
    except Exception as e:
        enc = "ERROR"
        # print(f"**** {model_id}: error attempting to retrieve encoding: (add _e_ for deets)")
    enc_map.setdefault(enc, []).append(model_id)

# print("= = = = = ")
for enc, model_ids in enc_map.items():
    print(f"= {enc}: {sorted(model_ids)}")

# Define some tokenization helpers
def encode_string(encoder, string: str) -> list[int]:
    return encoder.encode(string)

def encode_from_file(encoder, path: str) -> list[int]:
    with open(path) as f:
        data = f.read().strip()
        return encode_string(encoder, data)

def decode_tokens(encoder, tokens: list[int]) -> str:
    return encoder.decode(tokens)

# Do some tokenization tests
encoder = client.encoding_for_model_id('gpt-4o')
tokens = encode_string(encoder, "what is this, I don't even...?")
print(tokens)
print(decode_tokens(encoder, tokens))
print(f"Just one: {encoder.decode_single_token_bytes(186964)}")

# Note: file with 21543 words (wc -w) gives back 39648 tokens
tokens = encode_from_file(encoder, "junk/in.orig")
print(f"token count: {len(tokens)}")
# And here's why: `Chris Bizon` -> [b'Chris', b' Biz', b'on']; `Tyler Beck` -> [b'Ty', b'ler', b' Beck'];
# Punctuation is also an independent token, etc. So wc -w will undercount by quite a lot.
## print(encoder.decode_tokens_bytes(tokens))

print (sys.path)
