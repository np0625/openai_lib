
import tiktoken
import openai
import json

from openai_lib import OpenAIClient

# Read in the API key
filename = '.key'
with open(filename) as f:
    secret = f.read().strip()
# print(secret)

# Define some tokenization helpers
def encode_string(encoder, string: str) -> list[int]:
    return encoder.encode(string)

def encode_from_file(encoder, path: str) -> list[int]:
    with open(path) as f:
        data = f.read().strip()
        return encode_string(encoder, data)

def decode_tokens(encoder, tokens: list[int]) -> str:
    return encoder.decode(tokens)


client = OpenAIClient(secret, {"store": True})
# models = client.get_model_list()
# sorted_model_ids = sorted(m.id for m in models.data)
# print(sorted_model_ids)
# print(client.encoding_for_model_id('gpt-4o'))
# print("upload file")
# retval = client.upload_file("junk/in2.txt") # -- tested and worked
# print(retval) ->
# FileObject(id='file-Ltm8GK7Si7YWDyGL4HmwUh', bytes=753, created_at=1743883437, filename='in2', object='file',
#            purpose='user_data', status='processed', expires_at=None, status_details=None)

list = client.list_files()
print(list)
print(client.responsesAPI_configs)

## TODO: 1. json schema; 2. Logging. 3. Batches. 4. Referencing a file

input_standalone = [
        { "role": "developer", "type": "message", "content": "You have a wry, dry, erudite british sense of humor"},
        { "role": "user", "type": "message", "content": "Tell me a joke"}
    ]

input_file_ref = [
    { "role": "developer", "type": "message", "content": """You have an expert understanding of conversational flows.
        You are able to keep track of back-and-forth, of interjections, repetitions, thinking-out-loud, and all other facets
        of multiple-participant conversations. You are able to glean the essential portions of the overall discussion
        and produce crisp, accurate summaries.
        You are also aware of the NIH/NCATS Translator project and are familiar with its terminology"""},
    { "role": "user", "type": "message", "content": [
        {"type": "input_text", "text": "Summarize the contents of the file referenced in this input"},
        {"type": "input_file", "file_id": "file-B23kj1jKEVceUTx5g8Rb3y"}]
    }]
res = client.submit_responsesAPI_request(input_file_ref)

print(res)

exit()

