import os
import json
import asyncio
import tiktoken
import sys
from openai_lib import OpenAIClient

# Tokenisation helpers (kept synchronous)

def encode_string(encoder, string: str) -> list[int]:
    return encoder.encode(string)

def encode_from_file(encoder, path: str) -> list[int]:
    with open(path) as f:
        data = f.read().strip()
        return encode_string(encoder, data)

def decode_tokens(encoder, tokens: list[int]) -> str:
    return encoder.decode(tokens)


async def main():
    secret = os.environ['OPENAI_KEY']
    client = OpenAIClient(secret, {"store": True})
    print(client._client)
    files = await client.list_files()
    print(files)
    print(client.responsesAPI_configs)
    input_file_ref = [
        {
            "role": "developer",
            "type": "message",
            "content": (
                "You have an expert understanding of conversational flows. "
                "You are able to keep track of back-and-forth, of interjections, repetitions, thinking-out-loud, "
                "and all other facets of multiple-participant conversations. You are able to glean the essential "
                "portions of the overall discussion and produce crisp, accurate summaries. "
                "You are also aware of the NIH/NCATS Translator project and are familiar with its terminology"
            ),
        },
        {
            "role": "user",
            "type": "message",
            "content": [
                {"type": "input_text", "text": "Summarize the contents of the file referenced in this input"},
                {"type": "input_file", "file_id": "file-B23kj1jKEVceUTx5g8Rb3y"},
            ],
        },
    ]

    res = await client.submit_responsesAPI_request(input_file_ref)
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
