import os
import json
import asyncio
import argparse
import urllib.request
import urllib.parse

from openai_lib import OpenAIClient, expand_yaml_template

# Make sure the OpenAI key is available in the environment
secret = os.environ['OPENAI_KEY']


def fun_caller(name, args):
    if name != 'get_publication_info':
        raise Exception(f"I know nothing about: {name}")

    args = json.loads(args)
    res = urllib.request.urlopen(
        'https://docmetadata.transltr.io/publications?' + urllib.parse.urlencode(args)
    ).read().decode('utf-8')
    return json.dumps(res)


async def run_loop(client: OpenAIClient):
    print("*** *** *** Load template *** *** ***")
    q = expand_yaml_template('tests/tool-call.yaml', ('instructions', 'tools'))
    print(q)

    print("*** *** *** Run as loop *** *** ***")
    orig_input = q['input']
    del q['input']
    res = await client.run_as_loop(orig_input, q, fun_caller)
    print(res)
    return res


async def main():
    """Entry point that dispatches to individual client methods based on CLI flags."""
    parser = argparse.ArgumentParser(description="OpenAIClient test harness")

    # Mutually exclusive operations
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lf', action='store_true', help='List files')
    group.add_argument('--gf', metavar='FILE_ID', help='Get file by id')
    group.add_argument('--gfc', metavar='FILE_ID', help='Get file contents by id')
    group.add_argument('--lb', action='store_true', help='List batches')
    group.add_argument('--gm', action='store_true', help='Get model list')
    group.add_argument('--rl', action='store_true', help='Run as loop (tool-calling example)')

    args = parser.parse_args()

    client = OpenAIClient(secret)

    # Dispatch to the correct client method
    if args.lf:
        res = await client.list_files()
    elif args.gf:
        res = await client.get_file(args.gf)
    elif args.gfc:
        res = await client.get_file_content(args.gfc)
    elif args.lb:
        res = await client.list_batches()
    elif args.gm:
        res = await client.get_model_list()
    elif args.rl:
        res = await run_loop(client)
    else:
        # This should be impossible because of mutually exclusive group + required=True
        parser.error("No valid operation specified.")

    # Pretty-print the result; fallback to str() for non-serialisable objects
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
