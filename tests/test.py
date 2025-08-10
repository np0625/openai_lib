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
    input = q['input']
    del q['input']
    res = await client.run_as_loop(input, q, fun_caller)
    print(res)
    return res

async def run_loop_streaming(client: OpenAIClient, simple=False):

    if simple:
        input = """Provide a brief two-paragraph summary of what precision medicine is, suitable
        for a general audience."""
        q = {
            'model': 'o3',
            'reasoning': {
                'effort': 'high',
                'summary': 'detailed'
            }
        }
    else:
        q = expand_yaml_template('tests/tool-call-2.yaml', ('instructions', 'tools'))
        input = q['input']
        del q['input']

    event_type_counts = {}
    event_runs = []  # Track runs of consecutive event types
    last_event_type = None
    current_run_count = 0

    async for event in client.run_as_loop_streaming(input, q, fun_caller):
        #print(event.type)
        print(event)
        tname = type(event).__name__
        event_type_counts[tname] = event_type_counts.get(tname, 0) + 1

        # Track runs of consecutive event types
        if tname != last_event_type:
            # End the previous run (if any)
            if last_event_type is not None:
                event_runs.append((last_event_type, current_run_count))
            # Start a new run
            last_event_type = tname
            current_run_count = 1
        else:
            # Continue the current run
            current_run_count += 1

    # End the final run
    if last_event_type is not None:
        event_runs.append((last_event_type, current_run_count))

    # Print event runs report
    print("\nEvent runs (in order of occurrence):")
    for event_type, count in event_runs:
        print(f"{event_type}: {count}")

    # Print final total counts (ascending by count, then by type name)
    print("\nFinal event type counts (ascending):")
    for tname, count in sorted(event_type_counts.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"{tname}: {count}")

"""    return {
        "streamed_events": sum(event_type_counts.values()),
        "event_type_counts": event_type_counts,
        "event_runs": event_runs,
    }"""

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
    group.add_argument('--rls', action='store_true', help='Run as loop (streaming example)')

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
    elif args.rls:
        res = await run_loop_streaming(client, True)
    else:
        # This should be impossible because of mutually exclusive group + required=True
        parser.error("No valid operation specified.")

    # Pretty-print the result; fallback to str() for non-serialisable objects
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
