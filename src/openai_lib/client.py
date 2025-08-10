import openai
import tiktoken
import math
import io
import json
from collections.abc import Callable


class OpenAIClient:

    BATCH_ENDPOINT = '/v1/responses'

    def __init__(self, key: str, config: dict = {}):
        self._client = openai.AsyncOpenAI(api_key=key)
        default_responsesAPI_configs = {
            "model": "gpt-4o",
            "max_output_tokens": 8192,
            "store": False,
            "user": "my-api-client",
        }
        self.responsesAPI_configs = {**default_responsesAPI_configs, **config}

    # Automatic fall-through
    def __getattr__(self, name):
        return getattr(self._client, name)

    async def upload_file(self, file: str, purpose: str = 'batch'):
        with open(file, 'rb') as fh:
            return await self._client.files.create(file=fh, purpose=purpose)

    async def list_files(self):
        return await self._client.files.list()

    async def get_file(self, id: str):
        return await self._client.files.retrieve(id)

    async def get_file_content(self, id: str):
        return await self._client.files.content(id)

    async def list_batches(self, mode: str = 'all'):
        batches = await self._client.batches.list()
        retval = {}
        for b in batches.data:
            if mode == 'all' or b.status != 'completed':
                retval.setdefault(b.status, []).append(b.to_dict())
        return retval

    async def get_batch(self, id: str):
        return await self._client.batches.retrieve(id)

    async def get_model_list(self):
        return await self._client.models.list()

    def encoding_for_model_id(self, model_id: str):
        return tiktoken.encoding_for_model(model_id)

    async def submit_responsesAPI_request(self, input: list, config: dict = {}):
        payload = {**self.responsesAPI_configs, **config, "input": input}
        return await self._client.responses.create(**payload)

    async def run_as_batch(
        self,
        requests: dict | list[dict],
        custom_id_prefix: str,
        metadata: dict = {},
    ):
        if isinstance(requests, dict):
            requests = [requests]

        pad = int(math.log10(len(requests))) + 1
        batch_input_data = [{
                'body': r,
                'method': 'POST',
                'url': self.BATCH_ENDPOINT,
                'custom_id': f"{custom_id_prefix}-{str(i + 1).zfill(pad)}",
            } for i, r in enumerate(requests)
        ]

        batch_input_bytes = io.BytesIO("\n".join([json.dumps(b) for b in batch_input_data]).encode())
        # This is not a real File object, but the API accepts it. The filename gets recorded as 'upload'
        batch_file = await self._client.files.create(file=batch_input_bytes, purpose="batch")
        batch = await self._client.batches.create(
            input_file_id=batch_file.id,
            endpoint=self.BATCH_ENDPOINT,
            completion_window='24h',
            metadata=metadata,
        )
        return await self._client.batches.retrieve(batch.id)

    # `Params` is expected to contain a `tools` entry for this loop to be meaningful.
    async def run_as_loop(
        self,
        orig_input: str | dict,
        params: dict,
        funcaller: Callable,
        max_turns: int = 10,
    ):
        if isinstance(orig_input, dict):
            orig_input = [orig_input]
        input_data = orig_input
        prev_resp_id = None
        turns = 0

        while turns < max_turns:
            turns += 1
            resp = await self._client.responses.create(**params, input=input_data, previous_response_id=prev_resp_id)
            prev_resp_id = resp.id
            input_data = []

            for elem in resp.output:
                if elem.type == 'message':
                    return resp
                elif elem.type == 'function_call':
                    fun_call_res = await funcaller(elem.name, elem.arguments)
                    input_data.append({
                        'type': 'function_call_output',
                        'call_id': elem.call_id,
                        'output': fun_call_res
                    })
                else:
                    pass

        raise Exception(f"Tool calling loop exceeded max turns: {max_turns}")


    async def run_as_loop_streaming(
        self,
        orig_input: str | dict | list,
        params: dict,
        funcaller: Callable,
        turn: int = 1,
        previous_response_id = None,
        max_turns: int = 10,
        text_chunk: int = 10,
    ):
        if isinstance(orig_input, dict):
            orig_input = [orig_input]
        input_data = orig_input
        params['stream'] = True
        collections = {}
        function_outputs = []
        stream = await self._client.responses.create(**params, input=input_data, previous_response_id=previous_response_id)
        async for event in stream:
            # print(event)
            etype = event.type
            if etype == 'response.created':
                # Only the initial few events will contain the response id; grab it for prev_resp_id if needed
                previous_response_id = event.response.id
                # print(previous_response_id)
            elif etype == 'response.output_item.added':
                # Note event.item.id in here, but event.item_id in other clauses
                # In this clause, we check for and initialize textual outputs, for now restricted to reasoning
                # summaries and the final message
                if event.item.type in ('message', 'reasoning'):
                    collections[event.item.id] = {
                        'type': event.item.type,
                        'output_text': '',
                        'n_chunks': 0,
                        'done': False
                    }
                else:
                    pass
            elif (etype in ('response.content_part.added', 'response.output_text.delta',
                            'response.reasoning_summary_text.delta', 'response.reasoning_summary_part.added')
                and (event.item_id in collections)):
                # Here we collect the initial and delta values for the textual outputs of interest, yielding
                # back chunks when we collect the specified amount.

                # need this 'lazy' evaluation of the fallback
                collections[event.item_id]['output_text'] += event.part.text if hasattr(event, "part") else getattr(event, "delta")
                collections[event.item_id]['n_chunks'] += 1
                # If we've collected #chunk pieces of output, yield them
                if collections[event.item_id]['n_chunks'] % text_chunk == 0:
                    yield collections[event.item_id]
                else:
                    pass
            elif etype in ('response.output_text.done', 'response.reasoning_summary_text.done'):
                # Yield the completed textual output
                collections[event.item_id]['done'] = True
                yield collections[event.item_id]
            elif etype == 'response.output_item.done' and event.item.type == 'function_call':
                # Here we collect the output of any function calls that the model requests.
                # Note that with parallel_tool_calls=True by default, a single response may
                # request multiple function calls
                fun_call_res = await funcaller(event.item.name, event.item.arguments)
                function_outputs.append({
                    'type': 'function_call_output',
                    'call_id': event.item.call_id,
                    'output': fun_call_res
                })
            else:
                pass

        if len(function_outputs) > 0:
            if turn >= max_turns:
                raise Exception(f"Tool calling loop exceeded max turns: {max_turns}")
            else:
                yield({'type': 'function_call_outputs', 'outputs': function_outputs})
                # If we're here, continue streaming by restarting the generation with prev_response_id + function
                # call outputs as the input.
                async for event in self.run_as_loop_streaming(function_outputs, params, funcaller, turn + 1,
                    previous_response_id, max_turns, text_chunk):
                    yield event

"""
Responses request:

{
    model^: "gpt-4o",  // required

    // <less imp. params>
        // this is additional output data from other sources e.g. image urls, tool calls, file search results. Not immediately imp.
        include: [...],

        // Inserts a system (or developer) message as the first item in the model's context. [Why needed if using input.role??]
        instructions: "...",

        max_output_tokens: int // how to turn off??

        metadata | parallel_tool_calls | previous_response_id | tool_choice | tools : not immediately relevant

        reasoning: { effort: "high|medium|low", generate_summary: "concise|detailed"} // for o-series models only

        stream: boolean // defaults to false
    // </less imp. params>

    store: boolean // defaults to true

    temperature: 0..2 // default 1, higher = more random
    top_p: number, // defaults to 1, todo: how does this work??

    // completely unable to understand how to use this?? It seems to imply that one of two things will happen if model OUTPUT
    // exceeds INPUT context buffer: either the request fails w/ 400 or "the model will truncate the response to fit the
    // context window by dropping input items in the middle of the conversation". Both seem shitty??
    truncation: "disabled(default)|auto"

    user: "..." // any string to associate a request with a "user"

    // This parameter is terribly named; it should be called "output"
    text: {
      format: {type^: "text (default)"}, // but json_schema is strongly rec. for new models, so exploring that below:
      ||
      format: {type^: "json_schema", name^: "a name for the response format. arbitrary??", description: "optional", strict: bool, def. false,
        schema^: // see elsewhere}
    }

    input^: "tell me a joke", // implies role = user
    -- OR --
    input: [
      Input item list can contains items of three types:
      1. Input message;
      2. Item (a sort of composite type);
      3. Item reference

      Giving examplles of each below. Note that you can specify multiple inputs of differing types as long as they all sort of "make sense" together

      1. Input message:
      {
        role: "developer|user"
        type: "message^",
        content: "just a string containing the content"
          || an array containing N inputs to the model which are of three types:
             1. text, expressed by an object like so: {type: "input_text^", text: "the actual text" },
             2. image input: gonna skip this
             3. File input: { type: "input_file^" , and then one of: file_data (contents), file_id, or filename. }
                Presumably 'filename' must mean a file in the OpenAI store previously uploaded
      } // this type is weird because it seems to recapitulate all the things possible w/ the "Item" type?

      2. Item, an object that is (one of?):
         2.1 Input message
           { type: "message^", // but optional
             status: "in_progress|completed|incomplete", //possibly only useful in chains -- do not understand??
             role: "user|system|developer" (I believe "system" is now deprecated?)
             content: see prior section for description of the content ARRAY field: note that here, it MUST be an array
              and cannot be either an array or a string
           }
           Note: this portion of the Item object is almost exactly like the prior input type, the "Input Message". The diffs are:
             a) content cannot be a string; b) there is a "status" property

           As best as I can tell, this type is mainly for chain-like interactions.
           For now, we will just use the plain "input message" type, as I don't see examples of this more complex type and it's
           also super hard to tell if the elements below are REQUIRED or not...

         2.2 Output message
             This is super confusing, because why is OUTPUT an input into a request? This appears to be about statekeeping and having
             chain-like interactions. Skipping for now
         2.3 Results from tools calls (tc's): File search tc, computer tc, computer tool tc, web search tc, function tc, function tool tc
             Essentially, if you are using tool calls, then you can reference and provide the results of those calls with a lot of granularity
             (using ref ids and other controlling params) as input into a Responses API request
         2.4 "A description of the chain of thought used by a reasoning model while generating a response." -- look into when getting into using
             reasoning (but how and why is this an INPUT?? Maybe for stateful interactions??)

      3. Item Reference. Mercifully simple:
      { type^: "item reference^", id^: "<item_id, e.g. file-Ltm8GK7Si7YWDyGL4HmwUh>"}




    ]
}
"""
