import openai
import tiktoken
import math
import io
import json

class OpenAIClient:

    BATCH_ENDPOINT = '/v1/responses'

    def __init__(self, key:str, config:dict = {}):
        self._client = openai.OpenAI(api_key=key)
        default_responsesAPI_configs = {
            "model": "gpt-4o",
            "max_output_tokens": 8192,
            "store": False,
            "user": "my-api-client"
        }
        self.responsesAPI_configs = {**default_responsesAPI_configs, **config}

    # Automatic fall-through
    def __getattr__(self, name):
        return getattr(self._client, name)

    def upload_file(self, file, purpose='batch'):
        return self._client.files.create(
            file = open(file, 'rb'),
            purpose = purpose
        )

    def list_files(self):
        return self._client.files.list()

    def list_batches(self, mode='all'):
        batches = self._client.batches.list()
        retval = {}
        for b in batches.data:
            if mode == 'all' or b.status != 'completed':
                if b.status not in retval:
                    retval[b.status] = []
                retval[b.status].append(b.to_dict())
        return retval

    def get_file(self, id):
        return self._client.files.retrieve(id)

    def get_batch(self, id):
        return self._client.batches.retrieve(id)

    def get_file_content(self, id):
        return self._client.files.content(id)

    def get_model_list(self) -> dict[str, str]:
        return self._client.models.list()

    def encoding_for_model_id(self, model_id: str) -> str:
        return tiktoken.encoding_for_model(model_id)

    def submit_responsesAPI_request(self, input: list, config: dict={}) -> dict:
        # print(input)
        payload = {**self.responsesAPI_configs, **config, "input": input}
        return self._client.responses.create(**payload)


    def run_as_batch(self, requests: dict | list[dict], custom_id_prefix: str, metadata: dict = {}) -> dict:
        if isinstance(requests, dict):
            requests = [requests]
        pad = int(math.log10(len(requests))) + 1
        batch_input_data = [{
            'body': r,
            'method': 'POST',
            'url': self.BATCH_ENDPOINT,
            'custom_id': f"{custom_id_prefix}-{str(i+1).zfill(pad)}"
            } for i, r in enumerate(requests)]
        batch_input_data = io.BytesIO("\n".join([json.dumps(b) for b in batch_input_data]).encode())
        # This is not a real File object, but the API accepts it. The filename gets recorded as 'upload'
        batch_file = self._client.files.create(file=batch_input_data, purpose="batch")
        batch = self._client.batches.create(
            input_file_id = batch_file.id,
            endpoint = self.BATCH_ENDPOINT,
            completion_window = '24h',
            metadata = metadata
        )
        return self._client.batches.retrieve(batch.id)



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

