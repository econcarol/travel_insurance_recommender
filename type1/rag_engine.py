import os, faiss, json, pickle, openai
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# load openai api key
load_dotenv()
openai.api_key =os.getenv('OPENAI_API_KEY')

# load saved vectordb
index = faiss.read_index('vectorstore\plans.index')
with open('vectorstore\plan_metadata.pkl', 'rb') as f:
    plan_metadata = pickle.load(f)
with open('vectorstore\plan_texts.pkl', 'rb') as f:
    plan_texts = pickle.load(f)

# load embedding model
embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# specify LLM model
llm_model = 'gpt-4-turbo'

# a fn to retrieve top 2 plans given user profile
top_k = 2
def search_plan(user_profile: str) -> str:
    # embed user profile
    query_embedding = embed_model.encode([user_profile])[0]
    D, I = index.search(np.array([query_embedding]), k=top_k)
    
    # retrieve top plans
    top_plans = [(plan_metadata[i], plan_texts[i]) for i in I[0]]
    result = "\n\n".join([f"{name}: {desc}" for name, desc in top_plans])
    return result

# define the above fn as a tool
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'search_plan',
            'description': 'Searche for best matching travel insurance plans based on user profile',
            'parameters': {
                'type': 'object',
                'properties': {
                    'user_profile': {'type': 'string'}
                },
                "required": ['user_profile']
            }
        }
    }
]

# let AI decide whether and how to use the tools
def recommender(user_input: str) -> str:
    system_prompt = '''
    Please recommend the best plan from all retrieved plans based on traveler's profile. Also, give a concise reasoning.
    '''

    messages = [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # send user profile to LLM with tools
    # and get initial response
    completion = openai.chat.completions.create(
        model = llm_model,
        messages = messages,
        tools = tools,
        tool_choice = 'auto'
    )   
    response = completion.choices[0].message

    # process initial response
    tool_calls = response.tool_calls
    # call tools if requested by LLM
    if tool_calls:
        # add each LLM's tool call request to messages
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            tool_call_request = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": fn_name,
                            "arguments": str(fn_args)
                        }
                    }
                ]       
            }
            messages.append(tool_call_request)

        # for each tool call request, call the tool 
        # and add response to messages
        available_fns = {
            'search_plan': search_plan
        }
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_to_call = available_fns[fn_name]
            fn_args = json.loads(tool_call.function.arguments)
            fn_result = fn_to_call(**fn_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": "Here are the retrieved plans:"+fn_result
                }
            )
        
        completion = openai.chat.completions.create(
            model = llm_model,
            messages = messages,
            tools = tools,
            tool_choice = 'auto'
        )
        response = completion.choices[0].message
        print(messages)
        return response.content
    # if LLM didn't request any tool call
    # then it means user input is out of scope
    else:
        messages.append(
            {
                "role": "assistant",
                "content": "Please enter relevant travel profile info."
            }
        )
        print(messages)
        return 'Please enter relevant travel profile info.'