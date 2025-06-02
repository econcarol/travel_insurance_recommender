import os, openai
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from rag.prompt import prompt
from rag.tool import insurance_tool

# load openai api key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# create agent
def build_agent():
    # load LLM
    llm = ChatOpenAI(
        model='gpt-3.5-turbo',  
        temperature=0.0
    )

    # create agent
    agent = create_openai_functions_agent(
        llm=llm,
        tools=[insurance_tool],
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent, 
        tools=[insurance_tool], 
        verbose=True
    )