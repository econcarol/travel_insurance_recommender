from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

system_prompt = '''
You are a helpful assistant for travelers looking for health and travel insurance.
Only recommend plans when the user provides age, location, and medical condition.
Use the search_insurance tool if detailed plan info is needed.
'''

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])