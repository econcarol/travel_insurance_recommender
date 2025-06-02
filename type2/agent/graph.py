from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from .agent import build_agent

# define state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], 'chat_history']

# define node
def agent_node(state: AgentState):
    agent = build_agent()
    result = agent.invoke({
        'chat_history': state['messages'],
        # assume last message is user input
        'input': state['messages'][-1].content  
    })
    # append agent reply to chat history
    state['messages'].append(AIMessage(content=result['output']))
    return state

# define graph
graph = StateGraph(AgentState)
graph.add_node('agent', agent_node)
graph.set_entry_point('agent')
graph.add_edge('agent', END)

# compile graph
app = graph.compile()