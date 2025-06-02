# A Conversational Recommender
This is a conversational agent that decides when to RAG (retrieve insurance plans) when making a recommendation. 

Goal 
- Use an agent to recommend a plan and offer reasoning via a chat with a traveler

Approach
- Store structured insurance plan data in a vector database
- Let LLM decide when and how to retrieve those plan details
- Chat with the user and recommend best plan based on the chat

Tech Stack
- Vector database: FAISS
- Embedding model: all-mpnet-base-v2
- LLM: gpt-3.5-turbo
- Agent framework: LangGraph
- UI: Gradio