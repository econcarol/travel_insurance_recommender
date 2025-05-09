# A Simple Recommender
This is an MVP using RAG for real-time recommendation. 

Goal 
- Retrieve plan details from a database and use an LLM to compare/match them to the traveler’s profile

Approach
- Store structured insurance plan data in a vector database
- Use the traveler’s profile as a query to fetch top relevant plans
- Let LLM generate a natural-language recommendation

Tech Stack
- Vector database: FAISS
- Embedding model: all-mpnet-base-v2
- LLM: gpt-4-turbo (with tool calling)
- No framework such as LangChain or LangGraph
- UI: Gradio