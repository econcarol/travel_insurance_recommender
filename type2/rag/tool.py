import json
from langchain.tools import Tool
from .retriever import get_retriever

def search_insurance(query: str) -> str:
    retriever = get_retriever()
    results = retriever.get_relevant_documents(query)
    return '\\n'.join([f'Plan Metadata: {json.dumps(doc.metadata)}\nPlan Description: {doc.page_content}' for doc in results])

insurance_tool = Tool(
    name='search_insurance',
    func=search_insurance,
    description='Searche for insurance plans based on user needs'
)