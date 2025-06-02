import json
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

# load plan data into Document format
with open('../data/plans.json', 'r') as file:
    plans = json.load(file)

docs = []
for index, plan in enumerate(plans):
    plan_text = plan['plan_text'] 
    plan_metadata = {
        'plan_id': plan['plan_id'], 
        'plan_name': plan['plan_name'], 
        'plan_type': plan['plan_type']
    }
    doc = Document(
        page_content=plan_text,
        metadata=plan_metadata,
    )
    docs.append(doc)

# load embedding model
embed_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

# embed and save all Documents to vector store
db = FAISS.from_documents(docs, embed_model)
db.save_local('../vectorstore')

print('Plan data loaded to vector store.')