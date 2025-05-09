import faiss, json, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# load embedding model
embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# load plan data from JSON
with open('data\plans.json', 'r') as file:
    plans = json.load(file)

plan_metadata = [
    {
        'plan_id': item['plan_id'], 
        'plan_name': item['plan_name'], 
        'plan_type': item['plan_type']
    } for item in plans
]
plan_texts = [item['plan_text'] for item in plans]

# embed plan texts
embeddings = embed_model.encode(plan_texts)

# create faiss index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# save index and metadata
faiss.write_index(index, 'vectorstore\plans.index')
with open('vectorstore\plan_metadata.pkl', 'wb') as f:
    pickle.dump(plan_metadata, f)
with open('vectorstore\plan_texts.pkl', 'wb') as f:
    pickle.dump(plan_texts, f)

print('FAISS index and metadata saved.')