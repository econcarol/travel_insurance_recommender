import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# load saved vectordb
index = faiss.read_index('vectorstore\plans.index')
with open('vectorstore\plan_metadata.pkl', 'rb') as f:
    plan_metadata = pickle.load(f)
with open('vectorstore\plan_texts.pkl', 'rb') as f:
    plan_texts = pickle.load(f)

# load embedding model
embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# user query
print('''Here is an example query:
I'm a 63-year-old traveling to Thailand for 3 weeks. I have hypertension.
I need a travel insurance plan with good medical coverage and COVID protection.
''')
query = input("Enter your query:")

# embed query and search vectordb
query_embedding = embed_model.encode([query])[0]
D, I = index.search(np.array([query_embedding]), k=2)

# retrieve and print result
top_plans = [(plan_metadata[i], plan_texts[i]) for i in I[0]]
result = "\n\n".join([f"{name}: {desc}" for name, desc in top_plans])

print('Here are the top 2 retrieved results:\n\n', result)