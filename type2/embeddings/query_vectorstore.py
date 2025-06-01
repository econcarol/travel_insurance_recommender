from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# load embedding model
embed_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

# load saved vector store
db = FAISS.load_local(
    '../vectorstore', 
    embed_model,
    allow_dangerous_deserialization=True
)

# user query
print('''Here is an example query:
I'm a 63-year-old traveling to Thailand for 3 weeks. I have hypertension.
I need a travel insurance plan with good medical coverage and COVID protection.
''')
query = input('Enter your query:')

# embed query and search vectordb
result = db.similarity_search(query, k=2)

# retrieve and print result
print('Here are the top 2 retrieved results:\n\n')

for i, doc in enumerate(result):
    print(f'Result {i+1}:')
    print('Plan Name:', doc.metadata['plan_name'])
    print('Plan Text:', doc.page_content)
    print('-----')