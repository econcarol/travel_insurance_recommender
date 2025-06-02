from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def get_retriever():

    # load embedding model
    embed_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

    # load saved vector store
    db = FAISS.load_local(
        'vectorstore', 
        embed_model,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever()