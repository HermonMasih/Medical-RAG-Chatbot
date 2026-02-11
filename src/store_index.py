''' Pinecone store index implementation. '''

import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from src.helper import load_pdfs, filter_pdf_documents, perform_chunking, get_embeddings

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

extracted_documents = load_pdfs('pdfs')
filtered_documents = filter_pdf_documents(extracted_documents)
text_chunks = perform_chunking(filtered_documents)
embedding_model = get_embeddings()

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

INDEX_NAME = "medical-chatbot-index"

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding_model,
    index_name=INDEX_NAME
)
