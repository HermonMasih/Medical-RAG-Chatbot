'''Helper functions for loading PDFs, filtering documents, performing chunking, and creating embeddings.'''

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdfs(directory_path):
    '''Load all PDF files from the specified directory'''
    
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_pdf_documents(documents: List[Document]) -> List[Document]:
    '''Filter out documents that are not PDFs'''
    
    source_content: List[Document] = []
    for document in documents:
        if document.metadata['page'] !=0:
            source_content.append(
                Document(
                    metadata = {'page_no': document.metadata['page']},
                    page_content=document.page_content
                )
            )
    return source_content


def perform_chunking(source_content):
    '''Perform chunking on the source content'''
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    chunks = text_splitter.split_documents(source_content)
    return chunks


def get_embeddings():
    '''Create embeddings for the chunks'''
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

