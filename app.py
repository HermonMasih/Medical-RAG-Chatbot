from flask import Flask, render_template, request,jsonify
from src.helper import get_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['PINECONE_INDEX_NAME'] = os.getenv('PINECONE_INDEX_NAME')

embedding = get_embeddings()
index_name = os.environ['PINECONE_INDEX_NAME']
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type = 'similarity', search_kwargs={"k": 3})

chat = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model="openai/gpt-oss-20b:groq"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answering_chain = create_stuff_documents_chain(chat,prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

@app.route('/')
def index():
    '''Render the chat interface.'''
    return render_template('chat.html')


@app.route('/get', methods=['GET', 'POST'])
def chat():
    '''Handle user input and generate a response.'''
    msg = request.form['msg']
    input = msg
    print(f"User input: {input}")
    response = rag_chain.invoke({'input': msg})
    print("RAG Chain Response: ", response['answer'])
    return str(response['answer']) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
