'''This module defines the prompt template for the medical question-answering system.'''

from langchain.prompts import ChatPromptTemplate

system_prompt = (
    "You are a helpful assistant that provides information about medical conditions based on the retrieved documents."
    " Use the following retrieved documents to answer the user's question."
    " If the retrieved documents do not contain relevant information, respond with 'I don't know'."
    "Dont hallicinate and make up answers. Always use the retrieved documents to answer the question."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)