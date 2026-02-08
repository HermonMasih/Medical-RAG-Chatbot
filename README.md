# Medical-RAG-Chatbot

A Generative AI-powered medical chatbot built on Retrieval-Augmented Generation (RAG) architecture. This chatbot leverages LangChain, Pinecone vector database, and HuggingFace embeddings to provide accurate, document-based medical information retrieval and question answering.

## Overview

Medical-RAG-Chatbot is a conversational AI system designed to answer medical questions by retrieving relevant information from a knowledge base of medical documents (PDFs). It uses RAG to combine retrieval-based and generative capabilities, ensuring responses are grounded in actual medical documents rather than hallucinated content.

## Architecture

The application follows a typical RAG pipeline:

1. **Document Loading**: PDFs are loaded from a designated directory
2. **Text Processing**: Documents are filtered and split into manageable chunks
3. **Embeddings**: Chunks are converted to embeddings using HuggingFace models
4. **Vector Storage**: Embeddings are stored in Pinecone vector database
5. **Retrieval**: User queries retrieve the most relevant documents
6. **Generation**: A language model generates responses based on retrieved context

## Features

- **Document-Based QA**: Answers questions based on uploaded medical documents
- **Vector Similarity Search**: Uses semantic search to find relevant document chunks
- **Multi-LLM Support**: Integrates with OpenAI/Groq models via HuggingFace API
- **Conversational Interface**: User-friendly web chat interface
- **Context-Aware Responses**: Prevents hallucination by requiring document-based answers
- **No Hard-coded Knowledge**: All information comes from indexed documents

## Prerequisites

- Python 3.8 or higher
- Pinecone account (for vector database)
- HuggingFace API key
- API credentials for language model endpoints
- Medical PDF documents to index

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Medical-RAG-Chatbot.git
   cd Medical-RAG-Chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` file** in the project root with the following variables:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```

2. **Prepare your data**:
   - Create a `pdfs/` directory in the project root
   - Place your medical PDF documents in this directory

3. **Initialize the vector database**:
   ```bash
   python src/store_index.py
   ```
   This script will:
   - Load all PDFs from the `pdfs/` directory
   - Create embeddings using HuggingFace models
   - Initialize a Pinecone index and store the embeddings

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Access the chatbot**:
   - Open your browser and navigate to `http://localhost:8080`
   - The web interface will load automatically

3. **Interact with the chatbot**:
   - Type your medical questions in the message input field
   - The chatbot will retrieve relevant documents and provide answers
   - If no relevant information is found, it will respond with "I don't know"

## Project Structure

```
Medical-RAG-Chatbot/
├── app.py                      # Flask application and main API endpoints
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup configuration
├── template.sh                 # Shell script template
├── README.md                   # Project documentation
├── LICENSE                     # License file
├── data_files/                 # Directory for data storage
├── pdfs/                       # Directory for input medical PDFs
├── src/
│   ├── __init__.py
│   ├── helper.py              # PDF loading, chunking, and embedding utilities
│   ├── prompt.py              # LLM prompt templates
│   ├── store_index.py         # Pinecone index initialization and management
│   └── __pycache__/           # Python cache
├── static/
│   ├── style.css              # Chat interface styling
├── templates/
│   ├── chat.html              # Web interface template
└── medical_chatbot.egg-info/  # Package metadata
```

## Technologies Used

### Core Frameworks
- **Flask** - Web framework for REST API
- **LangChain** - LLM orchestration and RAG implementation
- **Pinecone** - Vector database for embeddings storage

### AI & NLP
- **Sentence Transformers** - Embedding generation (HuggingFace)
- **OpenAI/Groq** - Language models for response generation
- **LangChain Community** - Additional integrations

### Data Processing
- **PyPDF** - PDF document loading and parsing
- **RecursiveCharacterTextSplitter** - Document chunking

### Frontend
- **HTML/CSS** - User interface
- **jQuery** - Frontend interactions
- **Bootstrap** - Responsive design

## API Endpoints

### GET /
- Returns the main chat interface

### POST /get
- **Parameters**: 
  - `msg` (string): User's question
- **Returns**: Chatbot response based on retrieved documents

## Key Components

### helper.py
- `load_pdfs()`: Loads PDF files from a directory
- `filter_pdf_documents()`: Filters and formats documents
- `perform_chunking()`: Splits documents into chunks for embedding
- `get_embeddings()`: Creates HuggingFace embeddings

### prompt.py
- Defines system prompts for the chatbot
- Ensures responses are grounded in retrieved documents
- Prevents hallucination through explicit instructions

### store_index.py
- Manages Pinecone vector index initialization
- Handles document uploading to vector store
- Implements serverless Pinecone infrastructure setup

## Important Notes

- The chatbot will only answer based on provided documents - it won't use external knowledge
- Large PDFs should be chunked appropriately (default: 500 characters per chunk)
- Pinecone index creation may take some time with large document collections
- Vector embeddings are stored on Pinecone cloud, not locally

## Troubleshooting

**No answers from chatbot**: 
- Ensure PDFs are in the `pdfs/` directory before running `store_index.py`
- Verify Pinecone index is created successfully

**API key errors**:
- Check `.env` file has correct API keys
- Verify keys have proper permissions in respective services

**Memory issues**:
- Reduce chunk size in helper.py if processing many large PDFs
- Adjust search parameters (k value) in app.py retriever

## Future Enhancements

- Multi-language support
- Document upload via web interface
- Advanced filtering and metadata search
- Conversation history management
- Performance optimization with caching
- Integration with additional medical databases

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Created by Hermon Masih

## Disclaimer

This chatbot is for educational purposes. Always consult with qualified healthcare professionals for medical advice. The information provided is based on indexed documents and should not be used as a substitute for professional medical consultation.
