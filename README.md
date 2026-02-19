# Medical-RAG-Chatbot

A conversational AI system for medical question-answering that combines Retrieval-Augmented Generation (RAG) with LangChain, Pinecone vector storage, and HuggingFace embeddings. The chatbot maintains conversation history through Conversation Buffer Memory, delivering contextually-aware responses grounded in medical documents.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup Instructions](#detailed-setup-instructions)
- [API Keys & Configuration](#api-keys--configuration)
- [Running the Application](#running-the-application)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [API Endpoints](#api-endpoints)
- [Key Components](#key-components)
- [Troubleshooting](#troubleshooting)
- [Docker Deployment](#docker-deployment)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview

Medical-RAG-Chatbot is a conversational AI system designed to answer medical questions by retrieving relevant information from a knowledge base of medical documents (PDFs). It uses RAG to combine retrieval-based and generative capabilities, ensuring responses are grounded in actual medical documents rather than hallucinated content.

## Features

- ✅ **Document-Based QA**: Answers questions based on uploaded medical documents
- ✅ **Vector Similarity Search**: Uses semantic search to find relevant document chunks
- ✅ **Multi-LLM Support**: Integrates with OpenAI/Groq models via HuggingFace API
- ✅ **Conversational Interface**: User-friendly web chat interface
- ✅ **Context-Aware Responses**: Prevents hallucination by requiring document-based answers
- ✅ **Conversation History**: Maintains context across multiple turns
- ✅ **No Hard-coded Knowledge**: All information comes from indexed documents

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (Python package manager, typically included with Python)
- **git** (for cloning the repository) - [Download Git](https://git-scm.com/downloads)

### External Services (Free/Paid accounts required):

1. **Pinecone Account** - Vector database
   - Visit: https://www.pinecone.io/
   - Sign up for a free account

2. **HuggingFace Account** - For API access
   - Visit: https://huggingface.co/
   - Sign up for a free account

3. **Groq or OpenAI Account** - Language model provider
   - Groq: https://console.groq.com/ (recommended, free tier available)
   - OpenAI: https://platform.openai.com/

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Medical-RAG-Chatbot.git
cd Medical-RAG-Chatbot
```

### 2. Set Up Python Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### 5. Prepare Your Medical Documents
Place your PDF files in the location specified in `src/helper.py` (default location to configure).

### 6. Initialize Vector Database
```bash
python src/store_index.py
```

### 7. Run the Application
```bash
python app.py
```

Visit `http://localhost:8080` in your browser and start asking medical questions!

## Detailed Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Medical-RAG-Chatbot.git
cd Medical-RAG-Chatbot
```

### Step 2: Create a Virtual Environment

It's highly recommended to use a virtual environment to avoid dependency conflicts.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install the following packages:
- LangChain & related packages
- Flask
- Sentence Transformers
- PyPDF
- Python-dotenv
- Langchain-Pinecone & Langchain-OpenAI

### Step 4: Verify Installation

```bash
python -c "import flask; import langchain; import pinecone; print('All dependencies installed successfully!')"
```

## API Keys & Configuration

### 4.1 Get Your Pinecone API Key

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign in or create an account
3. Navigate to **API keys** in the left sidebar
4. Copy your API key
5. Note your **Region** (e.g., us-east-1)

### 4.2 Get Your HuggingFace API Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Sign in to your account
3. Click **New token**
4. Give it a name (e.g., "Medical-RAG-Chatbot")
5. Set role to **read**
6. Copy the token

### 4.3 Get Your Groq API Key (Recommended)

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to **API Keys**
4. Create a new API key
5. Copy it

### 4.4 Create .env File

Create a file named `.env` in the project root directory (same level as `app.py`):

```env
# Pinecone Configuration
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_INDEX_NAME=medical-rag-index

# HuggingFace Configuration
HUGGINGFACEHUB_API_TOKEN=your_actual_huggingface_token_here

# Optional: Model configuration
# GROQ_API_KEY=your_groq_api_key (if using Groq instead of HuggingFace router)
```

⚠️ **Important**: Never commit the `.env` file to version control. It's already listed in `.gitignore`.

### Step 5: Prepare Your Medical Documents

1. Collect your medical PDF files
2. Create or identify the directory where PDFs will be stored (check `src/helper.py` for the configured path)
3. Place all PDF files in that directory

### Step 6: Initialize the Pinecone Index

Before running the application, you need to create and populate the Pinecone vector database:

```bash
python src/store_index.py
```

This script will:
- Read PDF files from your documents directory
- Split documents into chunks
- Generate embeddings using HuggingFace models
- Create a Pinecone index (if it doesn't exist)
- Upload embeddings to Pinecone

**Note**: This process may take several minutes depending on the number and size of your documents.

## Architecture

The application follows a typical RAG pipeline:

1. **Document Loading**: PDFs are loaded from a designated directory
2. **Text Processing**: Documents are filtered and split into manageable chunks (default: 500 characters)
3. **Embeddings**: Chunks are converted to embeddings using HuggingFace models
4. **Vector Storage**: Embeddings are stored in Pinecone vector database
5. **Retrieval**: User queries retrieve the most relevant documents (default: top 3)
6. **Generation**: A language model generates responses based on retrieved context
7. **Memory**: Conversation buffer maintains context across multiple turns

## Running the Application

### Starting the Server

```bash
python app.py
```

You should see output similar to:
```
 * Running on http://0.0.0.0:8080
 * Debug mode: on
```

### Accessing the Web Interface

1. Open your web browser
2. Navigate to `http://localhost:8080`
3. You should see the chat interface

### Interacting with the Chatbot

1. Type your medical question in the message input field
2. Press **Send** or **Enter**
3. The chatbot will:
   - Retrieve relevant documents from Pinecone
   - Generate a response based on the retrieved information
   - Display the answer in the chat window
4. Continue the conversation - the chatbot maintains conversation history

## Project Structure

```
Medical-RAG-Chatbot/
├── app.py                      # Flask application and main API endpoints
├── Dockerfile                  # Docker configuration for containerization
├── LICENSE                     # License file
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup configuration
├── template.sh                 # Shell script template
├── .env.example               # Example environment variables (create .env based on this)
├── data_files/                 # Directory for storing vector indices and temporary data
├── pdfs/                       # Directory for storing medical PDF documents (create this)
├── research/
│   └── trials.ipynb           # Jupyter notebook for experimental trials and research
├── src/
│   ├── __init__.py            # Package initialization
│   ├── helper.py              # PDF loading, chunking, and embedding utilities
│   ├── prompt.py              # LLM prompt templates and system messages
│   ├── store_index.py         # Pinecone index initialization and management
│   └── __pycache__/           # Python cache directory
├── static/
│   └── style.css              # Chat interface styling
└── templates/
    └── chat.html              # Web chat interface HTML
```

## Technologies Used

### Core Frameworks & Libraries
- **Flask** (3.1.1) - Web framework for REST API and server
- **LangChain** (0.3.26) - LLM orchestration and RAG chain management
- **Pinecone** - Vector database for semantic search and embeddings storage
- **Sentence Transformers** (4.1.0) - State-of-the-art embedding generation

### AI & Language Models
- **HuggingFace** - Model hub and API for embeddings and language models
- **Groq APIs** - Fast language model inference (recommended for speed)
- **OpenAI** (via HuggingFace router) - Alternative language model provider
- **LangChain-Community** (0.3.26) - Community integrations and tools

### Data Processing & PDF Handling
- **PyPDF** (5.6.1) - PDF document loading and parsing
- **RecursiveCharacterTextSplitter** - Intelligent document chunking

### Frontend & UI
- **HTML/CSS** - User interface
- **jQuery** - Frontend interactions and AJAX calls
- **Bootstrap** - Responsive design framework

## API Endpoints

### GET /
- **Description**: Renders the main chat interface
- **Response**: HTML page with the chat UI
- **Example**: Visit `http://localhost:8080/` in your browser

### POST /get
- **Description**: Processes user input and returns chatbot response
- **Method**: POST
- **Parameters**: 
  - `msg` (string, required): User's question or input
- **Returns**: JSON response with the chatbot's answer based on retrieved documents
- **Example**:
  ```bash
  curl -X POST http://localhost:8080/get -d "msg=What is diabetes?" 
  ```

## Key Components

### app.py
The main Flask application that:
- Sets up the web server running on port 8080
- Initializes Pinecone vector store and retriever
- Creates the RAG chain with conversation memory
- Handles the `/get` endpoint for chatbot queries
- Manages conversation history across multiple turns

### src/helper.py
Utility functions for document processing:
- `load_pdfs()`: Loads PDF files from the configured directory
- `filter_pdf_documents()`: Validates and filters documents
- `perform_chunking()`: Splits documents into manageable chunks using RecursiveCharacterTextSplitter
- `get_embeddings()`: Initializes HuggingFace embeddings for vector conversion

### src/prompt.py
Defines the system and conversation prompts:
- `SYSTEM_PROMPT`: Instructions to ensure responses are grounded in retrieved documents
- Prevents hallucination by enforcing document-based answers only
- Customizable for different medical domains or response styles

### src/store_index.py
Manages vector database operations:
- Loads and processes medical PDF documents
- Creates embeddings using HuggingFace models
- Initializes Pinecone index if it doesn't exist
- Uploads document embeddings to Pinecone
- Run this script before starting the application for the first time

### templates/chat.html
Frontend chat interface:
- HTML structure for the chat UI
- jQuery code for sending messages via AJAX
- Real-time message display
- Responsive design using Bootstrap

### static/style.css
Styling for the chat interface:
- Layout and responsive design
- Colors and typography
- Message bubble styling

## Important Notes & Best Practices

- ✅ The chatbot will only answer based on provided documents - it won't use external knowledge
- ✅ Large PDFs should be chunked appropriately (default: 500 characters per chunk)
- ✅ Pinecone index creation may take some time with large document collections
- ✅ Vector embeddings are stored on Pinecone cloud, not locally
- ✅ Always keep your `.env` file secure and never commit it to version control
- ✅ Test with small document sets first before indexing large collections
- ✅ Monitor Pinecone usage to stay within your plan's limits

## Troubleshooting

### 1. **ImportError: No module named 'flask' or other packages**

**Problem**: Python can't find the required packages

**Solutions**:
- Ensure your virtual environment is activated:
  ```bash
  # Windows
  venv\Scripts\activate
  
  # macOS/Linux
  source venv/bin/activate
  ```
- Reinstall dependencies:
  ```bash
  pip install -r requirements.txt --upgrade
  ```
- Check Python version is 3.8+:
  ```bash
  python --version
  ```

### 2. **API Key Errors (PINECONE_API_KEY not found, etc.)**

**Problem**: Environment variables are not being loaded

**Solutions**:
- Verify `.env` file exists in the project root (same directory as `app.py`)
- Check `.env` file has correct format (no quotes around values):
  ```env
  PINECONE_API_KEY=your_key_without_quotes
  HUGGINGFACEHUB_API_TOKEN=your_token_without_quotes
  ```
- Restart the application after creating/modifying `.env`
- On Windows, ensure `.env` file extension is correct (not `.env.txt`)

### 3. **"No valid Pinecone index found" or "Index doesn't exist"**

**Problem**: Pinecone index hasn't been created or initialized

**Solutions**:
- Make sure you have PDF documents in the correct directory
- Run the initialization script before starting the app:
  ```bash
  python src/store_index.py
  ```
- Check your `PINECONE_INDEX_NAME` in `.env` matches what you created
- Verify Pinecone account is active and has available credits
- Check Pinecone console to ensure index was created successfully

### 4. **"No relevant documents found" or "I don't know" responses**

**Problem**: Chatbot can't find relevant information in documents

**Solutions**:
- Verify PDFs were processed correctly:
  - Check that PDFs are readable and not corrupted
  - Ensure PDFs contain actual text (not scanned images without OCR)
- Check that `store_index.py` completed successfully
- Try asking simpler, more specific questions
- Verify documents are relevant to your questions
- Check the chunk size in `src/helper.py` - very small chunks might reduce relevance

### 5. **Connection Error: "Failed to connect to Pinecone"**

**Problem**: Network connectivity issue with Pinecone

**Solutions**:
- Check your internet connection
- Verify `PINECONE_API_KEY` is correct and active
- Check if Pinecone services are online: https://status.pinecone.io/
- Try again after a few minutes
- Check firewall/proxy settings if on a corporate network

### 6. **"HuggingFace API Error" or Token Issues**

**Problem**: HuggingFace authentication failed

**Solutions**:
- Verify `HUGGINGFACEHUB_API_TOKEN` is correct
- Generate a new token from https://huggingface.co/settings/tokens if needed
- Ensure token has **read** permissions
- Check that token is not expired

### 7. **Port 8080 Already in Use**

**Problem**: Another application is using port 8080

**Solutions**:
- Find and stop the other application using the port
- Change the port in `app.py`:
  ```python
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8081, debug=True)  # Change 8080 to 8081
  ```
- Access at `http://localhost:8081`

### 8. **Memory Issues with Large PDFs**

**Problem**: Application crashes or runs slowly with many/large documents

**Solutions**:
- Reduce chunk size in `src/helper.py`:
  ```python
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=300,  # Reduce from 500
      chunk_overlap=50
  )
  ```
- Process documents in batches using `store_index.py`
- Increase system RAM if possible
- Use a smaller embedding model in `src/helper.py`

### 9. **Chat Interface Not Loading (Blank Page)**

**Problem**: Website shows blank page at `http://localhost:8080`

**Solutions**:
- Check that Flask app is running (check terminal for errors)
- Try refreshing the page (Ctrl+R or Cmd+R)
- Clear browser cache (Ctrl+Shift+Delete)
- Check browser console for JavaScript errors (F12 → Console)
- Ensure `templates/chat.html` file exists
- Check Flask routes are properly defined in `app.py`

### 10. **"No module named 'src'" Error**

**Problem**: Python can't find the src package

**Solutions**:
- Ensure you're running from the project root directory:
  ```bash
  cd Medical-RAG-Chatbot
  python app.py
  ```
- Check that `src/__init__.py` exists
- Verify PYTHONPATH includes the project root

### Getting Help

If you encounter an issue not listed here:

1. **Check the terminal output** - Look for error messages and stack traces
2. **Review the logs** - Check for any warnings or errors in the console
3. **Verify configuration** - Double-check `.env` file and API keys
4. **Test API keys independently** - Verify keys work in their respective dashboards
5. **Check dependencies** - Run `pip list` to verify all packages are installed
6. **Review documentation** - Check LangChain, Pinecone, and HuggingFace docs
7. **Create an issue** - Include error messages, your OS, Python version, and steps to reproduce

## Docker Deployment

### Building the Docker Image

```bash
docker build -t medical-rag-chatbot:latest .
```

### Running with Docker

```bash
docker run -p 8080:8080 --env-file .env medical-rag-chatbot:latest
```

### Using Docker Compose (Optional)

Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  chatbot:
    build: .
    ports:
      - "8080:8080"
    env_file:
      - .env
    volumes:
      - ./pdfs:/app/pdfs
      - ./data_files:/app/data_files
```

Then run:
```bash
docker-compose up
```

### Docker Notes
- Ensure `.env` file exists before running Docker commands
- Docker images will be larger due to all dependencies included
- Volume mounting allows accessing local PDFs from the container
- The application inside Docker runs the same as locally

## Future Enhancements

- 🔄 Multi-language support
- 📤 Document upload via web interface
- 🔍 Advanced filtering and metadata search
- ⚡ Performance optimization with caching
- 🏥 Integration with additional medical databases
- 👥 Multi-user support with authentication
- 📊 Analytics and usage tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Created by [Hermon Masih](https://www.linkedin.com/in/hermonmasih)

## Disclaimer

⚠️ **Important**: This chatbot is for educational and informational purposes only. Always consult with qualified healthcare professionals for medical advice. The information provided is based on indexed documents and should not be used as a substitute for professional medical consultation. Neither the developers nor the host assumes any responsibility for the use or misuse of this application.
