📚 Academic Assistant (CSV-based RAG)

This project implements a Retrieval-Augmented Generation (RAG) powered Academic Assistant for VIT University.
The assistant retrieves structured academic information (courses, subjects, credits, prerequisites, labs, electives, etc.) from a CSV dataset and answers student queries using Ollama LLM with Chroma VectorDB and HuggingFace Embeddings.

🚀 Features

✅ Load academic data from a CSV file into Chroma VectorDB

✅ Use Alibaba-NLP gte-large-en-v1.5 embeddings for semantic search

✅ Retrieve and process relevant chunks with RecursiveCharacterTextSplitter

✅ Generate student-friendly answers with Ollama (Llama2)

✅ Ensure accuracy by avoiding hallucinations – replies "I don’t know" if data is missing

✅ Present results in a structured format (tables when possible)

✅ Optional AGNO Agent integration with Gemini + Reasoning Tools

🛠️ Tech Stack

Python 3.9+

LangChain

Chroma VectorDB

HuggingFace Embeddings

Ollama
 (Llama2 local LLM)

dotenv

AGNO
 (optional)

📂 Project Structure
.
├── resources/
│   └── academics/       # Chroma vectorstore directory
├── result.csv           # CSV dataset with academic information
├── academic_assistant.py # Main script
└── README.md            # Documentation

⚙️ Installation

Clone the repository

git clone https://github.com/yourusername/academic-assistant.git
cd academic-assistant


Create & activate a virtual environment

python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Install Ollama (for Llama2)

Download Ollama

Pull the Llama2 model:

ollama pull llama2


Set environment variables
Create a .env file:

TOKENIZERS_PARALLELISM=false
