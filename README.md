# AmbedkarGPT-Intern-Task
**Simple command-line Q&A system** that ingests a short speech by Dr. B.R.
Ambedkar and answers questions using a local RAG pipeline (LangChain + Chroma +
HuggingFace embeddings + Ollama Mistral 7B).
---
## Features
- Loads `speech.txt` and splits it into chunks.
- Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores embeddings in a local ChromaDB vectorstore.
- Retrieves relevant chunks for a user question and answers using an LLM.
- Primary LLM: **Ollama (Mistral 7B)** (local, free). The code includes a
fallback to a HuggingFace model if Ollama is not available.
---
## Requirements
- Python 3.8+
- Install dependencies with `pip install -r requirements.txt` (see file
included).
---
## Setup
1. Clone this repository:
```bash
git clone <repo-url>
cd AmbedkarGPT-Intern-Task
