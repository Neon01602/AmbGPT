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
# Install Ollama (macOS/Linux instructions). See https://ollama.ai for platform-specific steps
curl -fsSL https://ollama.ai/install.sh | sh
```
2. Install Ollama and pull the Mistral model (run these in your terminal):
```bash
# Start ollama daemon (if required) and pull the model
ollama pull mistral
```
3. Verify Ollama is running:
```bash
ollama list
```
You should see *mistral* listed.

# 🧠 AmbedkarGPT – Intern Assignment (Phase 1)

## 📁 Files

### 📄 `speech.txt`
➤ Contains the provided speech excerpt to ingest.

### 🧩 `main.py`
The command-line tool responsible for:
- 📌 Loading `speech.txt`
- ✂️ Splitting the text into chunks
- 🗂️ Building or loading a **ChromaDB vectorstore** using HuggingFace embeddings
- 🤖 Running a **CLI question–answer loop** grounded only in the speech context

---

## Usage

### 1. Install Ollama & Pull Mistral Model

```bash
ollama pull mistral
```
2. Run the CLI Tool
```bash
python main.py
```
First Run Behavior
Builds a new Chroma vectorstore inside:

```bash

./chroma_db
Subsequent runs automatically reuse the persisted database for faster startup.
```
## Notes / Tips
The system answers ONLY from speech.txt (pure RAG, no external knowledge).

The LLM receives only the retrieved relevant chunks.

If you face issues with Ollama:

Ensure the Ollama service is running.

Verify with:

```bash
ollama --version
```
**This is a prototype**, not production-ready:

Better error handling may be needed.

More robust prompt engineering could improve reliability.

Caching strategies may optimize repeated queries.

##Contact
For questions regarding this implementation, please contact
the author / candidate.
