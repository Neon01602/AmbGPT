import os
import argparse

# ---------------- IMPORTS ---------------- #
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
# ---------------------------------------- #

# ---------------- PROMPT ---------------- #
template = """
You are AmbedkarGPT. Use the provided context to answer the question below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# ---------------- CONFIG ---------------- #
PERSIST_DIR = "./chroma_db"
SPEECH_FILE = r"D:\GDG\speech.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ---------------- VECTORSTORE ---------------- #
def build_or_load_vectorstore(speech_path: str, persist_directory: str = PERSIST_DIR):
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print(f"Loading persisted Chroma DB from {persist_directory}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    print("Persisted DB not found → Building vectorstore from speech.txt...")
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()
    if len(docs) == 0:
        raise ValueError("ERROR: speech.txt is empty or not found!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(docs)
    if len(split_docs) == 0:
        raise ValueError("Error: Splitter returned 0 chunks!")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vectorstore successfully built at: {persist_directory}")
    return vectordb

# ---------------- QA CHAIN ---------------- #
def create_qa_chain(vectordb):
    try:
        llm = Ollama(model="mistral")
    except Exception:
        print("❌ ERROR: Ollama is not running. Start Ollama and pull Mistral model:\n   ollama pull mistral")
        raise

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )
    return qa

# ---------------- CLI ---------------- #
def cli_loop(qa_chain):
    print("\nAmbedkarGPT - Ask questions about the speech. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not question or question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        try:
            result = qa_chain.invoke(question)
            answer = result.get("result")
            print("\nA:", answer)

            sources = result.get("source_documents", [])
            if sources:
                print("\n--- Sources ---")
                for i, doc in enumerate(sources, 1):
                    snippet = doc.page_content.replace("\n", " ")[:300]
                    print(f"[{i}] {snippet}...")
                print("----------------\n")
        except Exception as e:
            print(f"Error: {e}")

# ---------------- MAIN ---------------- #
def main():
    parser = argparse.ArgumentParser(description="AmbedkarGPT - RAG CLI (Ollama + Mistral)")
    parser.add_argument("--speech", type=str, default=SPEECH_FILE)
    parser.add_argument("--persist_dir", type=str, default=PERSIST_DIR)
    args = parser.parse_args(args=[])

    vectordb = build_or_load_vectorstore(
        speech_path=args.speech,
        persist_directory=args.persist_dir
    )

    qa_chain = create_qa_chain(vectordb)
    cli_loop(qa_chain)

if __name__ == "__main__":
    main()
