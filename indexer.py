import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "docs"
CHUNK_SIZE = 500   # palabras
CHUNK_OVERLAP = 50

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

CATEGORY_MAP: dict[str, str] = {
    "atractivos.txt": "ocio",
    "naturaleza.txt": "medio_ambiente",
    "geografia.txt": "medio_ambiente",
    "datosgenerales.txt": "general",
    "historia.txt": "historia",
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def index_documents() -> None:
    print("\n" + "=" * 55)
    print("  Indexador de documentos RAG")
    print("=" * 55)

    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(CHROMA_COLLECTION)

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"\n  Limpiados {len(existing['ids'])} chunks anteriores.")

    txt_files = sorted(DOCS_DIR.glob("*.txt"))
    if not txt_files:
        print(f"\n  ERROR: No se encontraron .txt en {DOCS_DIR}")
        return

    total_docs = 0
    total_chunks = 0

    print(f"\n  Indexando {len(txt_files)} documentos...\n")

    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8")
        filename = txt_file.name
        category = CATEGORY_MAP.get(filename, "general")

        chunks = chunk_text(content)

        response = client.embeddings.create(input=chunks, model=EMBED_MODEL)
        embeddings = [item.embedding for item in response.data]

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{filename}_chunk_{i:03d}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "chunk_id": chunk_id,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"  OK  {filename:<40} {len(chunks):>3} chunks  [{category}]")
        total_chunks += len(chunks)
        total_docs += 1

    print("\n" + "=" * 55)
    print(f"  Documentos indexados : {total_docs}")
    print(f"  Chunks totales       : {total_chunks}")
    print(f"  Modelo embeddings    : {EMBED_MODEL} (local, coste $0)")
    print(f"  ChromaDB             : {CHROMA_PATH}/{CHROMA_COLLECTION}")
    print("=" * 55)
    print("\n  Indexacion completa. Ya puedes arrancar la API:")
    print("  uvicorn api:app --reload\n")


if __name__ == "__main__":
    index_documents()
