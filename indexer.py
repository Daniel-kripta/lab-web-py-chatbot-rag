from pathlib import Path
from app.config import settings
from app.services.embeddings_service import embeddings_service
from app.core.chromadb_client import get_collection

DOCS_DIR = Path(__file__).parent / "docs"

CATEGORY_MAP: dict[str, str] = {
    "atractivos.txt": "ocio",
    "naturaleza.txt": "medio_ambiente",
    "geografia.txt": "medio_ambiente",
    "datosgenerales.txt": "general",
    "historia.txt": "historia"
}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def index_documents() -> None:
    print("\n" + "=" * 55)
    print("  NexusBot — Indexador de documentos")
    print("=" * 55)

    embeddings_service.load()

    collection = get_collection()


    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"\n  Limpiados {len(existing['ids'])} chunks anteriores.")

    txt_files = sorted(DOCS_DIR.glob("*.txt"))
    if not txt_files:
        print(f"\n  ERROR: No se encontraron .txt en {DOCS_DIR}")
        return

    total_chunks = 0
    total_docs = 0

    print(f"\n  Indexando {len(txt_files)} documentos...\n")

    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8")
        filename = txt_file.name
        category = CATEGORY_MAP.get(filename)
        if category is None:
            print(f" {filename} sin categoría en CATEGORY_MAP, usando 'general'")
            category = "general"

        chunks = chunk_text(content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        embeddings = embeddings_service.embed_batch(chunks)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings_list: list[list[float]] = []

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
            embeddings_list.append(embedding)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )

        print(f"  ✓  {filename:<40} {len(chunks):>3} chunks  [{category}]")
        total_chunks += len(chunks)
        total_docs += 1

    dims = len(embeddings_list[0]) if embeddings_list else 0  
    print("\n" + "=" * 55)
    print(f"  Documentos indexados : {total_docs}")
    print(f"  Chunks totales       : {total_chunks}")
    print(f"  Dimensiones embedding: {dims}")
    print(f"  Modelo embeddings    : {settings.EMBEDDING_MODEL}")
    print(f"  ChromaDB             : {settings.CHROMA_PATH}/{settings.CHROMA_COLLECTION}")
    print("=" * 55)
    print("\n  ¡Indexación completa! Ya puedes arrancar la API:")
    print("  uvicorn app.main:app --reload\n")


if __name__ == "__main__":
    index_documents()