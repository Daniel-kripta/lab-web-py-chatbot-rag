import os
import re

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OLLAMA_URL  = os.getenv("OLLAMA_URL",        "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL",     "qwen3:14b")
EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
CHROMA_PATH  = "./chroma_db"
CHROMA_COLLECTION = "docs"
TOP_K = 3

SYSTEM_PROMPT = """Eres un asistente que responde preguntas basándose ÚNICAMENTE en los documentos proporcionados como contexto.

Reglas:
- Responde solo con la información presente en el contexto.
- Si el contexto no contiene información suficiente, responde exactamente: "No tengo información sobre eso."
- No inventes datos, fechas, nombres ni hechos."""

client     = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(CHROMA_COLLECTION)

_histories: dict[str, list[dict]] = {}


def _strip_thinking(text: str) -> str:
    # qwen3 puede emitir <think>...</think> aunque se desactive con extra_body
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def chat(pregunta: str, session_id: str) -> dict:
    embed_resp = client.embeddings.create(input=[pregunta], model=EMBED_MODEL)
    query_embedding = embed_resp.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    fragments = results["documents"][0]
    metadatas = results["metadatas"][0]

    # dict.fromkeys preserva el orden de relevancia al deduplicar
    fuentes = list(dict.fromkeys(m["filename"] for m in metadatas))

    context = "\n\n---\n\n".join(fragments)
    user_message = f"Contexto:\n{context}\n\nPregunta: {pregunta}"

    history = _histories.setdefault(session_id, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        extra_body={"think": False},
    )
    respuesta = _strip_thinking(response.choices[0].message.content)

    # Solo pregunta/respuesta en el historial, sin el contexto RAG
    history.append({"role": "user",      "content": pregunta})
    history.append({"role": "assistant", "content": respuesta})

    return {
        "respuesta":        respuesta,
        "fuentes":          fuentes,
        "session_id":       session_id,
        "fragmentos_usados": len(fragments),
    }


def get_history(session_id: str) -> list[dict]:
    return _histories.get(session_id, [])
