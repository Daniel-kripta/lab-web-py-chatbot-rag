import logging
import re
import time

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from chatbot import chat, get_history

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API")

CHROMA_PATH       = "./chroma_db"
CHROMA_COLLECTION = "docs"
RATE_LIMIT        = 10   # peticiones por minuto por IP
MAX_QUESTION      = 500  # caracteres

# {ip: [timestamps de los últimos requests]}
_rate_store: dict[str, list[float]] = {}

PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # email
    r"\b\d{8,9}[A-Za-z]?\b",                                   # DNI / teléfono
]


class ChatRequest(BaseModel):
    pregunta:   str = Field(..., max_length=MAX_QUESTION)
    session_id: str


class ChatResponse(BaseModel):
    respuesta:         str
    fuentes:           list[str]
    session_id:        str
    fragmentos_usados: int
    pii_warning:       str | None = None


def check_rate_limit(ip: str) -> None:
    now    = time.time()
    window = now - 60
    recent = [t for t in _rate_store.get(ip, []) if t > window]
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Límite alcanzado: máximo 10 peticiones por minuto.",
        )
    recent.append(now)
    _rate_store[ip] = recent


def detect_pii(text: str) -> str | None:
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return (
                "Tu pregunta parece contener información personal (email o DNI/teléfono). "
                "Se ha procesado igualmente, pero evita compartir datos personales."
            )
    return None


@app.post("/chat", response_model=ChatResponse)
async def post_chat(request: Request, body: ChatRequest):
    ip = request.client.host
    check_rate_limit(ip)

    pii_warning = detect_pii(body.pregunta)

    logger.info(
        "POST /chat  session=%s  ip=%s  chars=%d  pii=%s",
        body.session_id, ip, len(body.pregunta), pii_warning is not None,
    )

    result = chat(body.pregunta, body.session_id)
    result["pii_warning"] = pii_warning
    return result


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    return {"session_id": session_id, "history": get_history(session_id)}


@app.get("/documentos")
async def get_documentos():
    chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(CHROMA_COLLECTION)
    data       = collection.get(include=["metadatas"])

    docs: dict[str, dict] = {}
    for meta in data["metadatas"]:
        filename = meta["filename"]
        if filename not in docs:
            docs[filename] = {
                "filename":     filename,
                "category":     meta.get("category", "general"),
                "total_chunks": meta.get("total_chunks", 0),
            }

    return {"documentos": list(docs.values()), "total": len(docs)}
