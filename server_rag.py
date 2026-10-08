"""
Servidor MCP (stdio) de la biblioteca técnica de Vertex Coders.

IMPORTANTE: en un servidor MCP por stdio, stdout es el canal JSON-RPC.
Nunca usar print(); todo el log va a stderr con logging.

Herramientas:
- search_technical_library(query, n_results, book): búsqueda semántica
- list_books(): libros indexados y número de fragmentos
"""
import logging
import os
import sys
from collections import Counter

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[vertex-rag] %(levelname)s %(message)s",
)
for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("vertex-rag")

from chromadb import PersistentClient
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "vertex_technical_library"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
MAX_RESULTS = 10

mcp = FastMCP("vertex-rag-library")

# Carga perezosa: el servidor arranca al instante y el modelo se carga
# en la primera consulta (evita timeouts de arranque en Claude Desktop).
_state: dict = {"collection": None, "model": None, "model_name": None}


def _load_model(name: str) -> SentenceTransformer:
    try:
        # Primero desde la caché local, sin tocar la red
        return SentenceTransformer(name, local_files_only=True)
    except Exception:
        log.info("Modelo %s no está en caché; descargándolo...", name)
        return SentenceTransformer(name)


def _get_backend():
    """Devuelve (collection, model, model_name) o lanza RuntimeError con mensaje útil."""
    if _state["collection"] is None:
        client = PersistentClient(path=DB_DIR)
        try:
            col = client.get_collection(COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                "La colección no existe todavía. Ejecuta 'python index_books.py' "
                "desde F:\\vertex-rag-library con el .venv activado."
            )
        _state["collection"] = col
        log.info("ChromaDB conectada: %s (%d fragmentos)", DB_DIR, col.count())

    col = _state["collection"]
    model_name = (col.metadata or {}).get("embedding_model", DEFAULT_MODEL)
    if _state["model"] is None or _state["model_name"] != model_name:
        log.info("Cargando modelo de embeddings: %s", model_name)
        _state["model"] = _load_model(model_name)
        _state["model_name"] = model_name
    return col, _state["model"], model_name


def _query_prefix(model_name: str) -> str:
    # Los modelos E5 se entrenaron con prefijos "query: " / "passage: "
    return "query: " if "e5" in model_name.lower() else ""


@mcp.tool()
def search_technical_library(query: str, n_results: int = 4, book: str | None = None) -> str:
    """
    Busca información técnica, fragmentos de código, esquemas o conceptos en la
    biblioteca privada de libros (programación, Jetpack Compose, IA, MCP, hacking,
    redes, manuales de hardware). Funciona con consultas en español o inglés.

    Args:
        query: Concepto, pregunta técnica, función o fragmento a buscar.
        n_results: Número de fragmentos a recuperar (1-10, por defecto 4).
        book: Opcional. Nombre exacto del archivo para buscar solo en ese libro
              (usa list_books para ver los nombres).
    """
    try:
        col, model, model_name = _get_backend()
        total = col.count()
        if total == 0:
            return "[!] La base vectorial está vacía. Ejecuta 'python index_books.py' primero."

        n = max(1, min(int(n_results), MAX_RESULTS, total))
        emb = model.encode(_query_prefix(model_name) + query, normalize_embeddings=True).tolist()

        kwargs = {"query_embeddings": [emb], "n_results": n,
                  "include": ["documents", "metadatas", "distances"]}
        if book:
            kwargs["where"] = {"source": book}
        res = col.query(**kwargs)

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not docs:
            scope = f" en '{book}'" if book else ""
            return f"No se encontró información relevante{scope} para: '{query}'."

        out = [f"### Biblioteca técnica — {len(docs)} resultados para: '{query}'\n"]
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            meta = meta or {}
            src = meta.get("source", "desconocido")
            page = meta.get("page")
            loc = f"{src}, pág. {page}" if page else src
            relevance = max(0.0, 1.0 - float(dist))  # distancia coseno -> similitud
            out.append(f"**[{i}] {loc}** (relevancia {relevance:.2f})\n{doc}\n\n---")
        return "\n".join(out)

    except RuntimeError as e:
        return f"[!] {e}"
    except Exception as e:
        log.exception("Error en search_technical_library")
        return f"Error consultando la base vectorial: {type(e).__name__}: {e}"


@mcp.tool()
def list_books() -> str:
    """Lista los libros/documentos indexados en la biblioteca y cuántos fragmentos tiene cada uno."""
    try:
        col, _, model_name = _get_backend()
        metas = col.get(include=["metadatas"]).get("metadatas") or []
        counts = Counter((m or {}).get("source", "desconocido") for m in metas)
        if not counts:
            return "La biblioteca está vacía."
        lines = [f"### {len(counts)} documentos indexados ({sum(counts.values())} fragmentos, modelo {model_name})\n"]
        lines += [f"- {name} — {n} fragmentos" for name, n in sorted(counts.items())]
        return "\n".join(lines)
    except RuntimeError as e:
        return f"[!] {e}"
    except Exception as e:
        log.exception("Error en list_books")
        return f"Error listando la biblioteca: {type(e).__name__}: {e}"


if __name__ == "__main__":
    log.info("Servidor MCP vertex-rag-library iniciado (DB: %s)", DB_DIR)
    mcp.run()
