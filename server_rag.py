"""
Servidor MCP (stdio) de la biblioteca técnica de Vertex Coders.

IMPORTANTE: en un servidor MCP por stdio, stdout es el canal JSON-RPC.
Nunca usar print(); todo el log va a stderr con logging.

Búsqueda HÍBRIDA:
- Semántica (embeddings E5 multilingües en ChromaDB): entiende el significado,
  sirve para preguntas en español sobre libros en inglés.
- Léxica (BM25, en memoria, sin dependencias): encuentra nombres exactos de
  código (mutableStateOf, stream=True, FastMCP...) en cualquier idioma.
- Se combinan con Reciprocal Rank Fusion (RRF).

Herramientas:
- search_technical_library(query, n_results, book)
- list_books()
"""
import logging
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

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
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
MAX_RESULTS = 10
CANDIDATES = 30      # candidatos de cada buscador antes de fusionar
RRF_K = 60           # constante estándar de Reciprocal Rank Fusion
MAX_CHUNK_CHARS = 1100

mcp = FastMCP("vertex-rag-library")

# Carga perezosa: el servidor arranca al instante; modelo e índice BM25
# se construyen en la primera consulta (evita timeouts de arranque).
_state: dict = {"collection": None, "model": None, "model_name": None, "bm25": None}


# ---------------------------------------------------------------------------
# Tokenización para BM25
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = set("""
a al algo como con cual de del el ella en es esta este esto la las lo los mas me mi
muy no o para pero por que se si sin su sus te tu un una uno unos unas y ya hay ser
fue son era cuando donde quien sobre entre tambien solo asi dame dime muestrame
the a an and or of to in on for with is are be was it this that as at by from
how what which can use using you your my me do does not no if then so into
""".split())


def _fold(text: str) -> str:
    """minúsculas + NFKC + sin tildes."""
    text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text).lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(_fold(text)) if len(t) > 1 and t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for idx, doc in enumerate(docs):
            toks = tokenize(doc)
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((idx, tf))
        self.n = len(docs)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 1.0

    def search(self, query: str, top: int, allowed: set[int] | None = None) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for idx, tf in plist:
                if allowed is not None and idx not in allowed:
                    continue
                dl = self.lengths[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / denom
        return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top]]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
def _load_model(name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(name, local_files_only=True)
    except Exception:
        log.info("Modelo %s no está en caché; descargándolo...", name)
        return SentenceTransformer(name)


def _get_backend():
    """Devuelve (collection, model, model_name, bm25_state)."""
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

    total = col.count()
    bm = _state["bm25"]
    if bm is None or bm["count"] != total:
        log.info("Construyendo índice BM25 (%d fragmentos)...", total)
        data = col.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = [m or {} for m in (data.get("metadatas") or [])]
        by_source: dict[str, set[int]] = defaultdict(set)
        for i, m in enumerate(metas):
            by_source[m.get("source", "desconocido")].add(i)
        bm = {
            "count": total, "ids": ids, "docs": docs, "metas": metas,
            "pos": {doc_id: i for i, doc_id in enumerate(ids)},
            "by_source": by_source, "index": BM25(docs),
        }
        _state["bm25"] = bm
        log.info("Índice BM25 listo.")
    return col, _state["model"], model_name, bm


def _query_prefix(model_name: str) -> str:
    return "query: " if "e5" in model_name.lower() else ""


def _resolve_book(book: str, sources) -> str | None:
    """Acepta el nombre exacto o un trozo del título (sin importar mayúsculas/tildes)."""
    if book in sources:
        return book
    key = _fold(book)
    matches = [s for s in sources if key in _fold(s)]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------
@mcp.tool()
def search_technical_library(query: str, n_results: int = 4, book: str | None = None) -> str:
    """
    Busca en la biblioteca privada de libros técnicos (programación, Jetpack Compose,
    Kotlin, Python, IA, agentes, MCP, OpenAI API, hacking, redes). Búsqueda híbrida:
    por significado y por palabras exactas.

    Cómo escribir la query para obtener buenos resultados:
    - Incluye SIEMPRE los nombres exactos de funciones, clases, parámetros o
      librerías tal como se escriben en código (ej: "remember mutableStateOf",
      "stream=True chat.completions", "FastMCP @mcp.tool").
    - Añade 3-6 palabras clave del tema; si el libro puede estar en inglés,
      pon también los términos en inglés (ej: "estado state remember mutableStateOf
      recomposition Jetpack Compose").
    - Un solo tema por búsqueda. Para comparar dos temas, haz dos búsquedas.

    Args:
        query: Palabras clave + nombres exactos de código del tema a buscar.
        n_results: Fragmentos a devolver (1-10, por defecto 4).
        book: Opcional. Nombre del archivo o parte del título para buscar solo
              en ese libro (usa list_books para ver los nombres).
    """
    try:
        col, model, model_name, bm = _get_backend()
        if bm["count"] == 0:
            return "[!] La base vectorial está vacía. Ejecuta 'python index_books.py' primero."

        n = max(1, min(int(n_results), MAX_RESULTS))
        source = None
        allowed = None
        if book:
            source = _resolve_book(book, bm["by_source"].keys())
            if not source:
                return (f"[!] No encontré un único libro que coincida con '{book}'. "
                        "Usa list_books para ver los nombres exactos.")
            allowed = bm["by_source"][source]

        # 1) Semántica
        emb = model.encode(_query_prefix(model_name) + query, normalize_embeddings=True).tolist()
        kwargs = {"query_embeddings": [emb],
                  "n_results": min(CANDIDATES, len(allowed) if allowed else bm["count"]),
                  "include": ["distances"]}
        if source:
            kwargs["where"] = {"source": source}
        sem_ids = (col.query(**kwargs).get("ids") or [[]])[0]
        sem_rank = [bm["pos"][i] for i in sem_ids if i in bm["pos"]]

        # 2) Léxica (BM25)
        lex_rank = bm["index"].search(query, CANDIDATES, allowed)

        # 3) Reciprocal Rank Fusion
        fused: dict[int, float] = defaultdict(float)
        for rank, idx in enumerate(sem_rank):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(lex_rank):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n]

        if not top:
            scope = f" en '{source}'" if source else ""
            return f"No se encontró información relevante{scope} para: '{query}'."

        sem_pos = {idx: r + 1 for r, idx in enumerate(sem_rank)}
        lex_pos = {idx: r + 1 for r, idx in enumerate(lex_rank)}
        out = [f"### Biblioteca técnica — {len(top)} resultados para: '{query}'\n"]
        for i, (idx, _) in enumerate(top, start=1):
            meta = bm["metas"][idx]
            src = meta.get("source", "desconocido")
            page = meta.get("page")
            loc = f"{src}, pág. {page}" if page else src
            match = []
            if idx in sem_pos:
                match.append(f"puesto {sem_pos[idx]} en búsqueda por significado")
            if idx in lex_pos:
                match.append(f"puesto {lex_pos[idx]} en búsqueda por palabras exactas")
            doc = bm["docs"][idx]
            if len(doc) > MAX_CHUNK_CHARS:
                doc = doc[:MAX_CHUNK_CHARS] + "…"
            out.append(f"**[{i}] {loc}** (ranking: {'; '.join(match)})\n{doc}\n\n---")
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
        _, _, model_name, bm = _get_backend()
        counts = {src: len(idxs) for src, idxs in bm["by_source"].items()}
        if not counts:
            return "La biblioteca está vacía."
        lines = [f"### {len(counts)} documentos indexados "
                 f"({sum(counts.values())} fragmentos, modelo {model_name})\n"]
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
