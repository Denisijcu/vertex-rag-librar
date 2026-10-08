"""
Indexador de la biblioteca técnica de Vertex Coders -> ChromaDB.

- Incremental: solo reindexa archivos nuevos o modificados (hash SHA-256)
  y elimina de la DB los libros que ya no están en la carpeta.
- Reconstruye todo automáticamente si cambia el modelo de embeddings.
- Guarda la página de origen en los metadatos de cada fragmento.

Uso (desde F:\\vertex-rag-library con el .venv activado):
    python index_books.py            # incremental
    python index_books.py --rebuild  # borra la colección y reindexa todo
"""
import argparse
import hashlib
import logging
import os
import sys

# Silenciar ruido de HF / httpx antes de importar las librerías
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.ERROR)

from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuración (rutas absolutas basadas en la ubicación del script)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKS_DIR = os.path.join(BASE_DIR, "my_technical_library")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "vertex_technical_library"

# Modelo multilingüe (español + inglés), 384 dims, contexto de 512 tokens.
# Los modelos E5 requieren prefijos "passage: " / "query: ".
# El servidor lee el modelo desde los metadatos de la colección, así que
# si cambias esta línea basta con reindexar: el servidor se adapta solo.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
ENCODE_BATCH = 32
CHROMA_BATCH = 1000
SUPPORTED_EXT = (".pdf", ".md", ".txt")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_pages(path: str) -> list[tuple[int, str]]:
    """Devuelve [(num_pagina, texto)]. Para .md/.txt la página es 0."""
    if path.lower().endswith(".pdf"):
        pages = []
        reader = PdfReader(path)
        for idx, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception as e:  # PDFs con objetos rotos
                print(f"    [!] Pág {idx} ilegible: {e}")
                txt = ""
            if txt.strip():
                pages.append((idx, txt))
        return pages
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return [(0, txt)] if txt.strip() else []


def get_or_rebuild_collection(client: PersistentClient, force_rebuild: bool):
    existing = {c.name if hasattr(c, "name") else c for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        col = client.get_collection(COLLECTION_NAME)
        stored_model = (col.metadata or {}).get("embedding_model")
        if force_rebuild or stored_model != EMBEDDING_MODEL:
            reason = "--rebuild" if force_rebuild else f"modelo cambió ({stored_model} -> {EMBEDDING_MODEL})"
            print(f"[*] Reconstruyendo colección: {reason}")
            client.delete_collection(COLLECTION_NAME)
        else:
            return col
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


def indexed_sources(collection) -> dict[str, str]:
    """{archivo: hash} de lo que ya está en la DB."""
    data = collection.get(include=["metadatas"])
    out = {}
    for meta in data.get("metadatas") or []:
        if meta and "source" in meta:
            out[meta["source"]] = meta.get("file_hash", "")
    return out


def index_library(force_rebuild: bool = False) -> None:
    print(f"[*] Biblioteca: {BOOKS_DIR}")
    print(f"[*] ChromaDB:   {DB_DIR}")

    if not os.path.isdir(BOOKS_DIR):
        os.makedirs(BOOKS_DIR)
        print("[!] La carpeta no existía; se creó. Coloca tus libros ahí y vuelve a correr.")
        return

    files = sorted(f for f in os.listdir(BOOKS_DIR) if f.lower().endswith(SUPPORTED_EXT))
    if not files:
        print("[!] No hay archivos .pdf/.md/.txt en la biblioteca.")
        return

    client = PersistentClient(path=DB_DIR)
    collection = get_or_rebuild_collection(client, force_rebuild)
    already = indexed_sources(collection)

    # 1) Eliminar libros que ya no existen en la carpeta
    for gone in sorted(set(already) - set(files)):
        collection.delete(where={"source": gone})
        print(f"[-] Eliminado de la DB (ya no está en la carpeta): {gone}")

    # 2) Detectar qué hay que (re)indexar
    pending = []
    for filename in files:
        path = os.path.join(BOOKS_DIR, filename)
        digest = file_sha256(path)
        if already.get(filename) == digest:
            continue
        pending.append((filename, path, digest))

    if not pending:
        print(f"[+] Todo al día. Total en DB: {collection.count()} fragmentos.")
        return

    print(f"[*] Cargando modelo de embeddings ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    skipped_low_text = []
    for filename, path, digest in pending:
        print(f"[*] Procesando: {filename}")
        if filename in already:
            collection.delete(where={"source": filename})  # versión vieja

        try:
            pages = extract_pages(path)
        except Exception as e:
            print(f"    [!] No se pudo leer {filename}: {e}")
            continue

        ids, docs, metas = [], [], []
        for page_num, text in pages:
            for i, chunk in enumerate(splitter.split_text(text)):
                ids.append(f"{filename}::p{page_num}::c{i}")
                docs.append(chunk)
                metas.append({"source": filename, "page": page_num, "file_hash": digest})

        if not docs:
            print(f"    [!] Sin texto extraíble (¿PDF escaneado? necesita OCR).")
            skipped_low_text.append(filename)
            continue
        if len(docs) < 10 and filename.lower().endswith(".pdf"):
            skipped_low_text.append(filename)

        print(f"    -> {len(docs)} fragmentos, generando embeddings...")
        embeddings = model.encode(
            [PASSAGE_PREFIX + d for d in docs],
            batch_size=ENCODE_BATCH,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        for start in range(0, len(docs), CHROMA_BATCH):
            end = start + CHROMA_BATCH
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
            )
        print(f"[+] {filename} indexado.")

    print(f"\n[+] Indexación completa. Total en DB: {collection.count()} fragmentos.")
    if skipped_low_text:
        print("[!] Poco o ningún texto extraído (probablemente escaneados, considera OCR):")
        for f in skipped_low_text:
            print(f"    - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa la biblioteca técnica en ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="Borra la colección y reindexa todo.")
    args = parser.parse_args()
    try:
        index_library(force_rebuild=args.rebuild)
    except KeyboardInterrupt:
        print("\n[!] Cancelado. Lo ya indexado se conserva; vuelve a correr para continuar.")
        sys.exit(1)
