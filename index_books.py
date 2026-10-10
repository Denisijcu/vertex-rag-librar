"""
Indexador de la biblioteca técnica de Vertex Coders -> ChromaDB.

- Incremental: solo reindexa archivos nuevos o modificados (hash SHA-256)
  y elimina de la DB los libros que ya no están en la carpeta.
- Reconstruye todo automáticamente si cambia el modelo de embeddings
  o la versión del pipeline de limpieza/corte (PIPELINE_VERSION).
- Limpia el texto extraído (ligaduras NFKC, espacios múltiples de pypdf).
- Corta el documento completo (no página a página), así un bloque de código
  que cruza de página no queda partido; cada fragmento guarda su página.
- Soporta .pdf, .md, .txt y .epub (sin dependencias extra).

Uso (desde F:\\vertex-rag-library con el .venv activado):
    python index_books.py            # incremental
    python index_books.py --rebuild  # borra la colección y reindexa todo
"""
import argparse
import bisect
import hashlib
import html
import logging
import os
import posixpath
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

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

# Modelo multilingüe (español + inglés), 768 dims, contexto de 512 tokens.
# 'base' recupera mucho mejor que 'small' cuando la pregunta está en español
# y el libro en inglés. Los modelos E5 requieren prefijos "passage: "/"query: ".
# El servidor lee el modelo desde los metadatos de la colección.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "

# Si cambias la limpieza o el corte, sube este número: fuerza reindexado total.
PIPELINE_VERSION = "4"

CHUNK_SIZE = 1000      # ~250-300 tokens, cabe de sobra en los 512 de E5
CHUNK_OVERLAP = 200
ENCODE_BATCH = 16
CHROMA_BATCH = 1000
SUPPORTED_EXT = (".pdf", ".md", ".txt", ".epub")


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------
_MULTI_SPACE = re.compile(r"[ \t\u00a0\u2000-\u200b]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """NFKC (ﬁ -> fi, etc.), quita guiones de corte, colapsa espacios."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # pala-\nbra -> palabra
    lines = [_MULTI_SPACE.sub(" ", ln).strip() for ln in text.split("\n")]

    # Algunos PDFs (exportados desde Word) salen de pypdf con UNA palabra por
    # línea. Si la mayoría de las líneas no tienen espacios, se unen las líneas
    # simples con espacio y solo se conservan los saltos de párrafo (línea vacía).
    non_empty = [ln for ln in lines if ln]
    if non_empty:
        one_word = sum(1 for ln in non_empty if " " not in ln)
        if one_word / len(non_empty) > 0.6:
            joined = "\n".join(lines)
            empties = len(lines) - len(non_empty)
            if empties >= 0.5 * len(non_empty):
                # Patrón "palabra\n\npalabra": las líneas vacías no son párrafos,
                # son separadores entre palabras -> se une todo con espacios.
                joined = re.sub(r"\n+", " ", joined)
            else:
                joined = re.sub(r"(?<!\n)\n(?!\n)", " ", joined)
            lines = [_MULTI_SPACE.sub(" ", ln).strip() for ln in joined.split("\n")]

    text = "\n".join(lines)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------
class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "pre", "section", "article", "blockquote", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return html.unescape("".join(self.parts))


def _epub_pages(path: str) -> list[tuple[int, str]]:
    """Capítulos del EPUB en orden de lectura (spine). 'página' = nº de capítulo."""
    pages = []
    with zipfile.ZipFile(path) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile").attrib["full-path"]
        opf_dir = posixpath.dirname(rootfile)
        opf = ET.fromstring(zf.read(rootfile))
        manifest = {it.attrib["id"]: it.attrib["href"]
                    for it in opf.findall(".//{*}manifest/{*}item")}
        spine = [it.attrib["idref"] for it in opf.findall(".//{*}spine/{*}itemref")]
        for num, idref in enumerate(spine, start=1):
            href = manifest.get(idref)
            if not href:
                continue
            full = posixpath.normpath(posixpath.join(opf_dir, href.split("#")[0]))
            try:
                raw = zf.read(full).decode("utf-8", errors="ignore")
            except KeyError:
                continue
            parser = _HTMLText()
            parser.feed(raw)
            txt = parser.text()
            if txt.strip():
                pages.append((num, txt))
    return pages


def extract_pages(path: str) -> list[tuple[int, str]]:
    """Devuelve [(num_pagina, texto_limpio)]. Para .md/.txt la página es 0."""
    low = path.lower()
    if low.endswith(".pdf"):
        raw_pages = []
        reader = PdfReader(path)
        for idx, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception as e:  # PDFs con objetos rotos
                print(f"    [!] Pág {idx} ilegible: {e}")
                txt = ""
            raw_pages.append((idx, txt))
    elif low.endswith(".epub"):
        raw_pages = _epub_pages(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_pages = [(0, f.read())]

    out = []
    for num, txt in raw_pages:
        txt = clean_text(txt)
        if txt:
            out.append((num, txt))
    return out


def chunk_document(pages: list[tuple[int, str]], splitter) -> list[tuple[int, str]]:
    """Une todas las páginas y corta el documento completo.
    Devuelve [(página_donde_empieza, fragmento)]."""
    starts, nums, buf, pos = [], [], [], 0
    for num, txt in pages:
        starts.append(pos)
        nums.append(num)
        buf.append(txt)
        pos += len(txt) + 2  # "\n\n" separador
    full = "\n\n".join(buf)

    chunks = []
    for doc in splitter.create_documents([full]):
        start = doc.metadata.get("start_index", 0)
        i = max(0, bisect.bisect_right(starts, start) - 1)
        chunks.append((nums[i], doc.page_content))
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_or_rebuild_collection(client: PersistentClient, force_rebuild: bool):
    existing = {c.name if hasattr(c, "name") else c for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        col = client.get_collection(COLLECTION_NAME)
        meta = col.metadata or {}
        stored_model = meta.get("embedding_model")
        stored_pipe = meta.get("pipeline_version")
        if force_rebuild or stored_model != EMBEDDING_MODEL or stored_pipe != PIPELINE_VERSION:
            if force_rebuild:
                reason = "--rebuild"
            elif stored_model != EMBEDDING_MODEL:
                reason = f"modelo cambió ({stored_model} -> {EMBEDDING_MODEL})"
            else:
                reason = f"pipeline cambió ({stored_pipe} -> {PIPELINE_VERSION})"
            print(f"[*] Reconstruyendo colección: {reason}")
            client.delete_collection(COLLECTION_NAME)
        else:
            return col
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine",
                  "embedding_model": EMBEDDING_MODEL,
                  "pipeline_version": PIPELINE_VERSION},
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
        print("[!] No hay archivos .pdf/.md/.txt/.epub en la biblioteca.")
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    skipped_low_text = []
    for n, (filename, path, digest) in enumerate(pending, start=1):
        print(f"[*] ({n}/{len(pending)}) Procesando: {filename}")
        if filename in already:
            collection.delete(where={"source": filename})  # versión vieja

        try:
            pages = extract_pages(path)
        except Exception as e:
            print(f"    [!] No se pudo leer {filename}: {e}")
            continue

        chunks = chunk_document(pages, splitter) if pages else []
        if not chunks:
            print("    [!] Sin texto extraíble (¿PDF escaneado? necesita OCR).")
            skipped_low_text.append(filename)
            continue
        if len(chunks) < 10 and filename.lower().endswith(".pdf"):
            skipped_low_text.append(filename)

        ids = [f"{filename}::c{i}" for i in range(len(chunks))]
        docs = [c for _, c in chunks]
        metas = [{"source": filename, "page": p, "file_hash": digest} for p, _ in chunks]

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
