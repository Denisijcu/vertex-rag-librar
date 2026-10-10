"""
Diagnóstico rápido: cuánto texto extrae pypdf de cada libro y si contiene
ciertas palabras clave. No usa el modelo ni ChromaDB, es rápido.

Uso (desde F:\\vertex-rag-library con el .venv activado):
    python diag_pdfs.py
"""
import os

from index_books import BOOKS_DIR, SUPPORTED_EXT, extract_pages

KEYWORDS = ["mutableStateOf", "remember", "stream=True", "FastMCP", "Composable"]

for name in sorted(os.listdir(BOOKS_DIR)):
    if not name.lower().endswith(SUPPORTED_EXT):
        continue
    try:
        pages = extract_pages(os.path.join(BOOKS_DIR, name))
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        continue
    text = "\n".join(t for _, t in pages)
    found = [k for k in KEYWORDS if k.lower() in text.lower()]
    flag = "  <-- POCO TEXTO" if len(text) < 20000 else ""
    print(f"{len(pages):5d} pág | {len(text):9,d} chars | {', '.join(found) or '-':45s} | {name}{flag}")
