"""
Prueba la búsqueda híbrida directamente, sin LM Studio ni el agente.

Uso (desde F:\\vertex-rag-library con el .venv activado):
    python test_search.py                    # preguntas de ejemplo
    python test_search.py "tu pregunta aquí" # una pregunta concreta
"""
import sys

from server_rag import list_books, search_technical_library

QUERIES = [
    "estado state remember mutableStateOf recomposition Jetpack Compose",
    "streaming stream=True chat.completions OpenAI API respuestas en tiempo real",
    "definir herramienta servidor MCP FastMCP @mcp.tool",
]

if __name__ == "__main__":
    print(list_books())
    for q in (sys.argv[1:] or QUERIES):
        print("\n" + "=" * 90)
        print(search_technical_library(q, n_results=4))
