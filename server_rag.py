import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# Inicializar servidor MCP
mcp = FastMCP("vertex-rag-library")

# BLINDAJE DE RUTAS: Usar ruta absoluta basada en la ubicación de este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print(f"[*] Conectando a ChromaDB en la ruta absoluta: {DB_DIR}")

# Cargar modelo de embeddings y base de datos vectorial
print("[*] Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

client = PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="vertex_technical_library")

@mcp.tool()
def search_technical_library(query: str, n_results: int = 3) -> str:
    """
    Busca información técnica, fragmentos de código, esquemas o conceptos en la biblioteca privada de libros (programación, IA, hacking, electrónica).
    Utiliza esta herramienta cuando necesites contrastar información técnica compleja o consultar manuales específicos.
    
    Args:
        query: Concepto, pregunta técnica, función o fragmento a buscar en los libros.
        n_results: Número de fragmentos relevantes a recuperar (por defecto 3).
    """
    try:
        # Verificar si la base de datos tiene fragmentos indexados
        total_docs = collection.count()
        if total_docs == 0:
            return "[!] Advertencia: La base de datos vectorial está vacía. Debes ejecutar 'index_books.py' primero."

        # Generar embedding de la consulta del modelo
        query_embedding = model.encode(query).tolist()
        
        # Ajustar n_results si hay menos documentos guardados que el límite pedido
        actual_n = min(n_results, total_docs)

        # Buscar en ChromaDB los fragmentos más cercanos semánticamente
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n
        )
        
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        
        if not documents or not documents[0]:
            return f"No se encontró información relevante para la consulta: '{query}'."
        
        response_text = f"### Resultados extraídos de la Biblioteca Técnica (Total en DB: {total_docs} fragmentos):\n\n"
        
        for i, (doc, meta) in enumerate(zip(documents[0], metadatas[0])):
            source = meta.get("source", "Desconocido") if meta else "Desconocido"
            response_text += f"**[Referencia {i+1} - Libro: {source}]**\n{doc}\n\n---\n"
            
        return response_text

    except Exception as e:
        return f"Error crítico consultando la base de datos vectorial: {str(e)}"

if __name__ == "__main__":
    mcp.run()