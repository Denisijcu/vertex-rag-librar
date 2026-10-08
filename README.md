# 🧠 Vertex RAG Library — Intelligent Technical Assistant

Servidor **Model Context Protocol (MCP)** con recuperación aumentada por recuperación (**RAG**) local. Diseñado para conectar modelos locales (vía LM Studio o clientes MCP) con una biblioteca técnica privada de libros (programación, IA, hacking ético, electrónica).

## 🚀 Arquitectura
- **Motor Vectorial:** ChromaDB (Persistente local).
- **Embeddings:** `all-MiniLM-L6-v2` vía `sentence-transformers` (100% local, sin consumo de APIs externas).
- **Procesamiento:** Soporte multiformato (`.pdf`, `.md`, `.txt`) con división inteligente de fragmentos (`RecursiveCharacterTextSplitter`).
- **Protocolo:** Model Context Protocol (MCP) estándar.

## 🛠️ Instalación y Despliegue

1. **Clonar e inicializar entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\Activate