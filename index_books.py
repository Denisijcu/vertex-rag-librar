import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BOOKS_DIR = os.path.abspath("./my_technical_library")
DB_DIR = os.path.abspath("./chroma_db")

def index_library():
    print(f"[*] Buscando libros en: {BOOKS_DIR}")
    print(f"[*] Base de datos ChromaDB en: {DB_DIR}")
    
    if not os.path.exists(BOOKS_DIR):
        os.makedirs(BOOKS_DIR)
        print(f"[!] La carpeta no existía, se ha creado. Coloca tus archivos ahí.")
        return

    files = [f for f in os.listdir(BOOKS_DIR) if f.lower().endswith(('.pdf', '.md', '.txt'))]
    if not files:
        print(f"[!] No se encontraron archivos soportados en {BOOKS_DIR}")
        return

    print(f"[*] Inicializando modelo de embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    client = PersistentClient(path=DB_DIR)
    # Borrar colección anterior para evitar duplicados o estados corruptos
    try:
        client.delete_collection(name="vertex_technical_library")
    except:
        pass
    
    collection = client.create_collection(name="vertex_technical_library")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    for filename in files:
        file_path = os.path.join(BOOKS_DIR, filename)
        print(f"[*] Procesando: {filename}...")
        
        full_text = ""
        if filename.endswith('.pdf'):
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt:
                    full_text += f"\n[Fuente: {filename} - Pág {idx+1}]\n{txt}"
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()

        if not full_text.strip():
            print(f"[!] El archivo {filename} no tiene texto extraíble.")
            continue

        chunks = text_splitter.split_text(full_text)
        print(f"[*] Generando embeddings para {len(chunks)} fragmentos de {filename}...")

        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        embeddings = model.encode(chunks).tolist()
        metadatas = [{"source": filename} for _ in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"[+] ¡{filename} indexado con éxito!")

    print(f"[+] Indexación completa. Total en DB: {collection.count()} fragmentos.")

if __name__ == "__main__":
    index_library()