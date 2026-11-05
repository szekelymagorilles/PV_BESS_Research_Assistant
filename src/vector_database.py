import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = 'BAAI/bge-m3'):
    """
    Inicializálja és visszaadja a HuggingFace beágyazó modellt.
    A BAAI/bge-m3 modell ideális, mert natívan támogatja a többnyelvű 
    (pl. angol cikk - magyar kérdés) összevetéseket.
    """
    print(f"Beágyazó modell betöltése: {model_name}...")
    # A LangChain automatikusan megpróbálja a GPU-t (CUDA) használni, ha elérhető
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def build_or_load_vector_db(chunks=None, persist_dir: str = './chroma_db', model_name: str = 'BAAI/bge-m3'):
    """
    Felépíti a Chroma vektoradatbázist a megadott dokumentum darabokból (chunks), 
    vagy betölti a meglévőt a lemezről, hogy ne kelljen újra beágyazni mindent.
    
    Paraméterek:
    - chunks: LangChain Document objektumok listája. Ha None, akkor megpróbál betölteni egy létezőt.
    - persist_dir: A mappa, ahová a Chroma elmenti a vektorokat.
    """
    embeddings = get_embedding_model(model_name)
    
    if chunks:
        print(f"Új vektoradatbázis építése {len(chunks)} chunkból, mentés ide: '{persist_dir}'...")
        # A from_documents létrehozza az adatbázist és el is menti a persist_dir-be
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )
        print("Vektoradatbázis sikeresen felépítve és elmentve!")
    else:
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"A vektoradatbázis nem található itt: '{persist_dir}'. Előbb építsd fel chunks megadásával!")
        
        print(f"Meglévő vektoradatbázis betöltése innen: '{persist_dir}'...")
        vector_db = Chroma(
            persist_directory=persist_dir, 
            embedding_function=embeddings
        )
        print("Vektoradatbázis sikeresen betöltve!")
        
    return vector_db

# Helyi teszteléshez (csak akkor fut le, ha ezt a fájlt közvetlenül indítod)
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    # 1. Készítünk egy teszt chunkot
    dummy_chunks = [
        Document(page_content="A BESS (Battery Energy Storage System) kulcsfontosságú a PV rendszerek ingadozásának kiegyenlítésében.", metadata={"source": "test.pdf"}),
        Document(page_content="Az MPC (Model Predictive Control) figyelembe veszi a hálózat feszültség kényszereit.", metadata={"source": "test.pdf"})
    ]
    
    # 2. Felépítjük az adatbázist (egy test mappába)
    print("\n--- Modul tesztelése ---")
    test_db = build_or_load_vector_db(chunks=dummy_chunks, persist_dir='./test_chroma_db')
    
    # 3. Keresés tesztelése
    query = "Mivel egyenlítik ki a napelemek ingadozását?"
    results = test_db.similarity_search(query, k=1)
    
    print(f"\nKérdés: {query}")
    print(f"Legjobb találat: {results[0].page_content}")