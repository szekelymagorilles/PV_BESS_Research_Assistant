import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_split_documents(docs_dir: str = 'docs/', chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Betölti a PDF fájlokat a megadott mappából, és feldolgozható szövegdarabokra bontja.
    
    Paraméterek:
    - docs_dir: A PDF dokumentumokat tartalmazó mappa elérési útja.
    - chunk_size: Egy szövegdarab maximális karakterszáma.
    - chunk_overlap: A darabok közötti átfedés karakterszáma a kontextus megőrzéséhez.
    """
    # 1. Mappa ellenőrzése és létrehozása, ha nem létezik
    if not os.path.exists(docs_dir):
        print(f"FIGYELMEZTETÉS: A '{docs_dir}' mappa nem létezik. Létrehozás...")
        os.makedirs(docs_dir, exist_ok=True)
        
    print(f"Dokumentumok keresése a '{docs_dir}' mappában...")
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    
    # Biztonsági háló: Ha üres a mappa, egy minta dokumentumot adunk vissza a pipeline teszteléséhez
    if not documents:
        print("FIGYELMEZTETÉS: Nem találhatók PDF fájlok. Minta dokumentum generálása a RAG teszteléséhez...")
        documents = [Document(page_content='Ez egy alapértelmezett minta dokumentum a RAG pipeline inicializáláshoz. A BESS MPC irányítása során feszültség kényszereket alkalmazunk.')]
    else:
        print(f"Sikeres betöltés: {len(documents)} oldal feldolgozása következik.")
        
    # 2. Szövegfelosztó inicializálása
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # 3. Darabolás végrehajtása
    chunks = text_splitter.split_documents(documents)
    print(f"Feldolgozás kész: {len(chunks)} darab (chunk) jött létre.")
    
    return chunks

# Helyi teszteléshez (csak akkor fut le, ha közvetlenül ezt a szkriptet indítod el)
if __name__ == "__main__":
    test_chunks = load_and_split_documents()
    if test_chunks:
        print(f"Első chunk minta: {test_chunks[0].page_content[:100]}...")