# src/retriever.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_advanced_retriever(vector_db, search_k: int = 15, top_n: int = 3, model_name: str = 'BAAI/bge-reranker-base'):
    """
    Létrehoz egy kétlépcsős (kibővített) visszakeresőt.
    1. lépés: Alap vektoros keresés (Chroma) - lekéri a top 'search_k' (pl. 15) dokumentumot.
    2. lépés: Cross-Encoder újraértékelés - páronként pontozza a kérdést és a 15 chunkot,
              majd visszaadja a 'top_n' (pl. 3) legrelevánsabbat az LLM számára.
    """
    print(f"Alap visszakereső (Base Retriever) beállítása. Kezdeti találatok száma: {search_k}...")
    base_retriever = vector_db.as_retriever(search_kwargs={'k': search_k})
    
    print(f"Cross-Encoder Reranker modell betöltése: {model_name}...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    
    # Reranker kompresszor inicializálása
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    
    # A két lépés összekötése egyetlen objektummá
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    
    print("Kétlépcsős RAG visszakereső (Advanced Retriever) sikeresen felépítve!")
    return advanced_retriever