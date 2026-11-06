import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from ragas.run_config import RunConfig  # <-- ÚJ IMPORT
from src.vector_database import build_or_load_vector_db, get_embedding_model
from src.retriever import get_advanced_retriever
from src.agent import create_research_agent
from langchain_groq import ChatGroq

def run_evaluation(json_path: str = 'data/test_questions.json'):
    """Lefuttatja a RAGAS kiértékelést a megadott tesztkérdéseken."""
    print("Teszthalmaz betöltése...")
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    vector_db = build_or_load_vector_db(persist_dir='./chroma_db')
    advanced_retriever = get_advanced_retriever(vector_db, search_k=15, top_n=3)
    agent_executor = create_research_agent(model_name='qwen/qwen3.8-27b')
    
    # LLM és Beágyazó a bírónak
    ragas_llm = ChatGroq(model='qwen/qwen3.8-27b', temperature=0, max_tokens=2048)
    ragas_embeddings = get_embedding_model('BAAI/bge-m3')

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"\nVálaszok generálása {len(test_data)} kérdésre...")
    for item in test_data:
        q = item['question']
        docs = advanced_retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        
        resp = agent_executor.invoke({'input': q, 'context': '\n'.join(ctx)})
        
        questions.append(q)
        answers.append(resp['output'])
        contexts.append(ctx)
        ground_truths.append(item['ground_truth'])

    eval_dataset = Dataset.from_dict({
        'question': questions, 'answer': answers,
        'contexts': contexts, 'ground_truth': ground_truths
    })

    print("\nRAGAS metrikák számítása (Szekvenciális végrehajtás a limitek miatt)...")
    # <-- ÚJ: Sebességkorlátozás és timeout beállítása
    run_config = RunConfig(timeout=180, max_workers=1, max_retries=5)

    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config  # <-- ÚJ PARAMÉTER
    )

    df_results = results.to_pandas()
    
    # Biztonsági ellenőrzés, ha a Groq még így is blokkolna
    if df_results.empty or 'question' not in df_results.columns:
        print("\nHIBA: A Groq API elutasította a RAGAS kéréseit (Valószínűleg Token Limit miatt).")
        print("Próbálj meg várni 1-2 percet az újraindítás előtt!")
        return df_results

    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/ragas_results.csv', index=False)
    
    print("\n--- Objektív Kiértékelés Eredményei ---")
    print(df_results[['question', 'faithfulness', 'answer_correctness']])
    return df_results
