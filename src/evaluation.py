import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from src.vector_database import build_or_load_vector_db, get_embedding_model
from src.retriever import get_advanced_retriever
from src.agent import create_research_agent
from langchain_groq import ChatGroq

def run_evaluation(json_path: str = 'data/test_questions.json'):
    """Lefuttatja a RAGAS kiértékelést a megadott tesztkérdéseken."""
    print("Teszthalmaz betöltése...")
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 1. Inicializálása
    vector_db = build_or_load_vector_db(persist_dir='./chroma_db')
    advanced_retriever = get_advanced_retriever(vector_db, search_k=15, top_n=3)
    agent_executor = create_research_agent(model_name='qwen/qwen3.8-27b')
    
    # LLM-et és a beágyazó létrehozása
    ragas_llm = ChatGroq(model='qwen/qwen3.8-27b', temperature=0, max_tokens=900)
    ragas_embeddings = get_embedding_model('BAAI/bge-m3')

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # 2. Válaszok generálása
    print(f"\nVálaszok generálása {len(test_data)} kérdésre...")
    for item in test_data:
        q = item['question']
        
        # Kontextus kinyerése a PDF-ekből
        docs = advanced_retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        
        # LLM válasz generálása
        resp = agent_executor.invoke({'input': q, 'context': '\n'.join(ctx)})
        
        questions.append(q)
        answers.append(resp['output'])
        contexts.append(ctx)
        ground_truths.append(item['ground_truth'])

    # 3. Formázás a RAGAS számára (HuggingFace Dataset)
    eval_dict = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    eval_dataset = Dataset.from_dict(eval_dict)

    # 4. Kiértékelés végrehajtása
    print("\nRAGAS metrikák számítása (Faithfulness, Answer Relevancy)...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    # 5. Eredmények mentése és megjelenítése
    df_results = results.to_pandas()
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/ragas_results.csv', index=False)
    
    print("\n--- Objektív Kiértékelés Eredményei ---")
    print(df_results[['question', 'faithfulness', 'answer_relevancy']])
    print("\nEredmények elmentve a 'results/ragas_results.csv' fájlba.")

    return df_results