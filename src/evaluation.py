import os
import json
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig
from src.vector_database import build_or_load_vector_db, get_embedding_model
from src.retriever import get_advanced_retriever
from src.agent import create_research_agent
from langchain_groq import ChatGroq

def run_evaluation(json_path: str = 'data/test_questions.json'):
    print("Teszthalmaz betöltése...")
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    vector_db = build_or_load_vector_db(persist_dir='./chroma_db')
    advanced_retriever = get_advanced_retriever(vector_db, search_k=10, top_n=2)
    agent_executor = create_research_agent(model_name='qwen/qwen3.8-27b')
    
    ragas_llm = ChatGroq(model='qwen/qwen3.8-27b', temperature=0, max_tokens=950)
    ragas_embeddings = get_embedding_model('BAAI/bge-m3')

    all_results = []
    
    print(f"\nQWEN Kiértékelés indítása {len(test_data)} kérdésre (Kizárólag Faithfulness)...")

    for idx, item in enumerate(test_data):
        print(f"[{idx+1}/{len(test_data)}] Kérdés: {item['question']}")
        
        q = item['question']
        docs = advanced_retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        
        short_q = q + " Kérlek, maximum 2 mondatban válaszolj!"
        resp = agent_executor.invoke({'input': short_q, 'context': '\n'.join(ctx)})
        
        # Ground truth már nem is kell a hallucináció méréséhez
        eval_dataset = Dataset.from_dict({
            'question': [q],
            'answer': [resp['output']],
            'contexts': [ctx]
        })

        run_config = RunConfig(timeout=120, max_workers=1)

        try:
            res = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config
            )
            df_res = res.to_pandas()
            all_results.append(df_res)
            
            if 'faithfulness' in df_res.columns:
                print(f"-> Eredmény | Faithfulness (Hűség): {df_res['faithfulness'][0]:.2f}\n")
            else:
                print("-> NaN eredmény, de a script folytatódik.\n")
                
        except Exception as e:
            print(f"-> Hiba ennél a kérdésnél: {e}\n")

        if idx < len(test_data) - 1:
            print("--- Szünet: 65 másodperc várakozás a token limit miatt... ---")
            time.sleep(65)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        os.makedirs('results', exist_ok=True)
        final_df.to_csv('results/ragas_results.csv', index=False)
        
        print("\n===============================")
        print("VÉGSŐ OBJEKTÍV EREDMÉNYEK (QWEN)")
        print("===============================")
        cols = [c for c in ['question', 'faithfulness'] if c in final_df.columns]
        print(final_df[cols])
        return final_df
    else:
        return pd.DataFrame()
