import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# 1. Ideiglenes Eszköz (Tool) a struktúra felépítéséhez
class PVSimulationInput(BaseModel):
    capacity_kw: float = Field(..., gt=0, description='Napelem kapacitás kW-ban')

@tool(args_schema=PVSimulationInput)
def simulate_pv(capacity_kw: float) -> str:
    """
    Napelemes termelés szimulátor. Konkrét teljesítmény és hozam számításhoz használható.
    """
    return f'Szimuláció kész: a(z) {capacity_kw} kW kapacitású rendszer elméleti napi hozama kiszámítva.'

# 2. Ágens inicializálása - A te beállításaid alapján!
def create_research_agent(model_name: str = 'qwen/qwen3.8-27b', temperature: float = 0.0, max_tokens: int = 900):
    """
    Létrehozza a Tool-Calling ágenst a Groq LLM használatával.
    A felhasználó által tesztelt stabil modellt (Qwen) használja.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Hiányzik a GROQ_API_KEY környezeti változó!")
        
    print(f"LLM (Groq) inicializálása: {model_name} (Max tokens: {max_tokens})...")
    llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)
    
    tools = [simulate_pv]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Szakértő kutatási asszisztens vagy a PV-BESS (Napelem és Akkumulátoros Energiatároló) rendszerek optimalizációjának területén. "
         "Válaszolj a felhasználó kérdéseire a megadott szakirodalmi kontextus alapján magyar nyelven. "
         "FONTOS: A kontextus nyers PDF kivonatokat tartalmaz, amelyekben ábrafeliratok, táblázat töredékek és indexszámok (pl. #1, Index Terms) is lehetnek. "
         "Ezeket a zajokat hagyd figyelmen kívül, és fogalmazz meg egy tiszta, mérnöki szempontból pontos választ! "
         "Ha a felhasználó számítást kér, használd a kódolt eszközeidet.\n\nKontextus:\n{context}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("Kutatási Ágens (AgentExecutor) sikeresen felépítve!")
    return agent_executor