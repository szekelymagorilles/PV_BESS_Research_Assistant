import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# IMPORTÁLJUK A VALÓS FIZIKAI SZIMULÁTORT
from src.pv_simulation import simulate_clearsky_pv_yield

# 1. Valódi Eszköz (Tool) bemeneti sémájának definiálása
class PVSimulationInput(BaseModel):
    capacity_kw: float = Field(..., gt=0, description='Napelem rendszer egyenáramú (DC) kapacitása kW-ban')
    month: int = Field(default=6, ge=1, le=12, description='Hónap száma (1-12), alapértelmezetten 6 (Június)')
    tilt: float = Field(default=35.0, description='Napelemek dőlésszöge fokban, alapértelmezetten 35')
    azimuth: float = Field(default=180.0, description='Tájolás fokban, 180 a Dél, alapértelmezetten 180')

# A LangChain @tool dekorátora köti össze a sémát és a függvényt az LLM-mel
@tool(args_schema=PVSimulationInput)
def simulate_pv_tool(capacity_kw: float, month: int = 6, tilt: float = 35.0, azimuth: float = 180.0) -> str:
    """
    Napelemes termelés fizikai szimulátora (pvlib). 
    Kiszámítja egy budapesti PV rendszer elméleti napi hozamát és csúcsteljesítményét az adott hónapra.
    Használd ezt, ha a felhasználó napelemes hozamszámítást kér!
    """
    try:
        results = simulate_clearsky_pv_yield(
            capacity_kw=capacity_kw, 
            month=month, 
            tilt=tilt, 
            azimuth=azimuth
        )
        return (f"Fizikai szimuláció sikeres (Helyszín: {results['location']}, Dátum: {results['date']}).\n"
                f"Rendszer kapacitása: {results['capacity_kw']} kWp\n"
                f"Napi energiatermelés (AC hozam): {results['daily_yield_kwh']} kWh\n"
                f"Napi csúcsteljesítmény (AC): {results['peak_power_kw']} kW")
    except Exception as e:
        return f"Hiba történt a szimuláció során: {str(e)}"

# 2. Ágens inicializálása
def create_research_agent(model_name: str = 'qwen/qwen3.8-27b', temperature: float = 0.0, max_tokens: int = 900):
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Hiányzik a GROQ_API_KEY környezeti változó!")
        
    print(f"LLM (Groq) inicializálása: {model_name} (Max tokens: {max_tokens})...")
    llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)
    
    tools = [simulate_pv_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Szakértő kutatási asszisztens vagy a PV-BESS rendszerek optimalizációjának területén. "
         "Válaszolj a felhasználó kérdéseire magyar nyelven. "
         "Ha a felhasználó napelemes teljesítményt vagy hozamot akar számolni, KÖTELEZŐEN használd a 'simulate_pv_tool' eszközt, "
         "és a kapott fizikai eredményeket foglald bele a válaszodba!\n\nKontextus:\n{context}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("Valós fizikai eszközzel felszerelt Ágens felépítve!")
    return agent_executor