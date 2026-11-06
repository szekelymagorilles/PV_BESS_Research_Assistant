import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# 1. Valós fizikai és matematikai eszközök importálása
from src.pv_simulation import simulate_clearsky_pv_yield
from src.bess_optimization import optimize_bess_schedule

# --- TOOL 1: PV Szimuláció ---
class PVSimulationInput(BaseModel):
    capacity_kw: float = Field(..., gt=0, description='Napelem rendszer kapacitása kW-ban')
    month: int = Field(default=6, ge=1, le=12, description='Hónap száma (1-12)')
    tilt: float = Field(default=35.0, description='Napelemek dőlésszöge')
    azimuth: float = Field(default=180.0, description='Tájolás fokban')

@tool(args_schema=PVSimulationInput)
def simulate_pv_tool(capacity_kw: float, month: int = 6, tilt: float = 35.0, azimuth: float = 180.0) -> str:
    """Napelemes termelés fizikai szimulátora (pvlib). Használd napelemes hozamszámításhoz!"""
    try:
        results = simulate_clearsky_pv_yield(capacity_kw, month, tilt, azimuth)
        return (f"PV Szimuláció kész (Hónap: {month}). "
                f"Napi AC hozam: {results['daily_yield_kwh']} kWh. "
                f"Csúcsteljesítmény: {results['peak_power_kw']} kW.")
    except Exception as e:
        return f"Hiba a PV szimuláció során: {str(e)}"

# --- TOOL 2: BESS Optimalizáció ---
class BessOptimizationInput(BaseModel):
    pv_generation: list[float] = Field(..., description="PV termelési adatsor listaként (kW)")
    load_demand: list[float] = Field(..., description="Fogyasztási adatsor listaként (kW)")
    bess_capacity_kwh: float = Field(default=100.0, description="Akkumulátor kapacitása (kWh)")
    max_power_kw: float = Field(default=50.0, description="Inverter max teljesítménye (kW)")

@tool(args_schema=BessOptimizationInput)
def optimize_bess_tool(pv_generation: list[float], load_demand: list[float], bess_capacity_kwh: float = 100.0, max_power_kw: float = 50.0) -> str:
    """Akkumulátor (BESS) optimális töltési/kisütési menetrendjének kiszámítása MILP alapon."""
    try:
        res = optimize_bess_schedule(pv_generation, load_demand, bess_capacity_kwh, max_power_kw)
        return f"BESS Optimalizáció sikeres (Státusz: {res['status']}). A hálózatról vételezett energia minimalizálva: {res['total_grid_import_kwh']} kWh."
    except Exception as e:
        return f"Hiba az optimalizáció során: {str(e)}"

# --- ÁGENS INICIALIZÁLÁSA ---
def create_research_agent(model_name: str = 'qwen/qwen3.8-27b', temperature: float = 0.0, max_tokens: int = 1500):
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Hiányzik a GROQ_API_KEY!")
        
    llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)
    
    # A modell most már KÉT eszközt kap meg!
    tools = [simulate_pv_tool, optimize_bess_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Szakértő kutatási asszisztens vagy a PV-BESS rendszerek területén. "
         "Ha a felhasználó fizikai számítást kér, használd a megfelelő eszközöket (simulate_pv_tool vagy optimize_bess_tool), "
         "és az eredményeiket foglald bele a válaszodba. Válaszolj magyarul!\n\nKontextus:\n{context}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
