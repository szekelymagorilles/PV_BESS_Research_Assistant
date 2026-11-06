# src/bess_optimization.py
import pulp
import pandas as pd

def optimize_bess_schedule(pv_generation: list, load_demand: list, 
                           bess_capacity_kwh: float = 100.0, 
                           max_power_kw: float = 50.0, 
                           soc_min: float = 0.1, 
                           soc_max: float = 0.9, 
                           eta: float = 0.95) -> dict:
    """
    BESS optimális töltési/kisütési menetrendjének meghatározása lineáris programozással (MILP).
    Célfüggvény: A hálózatról vételezett energia minimalizálása (Peak Shaving) és a degradáció csökkentése.
    """
    hours = len(pv_generation)
    if hours != len(load_demand):
        raise ValueError("A PV és a Fogyasztási adatsorok hosszának egyeznie kell!")
    
    # 1. Probléma definiálása (Minimalizálás)
    prob = pulp.LpProblem("BESS_MPC_Optimization", pulp.LpMinimize)
    
    # 2. Döntési Változók (P_ch: Töltés, P_dis: Kisütés, SoC: Töltöttség, P_grid: Hálózati import)
    P_ch = [pulp.LpVariable(f"P_ch_{t}", lowBound=0, upBound=max_power_kw) for t in range(hours)]
    P_dis = [pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=max_power_kw) for t in range(hours)]
    SoC = [pulp.LpVariable(f"SoC_{t}", lowBound=soc_min * bess_capacity_kwh, upBound=soc_max * bess_capacity_kwh) for t in range(hours)]
    P_grid = [pulp.LpVariable(f"P_grid_{t}", lowBound=0) for t in range(hours)]
    
    # 3. Célfüggvény
    # Nagyon drága a hálózati energia vételezése (10.0 súly), 
    # és enyhén büntetjük az akkumulátor felesleges ciklizálását (degradáció) (0.1 súly)
    prob += pulp.lpSum([P_grid[t] * 10.0 + (P_ch[t] + P_dis[t]) * 0.1 for t in range(hours)])
    
    # 4. Kényszerek (Constraints) felírása
    for t in range(hours):
        # Teljesítmény egyenleg: PV + Hálózat + Kisütés == Fogyasztás + Töltés
        prob += pv_generation[t] + P_grid[t] + P_dis[t] == load_demand[t] + P_ch[t], f"Energy_Balance_{t}"
        
        # Akkumulátor dinamika (SoC frissítése)
        if t == 0:
            # Kezdeti SoC feltételezett 50%
            prob += SoC[t] == 0.5 * bess_capacity_kwh + (P_ch[t] * eta - P_dis[t] / eta), f"SoC_Update_{t}"
        else:
            prob += SoC[t] == SoC[t-1] + (P_ch[t] * eta - P_dis[t] / eta), f"SoC_Update_{t}"

    # 5. Optimalizáló (Solver) futtatása némán
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 6. Eredmények formázása
    results = []
    for t in range(hours):
        results.append({
            "Hour": t,
            "PV_kW": round(pv_generation[t], 2),
            "Load_kW": round(load_demand[t], 2),
            "Charge_kW": round(P_ch[t].varValue, 2),
            "Discharge_kW": round(P_dis[t].varValue, 2),
            "SoC_kWh": round(SoC[t].varValue, 2),
            "Grid_Import_kW": round(P_grid[t].varValue, 2)
        })
        
    df = pd.DataFrame(results)
    
    return {
        "status": pulp.LpStatus[prob.status],
        "total_grid_import_kwh": round(df['Grid_Import_kW'].sum(), 2),
        "summary": f"Optimalizáció sikeres. Összes hálózati import: {round(df['Grid_Import_kW'].sum(), 2)} kWh",
        "schedule": df
    }