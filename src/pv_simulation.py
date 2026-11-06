# src/pv_simulation.py
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

def simulate_clearsky_pv_yield(capacity_kw: float, month: int = 6, tilt: float = 35, azimuth: float = 180) -> dict:
    """
    Fizikai PV szimuláció a pvlib segítségével egy adott hónap 15. napjára (tiszta égbolt modellel).
    
    Paraméterek:
    - capacity_kw: A rendszer névleges egyenáramú (DC) kapacitása (kW)
    - month: Hónap száma (1-12), alapértelmezetten június (6)
    - tilt: Napelemek dőlésszöge fokban (alapértelmezett: 35°)
    - azimuth: Tájolás fokban, 180° a tiszta Dél (alapértelmezett: 180°)
    """
    
    # 1. Helyszín definiálása (Budapest)
    lat, lon = 47.4979, 19.0402
    tz = 'Europe/Budapest'
    site = Location(lat, lon, tz=tz, altitude=102, name='Budapest')

    # 2. Időablak generálása (Adott hónap közepe, 1 órás felbontás)
    times = pd.date_range(f'2026-{month:02d}-15 00:00', f'2026-{month:02d}-15 23:59', freq='h', tz=tz)
    
    # 3. Tiszta égbolt (Clear Sky) besugárzási adatok generálása
    clearsky = site.get_clearsky(times)
    weather = pd.DataFrame({
        'ghi': clearsky['ghi'],
        'dni': clearsky['dni'],
        'dhi': clearsky['dhi'],
        'temp_air': 25.0,  # Állandó 25°C levegőhőmérséklet feltételezése
        'wind_speed': 1.0  # 1 m/s szélsebesség
    })

    # 4. Napelem rendszer fizikai paraméterei (PVWatts modell)
    mount = FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)
    
    # Szabványos üveg-üveg panel hőmérsékleti modell
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    array = Array(
        mount=mount, 
        module_parameters=dict(pdc0=capacity_kw * 1000, gamma_pdc=-0.004), # W-ban adjuk meg
        temperature_model_parameters=temp_params
    )
    
    # Inverter hatásfok 96%
    system = PVSystem(arrays=[array], inverter_parameters=dict(pdc0=capacity_kw * 1000, eta_inv_nom=0.96))
    
    # 5. Szimulációs lánc felépítése és futtatása
    mc = ModelChain(system, site, aoi_model='physical', spectral_model='no_loss')
    mc.run_model(weather)
    
    # AC kimenet aggregálása (Wh -> kWh)
    daily_yield_kwh = mc.results.ac.sum() / 1000
    peak_power_kw = mc.results.ac.max() / 1000
    
    return {
        "location": site.name,
        "date": f"2026-{month:02d}-15",
        "capacity_kw": capacity_kw,
        "daily_yield_kwh": round(daily_yield_kwh, 2),
        "peak_power_kw": round(peak_power_kw, 2)
    }