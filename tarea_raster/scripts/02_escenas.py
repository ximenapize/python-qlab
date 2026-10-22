"""Paso 3 · Buscar imágenes Sentinel-2 L2A y medir sus nubes en la zona del fuego.

Zona del fuego = focos VIIRS del distrito con un buffer de 2 km.
Para cada fecha disponible se lee la capa SCL a 200 m (rápido) y se calcula
el % de la zona cubierta por nubes, sombra de nubes o sin dato.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
from rasterio.features import geometry_mask

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"

TOP = ["RIO TAMBO", "NUEVA REQUENA", "MASISEA", "RAYMONDI", "TAHUANIA"]
FECHAS = "2026-09-01/2026-10-21"
BUFFER_M = 2000
# SCL: 0 sin dato, 3 sombra de nube, 8-9 nube, 10 cirros
SCL_MALO = [0, 3, 8, 9, 10]

catalogo = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

focos = gpd.read_file(OUT / "focos_peru_7d.gpkg")
filas = []
zonas = []
for distrito in TOP:
    g = focos[focos.NOMBDIST == distrito].to_crs(32718)
    zona = g.buffer(BUFFER_M).union_all()
    zonas.append({"NOMBDIST": distrito, "geometry": zona})
    bbox = gpd.GeoSeries([zona], crs=32718).to_crs(4326).total_bounds

    items = catalogo.search(collections=["sentinel-2-l2a"], bbox=bbox,
                            datetime=FECHAS).item_collection()
    scl = odc.stac.load(items, bands=["SCL"], bbox=bbox, crs="EPSG:32718",
                        resolution=200, groupby="solar_day", chunks={})
    dentro = ~geometry_mask([zona], out_shape=scl.SCL.shape[1:],
                            transform=scl.odc.geobox.affine)
    for t in range(scl.sizes["time"]):
        capa = scl.SCL.isel(time=t).values
        malo = np.isin(capa, SCL_MALO) & dentro
        filas.append({"NOMBDIST": distrito,
                      "fecha": str(scl.time.values[t])[:10],
                      "nubes_pct": 100 * malo.sum() / dentro.sum()})
    print(f"{distrito}: {len(items)} escenas")

escenas = pd.DataFrame(filas)
escenas.to_csv(OUT / "escenas_nubes.csv", index=False)
gpd.GeoDataFrame(zonas, crs=32718).to_file(OUT / "zonas_fuego.gpkg")

tabla = escenas.pivot(index="fecha", columns="NOMBDIST", values="nubes_pct")
print("\n% de la zona del fuego con nubes o sin dato\n")
print(tabla[TOP].round(0).to_string())
