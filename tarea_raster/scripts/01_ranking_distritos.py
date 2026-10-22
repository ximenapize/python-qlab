"""Paso 2 · Ranking de distritos por focos VIIRS de los últimos 7 días.

Une los tres satélites VIIRS (S-NPP, NOAA-20, NOAA-21), recorta al Perú,
quita los focos de baja confianza y cuenta por distrito:
  - focos      : detecciones (un mismo fuego puede salir en los 3 satélites)
  - huella_ha  : celdas únicas de ~375 m con fuego (evita contar 3 veces)
  - frp_mw     : potencia radiativa total (intensidad)
Luego elige el distrito con mayor huella en cada departamento.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
OUT = RAIZ / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CELDA = 375 / 111_320  # grados ~ 375 m (aprox. válido cerca del ecuador)
HA_CELDA = 0.375 * 0.375 * 100  # 14,06 ha por píxel VIIRS

# 1 · focos de los tres satélites
csvs = sorted(RAW.glob("*_VIIRS_C2_South_America_7d.csv"))
focos = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
focos = focos[focos["confidence"].isin(["nominal", "high"])]

# 2 · límites distritales y cruce espacial
dist = gpd.read_file(RAW / "peru_distrital.geojson")
pts = gpd.GeoDataFrame(
    focos, geometry=gpd.points_from_xy(focos.longitude, focos.latitude), crs=4326
)
pts = pts.cx[-81.5:-68.5, -18.5:0.1]  # caja del Perú antes del cruce
pts = gpd.sjoin(pts, dist[["IDDIST", "NOMBDEP", "NOMBPROV", "NOMBDIST", "geometry"]],
                predicate="within", how="inner")

# 3 · huella: celdas únicas de ~375 m
pts["celda"] = (np.floor(pts.longitude / CELDA).astype(int).astype(str) + "_"
                + np.floor(pts.latitude / CELDA).astype(int).astype(str))

tabla = (pts.groupby(["IDDIST", "NOMBDEP", "NOMBPROV", "NOMBDIST"])
         .agg(focos=("celda", "size"), celdas=("celda", "nunique"),
              frp_mw=("frp", "sum"), dias=("acq_date", "nunique"),
              ultimo=("acq_date", "max"))
         .reset_index())
tabla["huella_ha"] = tabla.celdas * HA_CELDA
tabla = tabla.sort_values("huella_ha", ascending=False)

# 4 · top de distritos por huella (desempate: FRP)
tabla = tabla.sort_values(["huella_ha", "frp_mw"], ascending=False).reset_index(drop=True)
tabla.index += 1

print(f"Fechas: {pts.acq_date.min()} a {pts.acq_date.max()} · "
      f"{len(pts):,} focos en el Perú · {tabla.shape[0]} distritos con fuego\n")
cols = ["NOMBDEP", "NOMBPROV", "NOMBDIST", "huella_ha", "focos", "frp_mw", "dias", "ultimo"]
print(tabla[cols].head(15).round(0).to_string())

tabla.to_csv(OUT / "distritos_focos_7d.csv", index_label="rank")
pts.drop(columns="index_right").to_file(OUT / "focos_peru_7d.gpkg")
