"""Paso 8 · ¿Qué se quemó? (WorldCover) y exposición de edificaciones.

Uso:  python scripts/07_cobertura_indice.py "RIO TAMBO"

Para cada parte del distrito:
  - quemado = dNBR >= 0,27 y a <= 1 km de un foco (archivo *_sembrado.tif)
  - cobertura = ESA WorldCover 2021, remuestreada a la grilla del dNBR (20 m)
Por edificación:
  - riesgo por distancia al píxel quemado más cercano:
      muy alto <= 100 m · alto 100–500 m · medio 500 m–1 km · bajo > 1 km
  - cobertura dominante y % de vegetación (bosque, arbustal, pastizal) en 100 m
La vegetación no entra al riesgo: en la selva casi todo es "bosque" y
saturaba el índice (80 % en clase alta en Río Tambo).
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from odc.geo.geobox import GeoBox
from scipy import ndimage

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"

QUEMADO = 0.27
RADIO_M = 100
WORLDCOVER = {10: "bosque", 20: "arbustal", 30: "pastizal", 40: "cultivo",
              50: "urbano", 60: "suelo desnudo", 70: "nieve", 80: "agua",
              90: "humedal", 95: "manglar", 100: "musgo"}
VEGETACION = [10, 20, 30]
RIESGO_CORTES = [-1, 100, 500, 1000, np.inf]
RIESGO = ["muy alto", "alto", "medio", "bajo"]

catalogo = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


def worldcover(geobox, bbox):
    items = catalogo.search(collections=["esa-worldcover"], bbox=bbox,
                            datetime="2021").item_collection()
    ds = odc.stac.load(items, bands=["map"], geobox=geobox, resampling="mode")
    return ds["map"].max("time").values


def main(distrito: str):
    nombre = distrito.lower().replace(" ", "_")
    carpeta = OUT / "dnbr" / nombre
    edif = gpd.read_parquet(OUT / f"edificaciones_{nombre}.parquet").to_crs(32718)
    cent = edif.centroid
    edif["dist_quemado_m"] = np.inf
    edif["pct_vegetacion_100m"] = np.nan
    edif["cobertura_100m"] = None

    hectareas = {}
    for tif in sorted(carpeta.glob("parte_??_sembrado.tif")):
        with rasterio.open(tif) as src:
            d = src.read(1)
            geobox = GeoBox(d.shape, src.transform, src.crs)
            bbox = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            res = src.res[0]
            ha = res * res / 10_000
            transform = src.transform
        wc = worldcover(geobox, bbox)

        quemado = d >= QUEMADO
        for codigo in np.unique(wc[quemado]):
            clase = WORLDCOVER.get(int(codigo), "otro")
            hectareas[clase] = hectareas.get(clase, 0) + (quemado & (wc == codigo)).sum() * ha

        # distancia (m) de cada píxel al píxel quemado más cercano
        dist = ndimage.distance_transform_edt(~quemado) * res if quemado.any() \
            else np.full(d.shape, np.inf)
        # fracción de cada cobertura en un círculo de 100 m
        r = int(RADIO_M / res)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disco = (xx ** 2 + yy ** 2 <= r ** 2).astype("float32")
        disco /= disco.sum()
        codigos = [c for c in np.unique(wc) if c in WORLDCOVER]
        fracciones = np.stack([ndimage.convolve((wc == c).astype("float32"), disco,
                                                mode="nearest") for c in codigos])
        veg = fracciones[[i for i, c in enumerate(codigos) if c in VEGETACION]].sum(0)

        filas, cols = rasterio.transform.rowcol(transform, cent.x.values, cent.y.values)
        filas, cols = np.asarray(filas), np.asarray(cols)
        dentro = (filas >= 0) & (filas < d.shape[0]) & (cols >= 0) & (cols < d.shape[1])
        idx = np.where(dentro)[0]
        dd = dist[filas[idx], cols[idx]]
        # una casa puede caer en dos rectángulos: se queda con el más cercano al fuego
        mejor = dd < edif["dist_quemado_m"].values[idx]
        sel, f, c = idx[mejor], filas[idx][mejor], cols[idx][mejor]
        edif.iloc[sel, edif.columns.get_loc("dist_quemado_m")] = dd[mejor]
        edif.iloc[sel, edif.columns.get_loc("pct_vegetacion_100m")] = 100 * veg[f, c]
        dominante = np.array(codigos)[fracciones[:, f, c].argmax(0)]
        edif.iloc[sel, edif.columns.get_loc("cobertura_100m")] = [
            WORLDCOVER[int(k)] for k in dominante]

    edif["riesgo"] = pd.cut(edif.dist_quemado_m, RIESGO_CORTES, labels=RIESGO)
    edif.to_parquet(OUT / f"indice_{nombre}.parquet")

    cob = pd.Series(hectareas, name="ha").sort_values(ascending=False)
    cob = pd.DataFrame({"ha": cob, "pct": 100 * cob / cob.sum()})
    cob.to_csv(OUT / f"cobertura_quemada_{nombre}.csv", index_label="cobertura")

    cruce = pd.crosstab(edif.riesgo, edif.cobertura_100m, margins=True, margins_name="total")
    cruce = cruce.reindex(RIESGO + ["total"])
    cruce.to_csv(OUT / f"riesgo_por_cobertura_{nombre}.csv")

    print(f"\n{distrito} · ¿qué se quemó? (dNBR sembrado, {cob.ha.sum():,.0f} ha)")
    print(cob.round(0).to_string())
    print(f"\nEdificaciones: {len(edif):,} · dentro de lo quemado (<= 20 m): "
          f"{(edif.dist_quemado_m <= 20).sum():,}")
    print("Riesgo por distancia × cobertura dominante en 100 m:")
    print(cruce.to_string())
    print("\n% de vegetación en 100 m (mediana) por riesgo:")
    print(edif.groupby("riesgo", observed=False).pct_vegetacion_100m.median().round(0).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
