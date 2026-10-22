"""Paso 6 · dNBR sembrado con focos: quemado solo a <= 1 km de un foco VIIRS.

Uso:  python scripts/05_sembrado.py "RIO TAMBO" ["MASISEA" ...]

Lee los rasters dNBR del paso 4 y cuenta hectáreas por clase de severidad
en dos versiones: sin filtro y solo a <= 1 km de un foco del 14 al 21 oct.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"

DIST_FOCO_M = 1000
QUEMADO = 0.27
CLASES = [
    ("baja", 0.10, 0.27),
    ("moderada baja", 0.27, 0.44),
    ("moderada alta", 0.44, 0.66),
    ("alta", 0.66, np.inf),
]

focos = gpd.read_file(OUT / "focos_peru_7d.gpkg").to_crs(32718)


def resumen(distrito: str) -> dict:
    carpeta = OUT / "dnbr" / distrito.lower().replace(" ", "_")
    cerca = focos[focos.NOMBDIST == distrito].buffer(DIST_FOCO_M).union_all()
    fila = {"distrito": distrito, "zona_ha": 0.0, "sin_dato_ha": 0.0,
            "quemado_sin_filtro_ha": 0.0, "quemado_sembrado_ha": 0.0}
    fila |= {f"{n}_ha": 0.0 for n, _, _ in CLASES}

    for tif in sorted(carpeta.glob("parte_??.tif")):
        with rasterio.open(tif) as src:
            d = src.read(1)
            ha = abs(src.res[0] * src.res[1]) / 10_000
            sembrado = ~geometry_mask([cerca], out_shape=d.shape, transform=src.transform)
            # el sembrado queda guardado como banda aparte para los pasos siguientes
            perfil = src.profile
        fila["quemado_sin_filtro_ha"] += (d >= QUEMADO).sum() * ha
        fila["quemado_sembrado_ha"] += ((d >= QUEMADO) & sembrado).sum() * ha
        for n, lo, hi in CLASES:
            fila[f"{n}_ha"] += ((d >= lo) & (d < hi) & sembrado).sum() * ha

        with rasterio.open(tif.with_name(tif.stem + "_sembrado.tif"), "w", **perfil) as dst:
            dst.write(np.where(sembrado, d, np.nan).astype("float32"), 1)
    # zona y sin dato vienen del paso 4, que los mide dentro de cada parte
    partes = pd.read_csv(carpeta / "resumen_partes.csv")
    fila["zona_ha"] = partes.zona_ha.sum()
    fila["sin_dato_ha"] = partes.sin_dato_ha.sum()
    fila["sin_dato_pct"] = 100 * fila["sin_dato_ha"] / fila["zona_ha"]
    return fila


if __name__ == "__main__":
    tabla = pd.DataFrame([resumen(d) for d in sys.argv[1:]])
    destino = OUT / "dnbr_resumen.csv"
    if destino.exists():
        previo = pd.read_csv(destino)
        tabla = pd.concat([previo[~previo.distrito.isin(tabla.distrito)], tabla])
    tabla.to_csv(destino, index=False)
    print(tabla.round(0).to_string(index=False))
