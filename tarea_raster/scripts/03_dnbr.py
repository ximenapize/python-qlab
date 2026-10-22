"""Paso 4 · dNBR con compuestos de Sentinel-2 L2A, por distrito.

Uso:  python scripts/03_dnbr.py "RIO TAMBO"

Decisiones (ver README):
  - Antes   : mediana de píxeles sin nube, 20 sep – 13 oct
  - Después : último píxel sin nube, 16 – 21 oct
  - NBR = (B8A - B12) / (B8A + B12), reflectancia con el offset de -1000 de L2A
  - Quemado = dNBR >= 0,27; severidad según Key y Benson (2006)
  - Zona = focos VIIRS del distrito con buffer de 2 km; se lee solo el
    rectángulo de cada grupo de focos, no la caja completa del distrito
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
from rasterio.features import geometry_mask

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"

ANTES = "2026-09-20/2026-10-13"
DESPUES = "2026-10-16/2026-10-21"
RES = 20
CRS = "EPSG:32718"
# SCL válidos: 2 sombra de relieve, 4 vegetación, 5 suelo, 7 sin clasificar
SCL_OK = [2, 4, 5, 7]
OFFSET = 1000  # BOA_ADD_OFFSET de L2A desde la línea base 04.00 (2022)
CLASES = [  # (nombre, mínimo, máximo) de dNBR
    ("no quemado", -np.inf, 0.10),
    ("baja", 0.10, 0.27),
    ("moderada baja", 0.27, 0.44),
    ("moderada alta", 0.44, 0.66),
    ("alta", 0.66, np.inf),
]
HA_PIXEL = RES * RES / 10_000

catalogo = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


def nbr(items, bbox):
    """Pila de NBR por fecha, con NaN donde SCL marca nube, agua o sin dato."""
    ds = odc.stac.load(items, bands=["B8A", "B12", "SCL"], bbox=bbox, crs=CRS,
                       resolution=RES, groupby="solar_day", chunks={"x": 2048, "y": 2048})
    nir = ds.B8A.astype("float32") - OFFSET
    swir = ds.B12.astype("float32") - OFFSET
    valido = ds.SCL.isin(SCL_OK) & (nir > 0) & (swir > 0)
    return ((nir - swir) / (nir + swir)).where(valido)


def dnbr_parte(geom, bbox):
    antes = catalogo.search(collections=["sentinel-2-l2a"], bbox=bbox,
                            datetime=ANTES).item_collection()
    despues = catalogo.search(collections=["sentinel-2-l2a"], bbox=bbox,
                              datetime=DESPUES).item_collection()
    pre = nbr(antes, bbox).median("time")
    post = nbr(despues, bbox).sortby("time").ffill("time").isel(time=-1)
    d = (pre - post).compute()
    dentro = ~geometry_mask([geom], out_shape=d.shape, transform=d.odc.geobox.affine)
    return d.where(dentro), dentro, len(antes), len(despues)


def main(distrito: str):
    zonas = gpd.read_file(OUT / "zonas_fuego.gpkg")
    zona = zonas[zonas.NOMBDIST == distrito].geometry.iloc[0]
    partes = gpd.GeoSeries([zona], crs=32718).explode(index_parts=False)
    carpeta = OUT / "dnbr" / distrito.lower().replace(" ", "_")
    carpeta.mkdir(parents=True, exist_ok=True)

    filas = []
    for i, geom in enumerate(partes, start=1):
        bbox = gpd.GeoSeries([geom], crs=32718).to_crs(4326).total_bounds
        d, dentro, n_antes, n_despues = dnbr_parte(geom, bbox)
        valores = d.values
        fila = {"parte": i, "zona_ha": dentro.sum() * HA_PIXEL,
                "sin_dato_ha": (np.isnan(valores) & dentro).sum() * HA_PIXEL,
                "escenas_antes": n_antes, "escenas_despues": n_despues}
        for nombre, lo, hi in CLASES:
            fila[nombre] = ((valores >= lo) & (valores < hi)).sum() * HA_PIXEL
        filas.append(fila)

        perfil = {"driver": "GTiff", "dtype": "float32", "count": 1, "nodata": np.nan,
                  "width": d.shape[1], "height": d.shape[0], "crs": CRS,
                  "transform": d.odc.geobox.affine, "compress": "deflate"}
        with rasterio.open(carpeta / f"parte_{i:02d}.tif", "w", **perfil) as dst:
            dst.write(valores.astype("float32"), 1)
        print(f"  parte {i}/{len(partes)} · {fila['zona_ha']:,.0f} ha de zona")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / "resumen_partes.csv", index=False)
    tot = tabla.sum(numeric_only=True)
    quemado = tot[["moderada baja", "moderada alta", "alta"]].sum()
    print(f"\n{distrito} · zona {tot.zona_ha:,.0f} ha · sin dato "
          f"{100 * tot.sin_dato_ha / tot.zona_ha:.0f} %")
    for nombre, _, _ in CLASES:
        print(f"  {nombre:14s} {tot[nombre]:>10,.0f} ha")
    print(f"  QUEMADO (dNBR >= 0,27): {quemado:,.0f} ha")


if __name__ == "__main__":
    main(sys.argv[1])
