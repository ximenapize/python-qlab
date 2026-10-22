"""Paso 10 · Capas base del mapa web: relieve sombreado, límites y focos del Perú.

Uso:  python scripts/09_base_mapa.py

  - docs/data/relieve.png : sombreado del DEM Copernicus GLO-90 (~550 m), Perú central
  - docs/data/limites.json : departamentos y provincias (líneas) + etiquetas
  - docs/data/focos_peru.json : todos los focos VIIRS del Perú, 14–21 oct
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
import pystac_client

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"
WEB = RAIZ / "docs" / "data"

# caja que cubre los cinco distritos con margen (oeste, sur, este, norte)
CAJA = (-77.5, -13.5, -72.0, -7.0)
RES = 0.005  # grados, ~550 m


def relieve():
    catalogo = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = catalogo.search(collections=["cop-dem-glo-90"], bbox=CAJA).item_collection()
    dem = odc.stac.load(items, bands=["data"], bbox=CAJA, crs="EPSG:4326",
                        resolution=RES, resampling="average").data.isel(time=0).values
    dem = np.nan_to_num(dem.astype("float32"), nan=0)

    # sombreado con luz del noroeste; dx en metros según la latitud media
    dy = RES * 111_320
    dx = dy * np.cos(np.radians((CAJA[1] + CAJA[3]) / 2))
    gy, gx = np.gradient(dem, dy, dx)
    pendiente = np.arctan(2 * np.hypot(gx, gy))
    aspecto = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    sombra = (np.sin(alt) * np.cos(pendiente)
              + np.cos(alt) * np.sin(pendiente) * np.cos(az - aspecto))
    sombra = np.clip(sombra, 0, 1)

    # tinte oscuro: el relieve se lee pero no compite con los datos
    base = np.array([28, 33, 38]) / 255
    luz = np.array([92, 104, 112]) / 255
    rgb = base + (luz - base) * sombra[..., None] ** 1.4
    plt.imsave(WEB / "relieve.png", np.clip(rgb, 0, 1))
    print("relieve:", dem.shape)


def limites():
    dist = gpd.read_file(RAIZ / "data" / "raw" / "peru_distrital.geojson")
    prov = dist.dissolve("IDPROV" if "IDPROV" in dist else ["NOMBDEP", "NOMBPROV"]).reset_index()
    dep = dist.dissolve("NOMBDEP").reset_index()
    capas = {
        "departamentos": json.loads(dep[["NOMBDEP", "geometry"]].boundary
                                    .simplify(0.01).to_json()),
        "provincias": json.loads(prov[["geometry"]].boundary.simplify(0.005).to_json()),
        "etiquetas": [{"nombre": n.title().replace("Junin", "Junín").replace("Huanuco", "Huánuco"),
                       "pos": [round(p.x, 3), round(p.y, 3)]}
                      for n, p in zip(dep.NOMBDEP, dep.geometry.representative_point())],
    }
    (WEB / "limites.json").write_text(json.dumps(capas, separators=(",", ":"), ensure_ascii=False))
    print("límites:", len(dep), "departamentos,", len(prov), "provincias")


def focos():
    f = gpd.read_file(OUT / "focos_peru_7d.gpkg")
    datos = [[round(x, 4), round(y, 4), int(d[8:])] for x, y, d in
             zip(f.longitude, f.latitude, f.acq_date)]
    (WEB / "focos_peru.json").write_text(json.dumps(datos, separators=(",", ":")))
    print("focos Perú:", len(datos))


if __name__ == "__main__":
    limites()
    focos()
    relieve()
    meta = {"relieve_bounds": [CAJA[0], CAJA[1], CAJA[2], CAJA[3]]}
    (WEB / "base.json").write_text(json.dumps(meta))
