"""Paso 9 · Exporta los resultados a docs/data/ para el dashboard (Leaflet).

Uso:  python scripts/08_exportar.py

Por cada distrito con resultados completos escribe:
  - docs/data/<slug>.json : focos, edificaciones, límite y capas de dNBR
  - docs/data/img/<slug>_parte_XX.png : dNBR por clase, en EPSG:4326
y un docs/data/resumen.json con las cifras de todos.

Colores del PNG (el front decide cómo leerlos):
  clase de severidad (solo a <= 1 km de un foco) · gris = observado sin quemar
  o fuera del radio de focos · celeste = nubes, humo o sin dato · transparente
  = fuera de la zona analizada.
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"
WEB = RAIZ / "docs" / "data"
(WEB / "img").mkdir(parents=True, exist_ok=True)

TOP = ["RIO TAMBO", "NUEVA REQUENA", "MASISEA", "RAYMONDI", "TAHUANIA"]
RIESGO = ["muy alto", "alto", "medio", "bajo"]
COBERTURAS = ["bosque", "arbustal", "pastizal", "cultivo", "urbano", "suelo desnudo",
              "nieve", "agua", "humedal", "manglar", "musgo"]
# códigos del PNG: 0 fuera de zona, 1 sin dato, 2 no quemado, 3 baja, 4 mod. baja,
# 5 mod. alta, 6 alta
# lo no quemado va transparente y las nubes en gris tenue: así solo resalta lo quemado
COLORES = np.array([
    [0, 0, 0, 0],
    [140, 146, 152, 85],
    [0, 0, 0, 0],
    [254, 227, 145, 150],
    [254, 153, 41, 255],
    [215, 48, 31, 255],
    [103, 0, 13, 255],
], dtype=np.uint8)
CORTES = [0.10, 0.27, 0.44, 0.66]
# los límites del INEI vienen sin tildes
TILDES = {"Rio Tambo": "Río Tambo", "Junin": "Junín", "Raymondi": "Raymondi"}


def slug(distrito):
    return distrito.lower().replace(" ", "_")


def codificar(base, sembrado, zona):
    cod = np.zeros(base.shape, np.uint8)
    cod[zona] = 1
    observado = zona & ~np.isnan(base)
    cod[observado] = 2
    con_clase = observado & ~np.isnan(sembrado)
    cod[con_clase] = 2 + np.digitize(sembrado[con_clase], CORTES).astype(np.uint8)
    cod[con_clase & (sembrado < CORTES[0])] = 2
    return cod


def capas(distrito, zona_geom):
    carpeta = OUT / "dnbr" / slug(distrito)
    partes = gpd.GeoSeries([zona_geom], crs=32718).explode(index_parts=False)
    salida = []
    for i, geom in enumerate(partes, start=1):
        base_tif = carpeta / f"parte_{i:02d}.tif"
        with rasterio.open(base_tif) as src:
            base = src.read(1)
            zona = ~geometry_mask([geom], out_shape=base.shape, transform=src.transform)
            with rasterio.open(carpeta / f"parte_{i:02d}_sembrado.tif") as s:
                sembrado = s.read(1)
            cod = codificar(base, sembrado, zona)
            dst_t, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width,
                                                      src.height, *src.bounds)
            cod_ll = np.zeros((h, w), np.uint8)
            reproject(cod, cod_ll, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_t, dst_crs="EPSG:4326",
                      resampling=Resampling.nearest)
        nombre = f"{slug(distrito)}_parte_{i:02d}.png"
        plt.imsave(WEB / "img" / nombre, COLORES[cod_ll])
        oeste, norte = dst_t * (0, 0)
        este, sur = dst_t * (w, h)
        salida.append({"img": f"data/img/{nombre}", "bounds": [[sur, oeste], [norte, este]]})
    return salida


def main():
    ranking = pd.read_csv(OUT / "distritos_focos_7d.csv")
    dnbr = pd.read_csv(OUT / "dnbr_resumen.csv").set_index("distrito")
    focos = gpd.read_file(OUT / "focos_peru_7d.gpkg")
    zonas = gpd.read_file(OUT / "zonas_fuego.gpkg").set_index("NOMBDIST")
    limites = gpd.read_file(RAIZ / "data" / "raw" / "peru_distrital.geojson")

    resumen = []
    for distrito in TOP:
        indice = OUT / f"indice_{slug(distrito)}.parquet"
        if not indice.exists() or distrito not in dnbr.index:
            print(f"{distrito}: sin resultados completos, se omite")
            continue
        edif = gpd.read_parquet(indice).to_crs(4326)
        cent = edif.centroid
        fila_r = ranking[ranking.NOMBDIST == distrito].iloc[0]
        fila_d = dnbr.loc[distrito]
        cob = pd.read_csv(OUT / f"cobertura_quemada_{slug(distrito)}.csv")
        cruce = pd.read_csv(OUT / f"riesgo_por_cobertura_{slug(distrito)}.csv", index_col=0)
        f = focos[focos.NOMBDIST == distrito]
        lim = limites[limites.NOMBDIST == distrito].simplify(0.002)

        detalle = {
            "focos": [[round(y, 5), round(x, 5), d[5:]] for x, y, d in
                      zip(f.longitude, f.latitude, f.acq_date)],
            "edificaciones": [
                [round(y, 5), round(x, 5), RIESGO.index(r) if r in RIESGO else 3,
                 COBERTURAS.index(c) if c in COBERTURAS else -1, int(min(dq, 99999))]
                for x, y, r, c, dq in zip(cent.x, cent.y, edif.riesgo.astype(str),
                                          edif.cobertura_100m, edif.dist_quemado_m)],
            "limite": json.loads(lim.to_json()),
            "capas": capas(distrito, zonas.loc[distrito].geometry),
        }
        (WEB / f"{slug(distrito)}.json").write_text(json.dumps(detalle, separators=(",", ":")))

        resumen.append({
            "id": slug(distrito),
            "distrito": TILDES.get(distrito.title(), distrito.title()),
            "provincia": fila_r.NOMBPROV.title(),
            "departamento": TILDES.get(fila_r.NOMBDEP.title(), fila_r.NOMBDEP.title()),
            "rank": int(fila_r["rank"]),
            "huella_viirs_ha": round(float(fila_r.huella_ha)),
            "focos": int(fila_r.focos),
            "frp_mw": round(float(fila_r.frp_mw)),
            "quemado_ha": round(float(fila_d.quemado_sembrado_ha)),
            "quemado_sin_filtro_ha": round(float(fila_d.quemado_sin_filtro_ha)),
            "severidad_ha": {k: round(float(fila_d[f"{k}_ha"])) for k in
                             ["baja", "moderada baja", "moderada alta", "alta"]},
            "sin_dato_pct": round(float(fila_d.sin_dato_pct), 1),
            "zona_ha": round(float(fila_d.zona_ha)),
            "cobertura_quemada": {r.cobertura: round(float(r.ha)) for r in cob.itertuples()},
            "edificaciones": int(len(edif)),
            "dentro_quemado": int((edif.dist_quemado_m <= 20).sum()),
            "riesgo": {r: int(cruce.loc[r, "total"]) for r in RIESGO},
            "riesgo_por_cobertura": {
                r: {c: int(cruce.loc[r, c]) for c in cruce.columns if c != "total"}
                for r in RIESGO},
            "centro": [round(float(cent.y.median()), 4), round(float(cent.x.median()), 4)],
        })
        print(f"{distrito}: exportado ({len(detalle['capas'])} capas)")

    meta = {
        "corte": "2026-10-21",
        "ventana_focos": [str(focos.acq_date.min()), str(focos.acq_date.max())],
        "imagen_antes": "20 sep – 13 oct 2026 (mediana sin nubes)",
        "imagen_despues": "16 – 21 oct 2026 (último píxel sin nubes)",
        "distritos": resumen,
    }
    (WEB / "resumen.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
