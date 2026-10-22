"""Paso 7 · Edificaciones (Google + Microsoft Open Buildings, VIDA) en la zona del fuego.

Uso:  python scripts/06_buildings.py "RIO TAMBO"

Lee de forma remota el GeoParquet del Perú (2,7 GB) y baja solo las filas
dentro de la caja de la zona del fuego. Luego se queda con las que caen
dentro de la zona (focos + 2 km).
"""
import sys
from pathlib import Path

import duckdb
import geopandas as gpd

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"
URL = ("https://data.source.coop/vida/google-microsoft-open-buildings/"
       "geoparquet/by_country/country_iso=PER/PER.parquet")

distrito = sys.argv[1]
nombre = distrito.lower().replace(" ", "_")
zonas = gpd.read_file(OUT / "zonas_fuego.gpkg")
zona = zonas[zonas.NOMBDIST == distrito].to_crs(4326)
x0, y0, x1, y1 = zona.total_bounds

con = duckdb.connect()
con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET enable_progress_bar=false")
df = con.sql(f"""
    SELECT bf_source, confidence, area_in_meters, ST_AsWKB(geometry) AS wkb
    FROM '{URL}'
    WHERE bbox.xmin < {x1} AND bbox.xmax > {x0} AND bbox.ymin < {y1} AND bbox.ymax > {y0}
""").df()
edif = gpd.GeoDataFrame(df.drop(columns="wkb"),
                        geometry=gpd.GeoSeries.from_wkb(df.wkb.apply(bytes)), crs=4326)
edif = gpd.sjoin(edif, zona[["geometry"]], predicate="intersects").drop(columns="index_right")
edif.to_parquet(OUT / f"edificaciones_{nombre}.parquet")

print(f"{distrito}: {len(edif):,} edificaciones en la zona del fuego")
print(edif.groupby("bf_source").agg(n=("area_in_meters", "size"),
                                    area_mediana_m2=("area_in_meters", "median"),
                                    confianza_media=("confidence", "mean")).round(2))
