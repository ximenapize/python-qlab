"""Paso 5 · Mapa de severidad dNBR de la parte más grande de un distrito, con focos VIIRS."""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "processed"

distrito = sys.argv[1]
carpeta = OUT / "dnbr" / distrito.lower().replace(" ", "_")
resumen = pd.read_csv(carpeta / "resumen_partes.csv")
parte = int(resumen.sort_values("zona_ha").parte.iloc[-1])

with rasterio.open(carpeta / f"parte_{parte:02d}.tif") as src:
    d = src.read(1)
    ext = rasterio.plot.plotting_extent(src) if hasattr(rasterio, "plot") else None
    b = src.bounds
    ext = (b.left, b.right, b.bottom, b.top)

focos = gpd.read_file(OUT / "focos_peru_7d.gpkg")
focos = focos[focos.NOMBDIST == distrito].to_crs(32718).cx[b.left:b.right, b.bottom:b.top]

limites = [-1, 0.10, 0.27, 0.44, 0.66, 2]
colores = ["#d9d9d9", "#fee391", "#fe9929", "#d7301f", "#67000d"]
cmap = ListedColormap(colores)
norm = BoundaryNorm(limites, cmap.N)

fig, ax = plt.subplots(figsize=(9, 8))
ax.set_facecolor("#9ecae1")  # sin dato (nube o fuera de zona) se ve celeste
ax.imshow(np.where(np.isnan(d), np.nan, d), cmap=cmap, norm=norm, extent=ext)
ax.scatter(focos.geometry.x, focos.geometry.y, s=4, c="black", alpha=0.6, label="Focos VIIRS 14–21 oct")
ax.set_title(f"{distrito.title()} · severidad dNBR (antes: 20 sep–13 oct, después: 16–21 oct)", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colores] + [plt.Rectangle((0, 0), 1, 1, color="#9ecae1")]
etiquetas = ["No quemado", "Baja", "Moderada baja", "Moderada alta", "Alta", "Sin dato / fuera de zona"]
ax.legend(handles + [ax.collections[0]], etiquetas + ["Focos VIIRS"], loc="lower left", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(RAIZ / "outputs" / f"dnbr_{carpeta.name}.png", dpi=130)
print("parte", parte, "->", RAIZ / "outputs" / f"dnbr_{carpeta.name}.png")
