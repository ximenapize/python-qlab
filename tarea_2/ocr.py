"""OCR de los 25 documentos con PaddleOCR (Sesión 10). Lo usa 01_extraccion.ipynb.

    .venv/bin/python ocr.py                    # procesa los documentos de datos/fuentes.csv que falten
    .venv/bin/python ocr.py --rehacer doc_016  # vuelve a procesar esos documentos (p. ej., si se reemplazó la imagen)

Configuración (la misma para todos los documentos, sin ajustes por documento):
- PaddleOCR 3.7, motor PaddlePaddle en CPU, lang="es" -> PP-OCRv6_medium_det + PP-OCRv6_medium_rec.
- Sin clasificador de orientación, sin "planchado" de la hoja y sin giro por línea (como en la clase).
- Sin preprocesamiento: la única transformación es la de descargar_corpus.py (máximo 1920 px de
  ancho, RGB, sin EXIF).

Por cada documento guarda en datos/raw/documentos/:
- ocr/doc_XXX.json   respaldo crudo: texto, confianza y caja de cada línea
- recortes/doc_XXX/  una imagen por línea detectada (linea_000.jpg, ...)
Si el JSON ya existe, se lee en vez de volver a correr el modelo.
"""
from __future__ import annotations

import json
import logging
import os
import time
import warnings
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"   # no probar la conexión a cada servidor
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

CARPETA = Path(__file__).resolve().parent
RAW_DOCS = CARPETA / "datos" / "raw" / "documentos"
CRUDO = RAW_DOCS / "ocr"
RECORTES = RAW_DOCS / "recortes"

OPCIONES = dict(
    lang="es",
    device="cpu",
    use_doc_orientation_classify=False,   # no girar la página completa
    use_doc_unwarping=False,              # no "planchar" hojas dobladas
    use_textline_orientation=False,       # no girar cada línea
)


def crear_ocr():
    from paddleocr import PaddleOCR
    for nombre in ("paddlex", "paddleocr"):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    return PaddleOCR(**OPCIONES)


def descripcion_modelo() -> str:
    import paddleocr
    return f"PaddleOCR {paddleocr.__version__} (PP-OCRv6_medium det+rec, lang=es, CPU, sin preprocesamiento)"


def leer(ruta: Path, ocr) -> dict:
    """Corre el OCR sobre una imagen y devuelve líneas, confianzas, cajas y segundos."""
    t0 = time.perf_counter()
    r = ocr.predict(str(ruta))[0]
    cajas = np.asarray(r["rec_boxes"]).reshape(-1, 4).astype(int).tolist()   # x0, y0, x1, y1
    return {
        "archivo": ruta.name,
        "segundos": round(time.perf_counter() - t0, 2),
        "lineas": [{"texto": t, "confianza": round(float(c), 4), "caja": b}
                   for t, c, b in zip(r["rec_texts"], r["rec_scores"], cajas)],
    }


def guardar_recortes(ruta: Path, resultado: dict):
    destino = RECORTES / ruta.stem
    destino.mkdir(parents=True, exist_ok=True)
    for viejo in destino.glob("linea_*.jpg"):   # recortes de una corrida anterior
        viejo.unlink()
    im = Image.open(ruta).convert("RGB")
    for i, linea in enumerate(resultado["lineas"]):
        x0, y0, x1, y1 = linea["caja"]
        if x1 > x0 and y1 > y0:
            im.crop((x0, y0, x1, y1)).save(destino / f"linea_{i:03d}.jpg", quality=90)


def procesar(rutas: list[Path], ocr=None, rehacer: set[str] = frozenset()) -> dict[str, dict]:
    """OCR de cada ruta, usando el respaldo JSON cuando ya existe (salvo los ids en `rehacer`)."""
    CRUDO.mkdir(parents=True, exist_ok=True)
    salida = {}
    for ruta in rutas:
        json_ruta = CRUDO / f"{ruta.stem}.json"
        if json_ruta.exists() and ruta.stem not in rehacer:
            salida[ruta.stem] = json.loads(json_ruta.read_text(encoding="utf-8"))
            continue
        ocr = ocr or crear_ocr()
        resultado = leer(ruta, ocr)
        resultado["modelo_ocr"] = descripcion_modelo()
        json_ruta.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
        guardar_recortes(ruta, resultado)
        print(f"  {ruta.stem}: {len(resultado['lineas'])} líneas en {resultado['segundos']} s", flush=True)
        salida[ruta.stem] = resultado
    return salida


def texto(resultado: dict) -> str:
    """Las líneas en el orden en que las devuelve PaddleOCR (de arriba hacia abajo), una por renglón."""
    return "\n".join(l["texto"] for l in resultado["lineas"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehacer", nargs="*", default=[], help="ids a procesar de nuevo aunque ya tengan JSON")
    args = ap.parse_args()
    fuentes = pd.read_csv(CARPETA / "datos" / "fuentes.csv")
    docs = fuentes[fuentes["modalidad"] == "documento"]
    resultados = procesar([CARPETA / a for a in docs["archivo"]], rehacer=set(args.rehacer))
    tabla = pd.DataFrame([{"id": k, "lineas": len(v["lineas"]), "caracteres": len(texto(v)),
                           "conf_media": np.mean([l["confianza"] for l in v["lineas"]]) if v["lineas"] else None,
                           "segundos": v["segundos"]} for k, v in resultados.items()])
    print(tabla.round(3).to_string(index=False))
