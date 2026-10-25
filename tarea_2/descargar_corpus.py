"""Descarga el corpus de la Tarea 2 (documentos y audio peruanos) y escribe datos/fuentes.csv.

    .venv/bin/python descargar_corpus.py            # todo
    .venv/bin/python descargar_corpus.py --solo documentos
    .venv/bin/python descargar_corpus.py --solo audio

Qué baja (unos 80 MB) a datos/raw/, que no se versiona:

- 25 documentos: 8 impresos, 8 fotos y 9 manuscritos. 18 vienen de Wikimedia Commons, 1 del
  Archivo José Carlos Mariátegui y 6 son fotos propias que se copian a datos/raw/propias/.
  Se guardan como datos/raw/documentos/doc_XXX.jpg (1920 px de ancho como máximo).
- 17 entrevistas y testimonios en español peruano de Wikimedia Commons (unos 49 min) y
  8 frases leídas de OpenSLR 73 (Google, CC BY-SA 4.0) con su transcripción oficial.
  Todo se convierte a WAV mono de 16 kHz: datos/raw/audio/clip_XXX.wav.

Licencia y autor se leen de la API de Commons al momento de descargar, no se copian a mano.
Los nombres de archivo locales usan solo el id (doc_XXX, clip_XXX), nunca el nombre de la
persona entrevistada. Si un archivo ya está en disco no se vuelve a bajar (salvo con --rehacer).

El acceso a OpenSLR 73 (lectura parcial del zip con HTTP Range) está adaptado de
sessions/13-audio-a-texto/datos.py del curso (d2cml-ai/Data-Science-Python).
"""
from __future__ import annotations

import argparse
import io
import math
import random
import re
import time
import urllib.parse
import zipfile
from datetime import date
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from PIL import Image, ImageOps
from scipy.signal import resample_poly

CARPETA = Path(__file__).resolve().parent
RAW_DOCS = CARPETA / "datos" / "raw" / "documentos"
RAW_AUDIO = CARPETA / "datos" / "raw" / "audio"
FUENTES = CARPETA / "datos" / "fuentes.csv"

SR = 16_000
ANCHO = 1920          # ancho máximo; Commons solo sirve anchos estándar (1280, 1920, 3840...)
PAUSA = 1.5           # segundos entre peticiones (el enunciado pide de 1 a 2)
REHACER = False       # --rehacer: volver a bajar aunque el archivo ya exista
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "ds-python-up-tarea2/2026 (https://github.com/ximenapize/python-qlab; estudiante del curso)"}

sesion = requests.Session()
sesion.headers.update(UA)

# ───────────────────────────────────────────────────────── documentos
# Selección propia. (tipo, archivo en Commons, descripción corta)
DOCUMENTOS = [
    # impresos nítidos
    ("impreso", "El Comercio - Lima. Perú y Ecuador. 3-3-1842.jpg", "El Comercio, edición del 3 de marzo de 1842"),
    ("impreso", "Carta sobre los maricones 01.jpg", "Mercurio Peruano (1791), «Carta sobre los maricones», p. 1"),
    ("impreso", "Carta sobre los maricones 02.jpg", "Mercurio Peruano (1791), «Carta sobre los maricones», p. 2"),
    ("impreso", "Carta sobre los maricones 03.jpg", "Mercurio Peruano (1791), «Carta sobre los maricones», p. 3"),
    ("impreso", "Mercurio Peruano1.jpg", "Portada del Mercurio Peruano"),
    ("impreso", "Amauta magazine cover 01.jpg", "Portada de la revista Amauta, n.º 1 (1926)"),
    ("impreso", "El Comercio Issue 1.jpg", "Primer número de El Comercio (1839)"),
    ("impreso", "Football match report from Peruvian newspaper El Comercio (Lima, 1904).png",
     "Crónica de un partido de fútbol en El Comercio (1904)"),
    # fotos con sombras, reflejos o inclinación
    ("foto", "Menu 30042010.JPG", "Pizarra de menú"),
    ("foto", "Cartel Cevichería El Tiburón 11062011.jpg", "Cartel de la cevichería El Tiburón"),
    ("foto", "Camino Inca en Universidad Católica del Perú 01.jpg", "Cartel del Camino Inca en la PUCP"),
    ("foto", "Cartel Pueblo viejo Islay.png", "Cartel de Pueblo Viejo, Islay"),
    ("foto", "Cartel pantanos 2.jpg", "Cartel de los pantanos"),
    ("foto", "Menu Levaggi 09052010.jpg", "Pizarra de menú escrita con tiza (Levaggi, Lince)"),
    ("foto", "Placa Calle Capon.jpg", "Placa de la Calle Capón, Barrio Chino de Lima"),
    ("foto", "Cartel Picante de cuy.JPG", "Cartel pintado a mano en Ingenio, Junín"),
    # manuscritos
    ("manuscrito", "La Púrpura de la Rosa.jpg", "Partitura manuscrita de La púrpura de la rosa (1701)"),
    ("manuscrito", "Alma matinal.jpg", "Manuscrito de Alma matinal, José Carlos Mariátegui"),
]

# Manuscrito fuera de Commons: Archivo José Carlos Mariátegui. Se baja el PDF y se usa la página 1.
# Su robots.txt pide 10 s entre peticiones. Condición de acceso declarada: «De libre descarga y
# visualización»; no tiene licencia abierta explícita, así que la imagen no se publica en la app.
MARIATEGUI = {
    "descripcion": "Carta de José Carlos Mariátegui a Victoria Ferrer, Roma, 24-1-1920 (p. 1)",
    "fuente_url": "https://archivo.mariategui.org/index.php/carta-a-victoria-ferrer-24-1-1920",
    "pdf": "https://archivo.mariategui.org/uploads/r/archivo-jose-carlos-mariategui/b/7/7/"
           "b779fc8a5dd1103299fa2ab2b5fbd3171de3c01c445bb2894538fdbc99a4c3f5/C-1920-1-24.pdf",
    "licencia": "Libre descarga y visualización (Archivo José Carlos Mariátegui); sin licencia abierta explícita",
    "autor": "José Carlos Mariátegui; digitalización del Archivo José Carlos Mariátegui",
    "pausa": 10,
}

# Fotos propias de manuscritos: se copian a mano en datos/raw/propias/ (6 archivos .jpg/.jpeg/.png/.heic).
# El script las renombra en orden alfabético y les quita los metadatos EXIF (ubicación GPS, celular).
PROPIAS = CARPETA / "datos" / "raw" / "propias"
N_PROPIAS = 6

# ───────────────────────────────────────────────────────── audio
# (archivo en Commons, n.º de hablantes según la descripción, descripción corta)
ENTREVISTAS = [
    ("Entrevista 1 en la Av. Rivagüero - El Agustino.ogg", 2, "Entrevista en la calle a una vendedora de frutas, El Agustino (2022)"),
    ("Entrevista 2 en la Av. Rivagüero - El Agustino.ogg", 3, "Entrevista en la calle a dos vendedoras de dulces, El Agustino (2022)"),
    ("Entrevista-Aurora Coronado-organización de mujeres indígenas-fiestas patrias.ogg", 2,
     "Dirigente de Fenmucarinap en conferencia de prensa (2022)"),
    ("Entrevista a Luz Merida-Organización de mujeres indígenas-fiestas patrias-protesta.ogg", 2,
     "Dirigente de la base Apurímac de Fenmucarinap (2022)"),
    ("Testimonio Lucinda Quispealaya - Junín.ogg", 1, "Testimonio de una lideresa campesina de Junín (2024)"),
    ("Testimonio 1 - Salas, pantallas y archivos 01.ogg", 1, "Testimonio sobre salas de cine, WikiAcción Perú (2026)"),
    ("Testimonio 2 - Salas, pantallas y archivos.ogg", 1, "Testimonio sobre salas de cine, WikiAcción Perú (2026)"),
    ("Testimonio 3 - Salas, pantallas y archivos.ogg", 1, "Testimonio sobre salas de cine, WikiAcción Perú (2026)"),
    ("Entrevista 1,.wav", None, "Entrevista registrada por un estudiante de la UNMSM (2022)"),
    ("Entrevista 2.wav", None, "Entrevista registrada por un estudiante de la UNMSM (2022)"),
    ("Entrevista 3.wav", None, "Entrevista registrada por un estudiante de la UNMSM (2022)"),
    ("Entrevista 4.wav", None, "Entrevista registrada por un estudiante de la UNMSM (2022)"),
    ("Audio de la entrevista a una chica que participaba de la marcha.ogg", 2, "Entrevista a una joven en una marcha (2022)"),
    ("UNMSM Entrevista realizada en la marcha por Cristopher Serrano Pérez.wav", None, "Entrevista en una marcha, UNMSM (2023)"),
    ("UNMSM Entrevistas en la marcha 1 por Cristopher Serrano Pérez.wav", None, "Entrevistas en una marcha, UNMSM (2023)"),
    ("Entrevista a la señora Aurelia, mujer shuyasina.ogg", 2, "Entrevista sobre la fiesta de la Santísima Trinidad (2024)"),
    ("Entrevista a la señora Dina Espinoza, mujer pomabambina.ogg", 2, "Entrevista sobre la fiesta de la Santísima Trinidad (2024)"),
]

OPENSLR73 = "https://www.openslr.org/resources/73/es_pe_{sexo}.zip"
N_OPENSLR_POR_SEXO = 4
SEMILLA = 7


# ───────────────────────────────────────────────────────── utilidades
def _get(url, **kw):
    r = sesion.get(url, timeout=120, **kw)
    r.raise_for_status()
    time.sleep(PAUSA)
    return r


def _limpiar_html(texto: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", texto or ""))).strip()


def info_commons(titulos: list[str]) -> dict[str, dict]:
    """Licencia, autor y URL de descarga de cada archivo, según la API de Commons."""
    salida = {}
    for titulo in titulos:
        params = {"action": "query", "format": "json", "titles": "File:" + titulo,
                  "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata"}
        if titulo.lower().endswith((".jpg", ".jpeg", ".png")):
            params["iiurlwidth"] = ANCHO
        p = next(iter(_get(API, params=params).json()["query"]["pages"].values()))
        if "imageinfo" not in p:
            raise FileNotFoundError(f"No existe en Commons: {titulo}")
        ii = p["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        # para imágenes más anchas que ANCHO se baja la miniatura; si no, el original
        url = ii["thumburl"] if ii.get("thumburl") and ii.get("width", 0) > ANCHO else ii["url"]
        salida[titulo] = {
            "url_descarga": url,
            "fuente_url": ii["descriptionurl"],
            "licencia": _limpiar_html(meta.get("LicenseShortName", {}).get("value", "")),
            "autor": _limpiar_html(meta.get("Artist", {}).get("value", ""))[:200],
            "duracion_original_seg": round(ii.get("duration", 0) or 0, 1),
        }
    return salida


def _a_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype("float32")


def _guardar_wav_16k(contenido: bytes, ruta: Path):
    """Cualquier audio (ogg, opus, wav) -> WAV mono 16 kHz. PyAV (de faster-whisper) decodifica sin ffmpeg."""
    try:
        x, sr = sf.read(io.BytesIO(contenido), dtype="float32")
        x = _a_16k(x, sr)
    except Exception:
        from faster_whisper import decode_audio
        x = decode_audio(io.BytesIO(contenido), sampling_rate=SR)
    sf.write(ruta, x, SR)


# ───────────────────────────────────────────────────────── descargas
def _guardar_jpg(im: Image.Image, ruta: Path):
    """Endereza según EXIF, reduce a ANCHO px y guarda como JPEG sin metadatos."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width > ANCHO:
        im = im.resize((ANCHO, round(im.height * ANCHO / im.width)), Image.LANCZOS)
    im.save(ruta, "JPEG", quality=92)   # sin exif=...: Pillow no copia los metadatos


def bajar_documentos() -> pd.DataFrame:
    RAW_DOCS.mkdir(parents=True, exist_ok=True)
    filas = []
    acceso = "Wikimedia Commons; reproducir con descargar_corpus.py"

    # 1) Wikimedia Commons
    info = info_commons([t for _, t, _ in DOCUMENTOS])
    for tipo, titulo, desc in DOCUMENTOS:
        id_ = f"doc_{len(filas) + 1:03d}"
        ruta = RAW_DOCS / f"{id_}.jpg"
        meta = info[titulo]
        if REHACER or not ruta.exists():
            _guardar_jpg(Image.open(io.BytesIO(_get(meta["url_descarga"]).content)), ruta)
            print(f"  {id_} {tipo:<10} {desc}")
        filas.append({"id": id_, "modalidad": "documento", "tipo": tipo, "descripcion": desc,
                      "archivo": ruta.relative_to(CARPETA).as_posix(), "fuente_url": meta["fuente_url"],
                      "licencia": meta["licencia"], "autor": meta["autor"], "acceso": acceso})

    # 2) Archivo José Carlos Mariátegui: página 1 del PDF
    id_ = f"doc_{len(filas) + 1:03d}"
    ruta = RAW_DOCS / f"{id_}.jpg"
    if REHACER or not ruta.exists():
        import pypdfium2
        pdf = sesion.get(MARIATEGUI["pdf"], timeout=120)
        pdf.raise_for_status()
        time.sleep(MARIATEGUI["pausa"])
        pagina = pypdfium2.PdfDocument(pdf.content)[0]
        _guardar_jpg(pagina.render(scale=ANCHO / pagina.get_size()[0]).to_pil(), ruta)
        print(f"  {id_} manuscrito {MARIATEGUI['descripcion']}")
    filas.append({"id": id_, "modalidad": "documento", "tipo": "manuscrito",
                  "descripcion": MARIATEGUI["descripcion"], "archivo": ruta.relative_to(CARPETA).as_posix(),
                  "fuente_url": MARIATEGUI["fuente_url"], "licencia": MARIATEGUI["licencia"],
                  "autor": MARIATEGUI["autor"], "acceso": f"PDF en {MARIATEGUI['pdf']} (página 1)"})

    # 3) fotos propias de manuscritos
    fotos = sorted(f for f in PROPIAS.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(fotos) != N_PROPIAS:
        print(f"  ¡Ojo! Se esperaban {N_PROPIAS} fotos propias en {PROPIAS.relative_to(CARPETA)} y hay {len(fotos)}.")
    for foto in fotos:
        id_ = f"doc_{len(filas) + 1:03d}"
        ruta = RAW_DOCS / f"{id_}.jpg"
        if REHACER or not ruta.exists():
            _guardar_jpg(Image.open(foto), ruta)
            print(f"  {id_} manuscrito foto propia ({foto.name})")
        filas.append({"id": id_, "modalidad": "documento", "tipo": "manuscrito",
                      "descripcion": "Manuscrito fotografiado por la autora", "archivo": ruta.relative_to(CARPETA).as_posix(),
                      "fuente_url": None, "licencia": "Captura propia (Ximena Pinillos), sin datos personales",
                      "autor": "Ximena Pinillos", "acceso": "Foto propia; no se publica el original"})
    return pd.DataFrame(filas)


class ZipRemoto(io.RawIOBase):
    """Zip remoto leído por partes con HTTP Range: solo se bajan los bytes de las frases elegidas."""

    BLOQUE = 1 << 20

    def __init__(self, url):
        self.url, self.pos, self.buf_ini, self.buf = url, 0, 0, b""
        self.size = int(sesion.head(url, timeout=60, allow_redirects=True).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        fin = self.pos + n
        if not (self.buf_ini <= self.pos and fin <= self.buf_ini + len(self.buf)):
            self.buf_ini = self.pos
            hasta = min(self.pos + max(n, self.BLOQUE), self.size) - 1
            r = sesion.get(self.url, headers={"Range": f"bytes={self.pos}-{hasta}"}, timeout=120)
            r.raise_for_status()
            self.buf = r.content
        datos = self.buf[self.pos - self.buf_ini:fin - self.buf_ini]
        self.pos += len(datos)
        return datos


def bajar_audio() -> pd.DataFrame:
    RAW_AUDIO.mkdir(parents=True, exist_ok=True)
    filas = []

    # 1) entrevistas de Commons
    info = info_commons([t for t, _, _ in ENTREVISTAS])
    for i, (titulo, n_hab, desc) in enumerate(ENTREVISTAS, start=1):
        id_ = f"clip_{i:03d}"
        ruta = RAW_AUDIO / f"{id_}.wav"
        meta = info[titulo]
        if REHACER or not ruta.exists():
            _guardar_wav_16k(_get(meta["url_descarga"]).content, ruta)
            print(f"  {id_} {meta['duracion_original_seg']:6.0f} s  {desc}")
        filas.append({
            "id": id_, "modalidad": "audio", "tipo": "entrevista", "descripcion": desc,
            "archivo": ruta.relative_to(CARPETA).as_posix(), "fuente_url": meta["fuente_url"],
            "licencia": meta["licencia"], "autor": meta["autor"],
            "acceso": "Wikimedia Commons; reproducir con descargar_corpus.py",
            "n_hablantes_fuente": n_hab, "texto_oficial": None,
        })

    # 2) frases leídas de OpenSLR 73, una por hablante, con semilla fija
    siguiente = len(ENTREVISTAS) + 1
    for sexo in ["female", "male"]:
        url = OPENSLR73.format(sexo=sexo)
        z = zipfile.ZipFile(ZipRemoto(url))
        lineas = z.read("line_index.tsv").decode("utf-8").strip().splitlines()
        tabla = pd.DataFrame([l.split("\t") for l in lineas], columns=["id_openslr", "texto"])
        tabla["hablante"] = tabla["id_openslr"].str.rsplit("_", n=1).str[0]
        tabla = tabla[tabla["texto"].str.split().str.len().between(8, 25)]
        hablantes = sorted(tabla["hablante"].unique())
        random.Random(SEMILLA).shuffle(hablantes)
        for h in hablantes[:N_OPENSLR_POR_SEXO]:
            fila = tabla[tabla["hablante"] == h].sample(1, random_state=SEMILLA).iloc[0]
            id_ = f"clip_{siguiente:03d}"
            siguiente += 1
            ruta = RAW_AUDIO / f"{id_}.wav"
            if REHACER or not ruta.exists():
                _guardar_wav_16k(z.read(f"{fila['id_openslr']}.wav"), ruta)
                print(f"  {id_} OpenSLR 73 {fila['id_openslr']}")
            filas.append({
                "id": id_, "modalidad": "audio", "tipo": "lectura",
                "descripcion": f"Frase leída, {'mujer' if sexo == 'female' else 'hombre'} ({fila['id_openslr']})",
                "archivo": ruta.relative_to(CARPETA).as_posix(), "fuente_url": "https://www.openslr.org/73/",
                "licencia": "CC BY-SA 4.0", "autor": "Google (Guevara-Rukoz et al., 2020)",
                "acceso": f"Archivo {fila['id_openslr']}.wav dentro de {url}",
                "n_hablantes_fuente": 1, "texto_oficial": fila["texto"].strip(),
            })
        time.sleep(PAUSA)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo", choices=["documentos", "audio"])
    ap.add_argument("--rehacer", action="store_true", help="volver a bajar aunque el archivo ya exista")
    args = ap.parse_args()
    REHACER = args.rehacer

    partes = []
    if args.solo in (None, "documentos"):
        print("Documentos (Wikimedia Commons):")
        partes.append(bajar_documentos())
    if args.solo in (None, "audio"):
        print("Audio (Wikimedia Commons + OpenSLR 73):")
        partes.append(bajar_audio())

    nuevas = pd.concat(partes, ignore_index=True)
    nuevas["fecha_descarga"] = date.today().isoformat()
    if FUENTES.exists() and args.solo:   # si se bajó solo una modalidad, conservar la otra
        viejas = pd.read_csv(FUENTES)
        modalidad = "documento" if args.solo == "documentos" else "audio"
        nuevas = pd.concat([viejas[viejas["modalidad"] != modalidad], nuevas], ignore_index=True)
    FUENTES.parent.mkdir(parents=True, exist_ok=True)
    nuevas.to_csv(FUENTES, index=False)
    print(f"\n{len(nuevas)} archivos registrados en {FUENTES.relative_to(CARPETA)}")
    print(nuevas.groupby(["modalidad", "tipo"]).size().to_string())
