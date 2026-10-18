"""Utilidades de recoleccion para la Tarea 1 (RSS, sitemaps y Wayback Machine).

Respeta robots.txt: solo usa RSS y sitemaps declarados por cada medio, con
User-Agent descriptivo y pausas de 1-2 s entre solicitudes.
"""
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; QLAB-Tarea1-Estudiante/1.0; uso academico)"}
RAW = Path(__file__).parent / "datos" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
NS = {"news": "http://www.google.com/schemas/sitemap-news/0.9",
      "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def pausa():
    time.sleep(random.uniform(1, 2))


def get(url, params=None, reintentos=1, espera=8, timeout=60):
    """GET con pausa previa y reintentos limitados."""
    for i in range(reintentos + 1):
        pausa()
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(espera * (i + 1))
    return None


def limpiar_html(txt):
    txt = re.sub(r"<[^>]+>", " ", unescape(txt or ""))
    return re.sub(r"\s+", " ", txt).strip()


def parse_rss(contenido, medio, seccion):
    filas = []
    for it in ET.fromstring(contenido).findall(".//item"):
        filas.append({
            "titular": limpiar_html(it.findtext("title")),
            "bajada": limpiar_html(it.findtext("description")),
            "medio": medio,
            "seccion_original": seccion,
            "url": (it.findtext("link") or "").strip(),
            "fecha": (it.findtext("pubDate") or "").strip(),
        })
    return filas


def parse_sitemap_news(contenido, medio, seccion):
    filas = []
    for u in ET.fromstring(contenido).findall("sm:url", NS):
        n = u.find("news:news", NS)
        if n is None:
            continue
        filas.append({
            "titular": limpiar_html(n.findtext("news:title", namespaces=NS)),
            "bajada": "",
            "medio": medio,
            "seccion_original": seccion,
            "url": (u.findtext("sm:loc", namespaces=NS) or "").strip(),
            "fecha": n.findtext("news:publication_date", namespaces=NS) or "",
        })
    return filas


def guardar_raw(nombre, contenido):
    (RAW / nombre).write_bytes(contenido)


def el_comercio(secciones=("politica", "economia", "tecnologia", "deporte-total")):
    filas = []
    for s in secciones:
        r = get(f"https://elcomercio.pe/sitemap/news/{s}/?outputType=xml")
        if r is None:
            print("fallo EC", s)
            continue
        guardar_raw(f"ec_sitemap_{s}.xml", r.content)
        f = parse_sitemap_news(r.content, "El Comercio", s)
        print("EC", s, len(f))
        filas += f
    return filas


def la_republica_actual(secciones=("politica", "economia", "deportes", "ciencia")):
    filas = []
    for s in secciones:
        r = get(f"https://larepublica.pe/rss/{s}.xml")
        if r is None:
            print("fallo LR", s)
            continue
        guardar_raw(f"lr_rss_{s}_actual.xml", r.content)
        f = parse_rss(r.content, "La Republica", s)
        print("LR actual", s, len(f))
        filas += f
    return filas


def la_republica_wayback(secciones=("politica", "economia", "deportes", "ciencia"),
                         max_snapshots=12, desde="2025"):
    """Copias historicas del RSS por seccion en la Wayback Machine (archive.org)."""
    filas = []
    for s in secciones:
        cdx = get("https://web.archive.org/cdx/search/cdx",
                  params={"url": f"larepublica.pe/rss/{s}.xml", "output": "txt",
                          "fl": "timestamp", "filter": "statuscode:200",
                          "from": desde, "collapse": "timestamp:8"},
                  reintentos=3, espera=15)
        if cdx is None:
            print("fallo CDX", s)
            continue
        ts = cdx.text.split()
        paso = max(1, len(ts) // max_snapshots)
        ts = ts[::paso][:max_snapshots]
        ok = 0
        for t in ts:
            r = get(f"https://web.archive.org/web/{t}id_/https://larepublica.pe/rss/{s}.xml",
                    reintentos=2, espera=15)
            if r is None:
                continue
            try:
                f = parse_rss(r.content, "La Republica", s)
            except ET.ParseError:
                continue
            guardar_raw(f"lr_wayback_{s}_{t}.xml", r.content)
            filas += f
            ok += 1
        print("LR wayback", s, "snapshots ok", ok, "/", len(ts))
    return filas


def cargar_raw():
    """Reconstruye las filas a partir de los respaldos crudos de datos/raw/."""
    filas = []
    for p in sorted(RAW.glob("*.xml")):
        partes = p.stem.split("_")
        try:
            if p.name.startswith("ec_sitemap_"):
                f = parse_sitemap_news(p.read_bytes(), "El Comercio", partes[2])
                fuente = "sitemap"
            elif p.name.startswith("lr_"):
                f = parse_rss(p.read_bytes(), "La Republica", partes[2])
                fuente = "wayback" if partes[1] == "wayback" else "rss"
            else:
                continue
        except ET.ParseError:
            print("XML invalido:", p.name)
            continue
        for x in f:
            x["fuente"] = fuente
            x["archivo_raw"] = p.name
        filas += f
    return filas


def seccion_desde_url(url):
    """Ruta de seccion tal como figura en la URL del diario (sin fecha ni slug).

    https://larepublica.pe/economia/precio-del-dolar/2026/10/18/slug -> economia/precio-del-dolar
    https://elcomercio.pe/deporte-total/futbol-peruano/slug-noticia/  -> deporte-total/futbol-peruano
    """
    ruta = re.sub(r"^https?://[^/]+/", "", url).strip("/").split("/")
    secc = []
    for seg in ruta[:-1]:
        if re.fullmatch(r"\d{4}", seg):
            break
        secc.append(seg)
    return "/".join(secc) if secc else ruta[0]


CACHE_PAG = RAW / "paginas_validadas.json"


def validar_url(url, cache):
    """Comprueba que la URL responda 200 y extrae la meta description (bajada).

    Resultado cacheado en datos/raw/paginas_validadas.json para no repetir peticiones.
    """
    if url in cache:
        return cache[url]
    res = {"ok": False, "bajada": ""}
    for i in range(2):  # maximo 1 reintento
        pausa()
        try:
            # Solo se lee el <head> (donde esta la meta description), no la pagina completa
            with requests.get(url, headers=UA, timeout=30, stream=True) as r:
                if r.status_code != 200:
                    continue
                html = b""
                for trozo in r.iter_content(16384):
                    html += trozo
                    if b"</head>" in html or len(html) > 400_000:
                        break
            txt = html.decode("utf-8", "ignore")
            m = (re.search(r'<meta[^>]+(?:name|property)="(?:og:)?description"[^>]+content="([^"]*)"', txt)
                 or re.search(r'<meta[^>]+content="([^"]*)"[^>]+(?:name|property)="(?:og:)?description"', txt))
            res = {"ok": True, "bajada": limpiar_html(m.group(1)) if m else ""}
            break
        except requests.RequestException:
            time.sleep(3)
    cache[url] = res
    CACHE_PAG.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    return res


def cargar_cache():
    return json.loads(CACHE_PAG.read_text()) if CACHE_PAG.exists() else {}
