"""Parte 2, paso 3: extracción estructurada con un LLM local (Ollama) sobre los textos ya seudonimizados.

    .venv/bin/python llm.py          # extrae lo que falte y escribe datos/extraccion_llm.csv

- Modelo: qwen2.5:7b en Ollama (local; ningún texto sale de la computadora).
- Entrada: texto_ocr de datos/documentos.csv y texto_asr de datos/transcripciones.csv (ya seudonimizados).
- Salida forzada con un esquema JSON: solo los campos permitidos para cada modalidad, y null cuando
  el dato no aparece. temperature=0 y semilla fija para que sea reproducible.
- Respaldo por id en datos/raw/llm/ (si existe, no se vuelve a consultar al modelo).

datos/extraccion_llm.csv: id, modalidad, campos_json, modelo_llm, fecha.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import requests

CARPETA = Path(__file__).resolve().parent
CRUDO = CARPETA / "datos" / "raw" / "llm"
SALIDA = CARPETA / "datos" / "extraccion_llm.csv"

OLLAMA = "http://localhost:11434"
MODELO = "qwen2.5:7b"
OPCIONES = {"temperature": 0, "seed": 42, "num_ctx": 8192}

TEXTO = {"type": "string"}
TEXTO_O_NULL = {"type": ["string", "null"]}
ESQUEMAS = {
    "documento": {
        "type": "object",
        "properties": {"modalidad": {"type": "string", "enum": ["documento"]}, "tema": TEXTO, "resumen": TEXTO,
                       "emisor": TEXTO_O_NULL, "fecha": TEXTO_O_NULL, "monto": TEXTO_O_NULL},
        "required": ["modalidad", "tema", "resumen", "emisor", "fecha", "monto"],
        "additionalProperties": False,
    },
    "audio": {
        "type": "object",
        "properties": {"modalidad": {"type": "string", "enum": ["audio"]}, "tema": TEXTO, "resumen": TEXTO,
                       "citas_clave": {"type": ["array", "null"], "items": TEXTO},
                       "numero_hablantes": {"type": ["integer", "null"]}},
        "required": ["modalidad", "tema", "resumen", "citas_clave", "numero_hablantes"],
        "additionalProperties": False,
    },
}

INSTRUCCIONES = """Eres un asistente que extrae datos estructurados de textos peruanos para un archivo histórico.
Responde SOLO con un objeto JSON que siga el esquema indicado. Reglas:

1. Usa únicamente información que aparece en el texto. No inventes, no completes con conocimiento externo.
2. Si un dato no aparece en el texto, escribe null. Es mejor null que una suposición.
3. "tema": de 2 a 6 palabras que digan de qué trata el texto.
4. "resumen": 1 o 2 oraciones en español, fieles al texto.
5. El texto puede tener errores de OCR o de transcripción automática; no los corrijas en los campos copiados.
6. Las marcas como [PERSONA_1], [USUARIO], [DNI] o [TELEFONO] son datos seudonimizados: cópialas tal cual
   si forman parte de un campo y nunca intentes adivinar el dato real.
"""

PISTAS = {
    "documento": """Campos de un DOCUMENTO (texto extraído por OCR de una imagen):
- "modalidad": "documento".
- "emisor": quién emite o firma el documento (institución, periódico, autor), solo si aparece escrito; si no, null.
- "fecha": la fecha que aparece escrita en el documento, copiada tal como está; si no hay, null.
- "monto": un importe de dinero que aparezca escrito (con su moneda si la tiene); si no hay, null.""",
    "audio": """Campos de un AUDIO (transcripción automática de una entrevista o una lectura):
- "modalidad": "audio".
- "citas_clave": de 1 a 3 frases copiadas LITERALMENTE de la transcripción que resuman lo más importante;
  sin las marcas [HABLANTE_N]. Si el texto es demasiado corto para tener citas relevantes, null.
- "numero_hablantes": cuántas personas distintas hablan, solo si se puede sustentar con las marcas
  [HABLANTE_1], [HABLANTE_2]... del texto (cuenta las marcas distintas). Si el texto no tiene marcas de
  hablante y no dice cuántas personas hablan, null.""",
}


def extraer(texto: str, modalidad: str) -> tuple[dict, float]:
    """Una llamada a Ollama con el esquema JSON de la modalidad. Devuelve (campos, segundos)."""
    mensajes = [
        {"role": "system", "content": INSTRUCCIONES + "\n" + PISTAS[modalidad]},
        {"role": "user", "content": f"TEXTO ({modalidad}):\n\"\"\"\n{texto}\n\"\"\"\n\nDevuelve el JSON."},
    ]
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": MODELO, "messages": mensajes, "stream": False,
        "format": ESQUEMAS[modalidad], "options": OPCIONES,
    })
    r.raise_for_status()
    campos = json.loads(r.json()["message"]["content"])
    # el esquema ya lo exige, pero se vuelve a filtrar por si acaso: ningún campo fuera de la lista
    permitidos = ESQUEMAS[modalidad]["properties"]
    campos = {k: campos.get(k) for k in permitidos}
    return campos, round(time.perf_counter() - t0, 1)


def descripcion_modelo() -> str:
    version = requests.get(f"{OLLAMA}/api/version", timeout=10).json()["version"]
    return f"{MODELO} (Ollama {version}, Q4_K_M, temperature=0, seed=42, salida con esquema JSON)"


def textos() -> pd.DataFrame:
    docs = pd.read_csv(CARPETA / "datos" / "documentos.csv")
    clips = pd.read_csv(CARPETA / "datos" / "transcripciones.csv")
    return pd.concat([
        pd.DataFrame({"id": docs["id"], "modalidad": "documento", "texto": docs["texto_ocr"].fillna("")}),
        pd.DataFrame({"id": clips["id"], "modalidad": "audio", "texto": clips["texto_asr"].fillna("")}),
    ], ignore_index=True)


def procesar(tabla: pd.DataFrame, rehacer: bool = False) -> pd.DataFrame:
    CRUDO.mkdir(parents=True, exist_ok=True)
    modelo = descripcion_modelo()
    filas = []
    for f in tabla.itertuples():
        ruta = CRUDO / f"{f.id}.json"
        if ruta.exists() and not rehacer:
            r = json.loads(ruta.read_text(encoding="utf-8"))
        else:
            campos, seg = extraer(f.texto, f.modalidad)
            r = {"id": f.id, "modalidad": f.modalidad, "campos": campos, "segundos": seg,
                 "modelo_llm": modelo, "fecha": date.today().isoformat()}
            ruta.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {f.id}: {seg} s · tema: {campos.get('tema')}", flush=True)
        filas.append({"id": r["id"], "modalidad": r["modalidad"],
                      "campos_json": json.dumps(r["campos"], ensure_ascii=False),
                      "modelo_llm": r["modelo_llm"], "fecha": r["fecha"], "segundos": r["segundos"]})
    return pd.DataFrame(filas)


# ───────────────────────────────────────────────────────── chequeos automáticos (no reemplazan la auditoría)

def _norm(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def chequeos(extraccion: pd.DataFrame, tabla: pd.DataFrame) -> pd.DataFrame:
    """¿Lo extraído aparece en el texto? Citas literales, fecha y monto presentes; hablantes = marcas distintas."""
    texto = dict(zip(tabla["id"], tabla["texto"]))
    filas = []
    for f in extraccion.itertuples():
        c, t = json.loads(f.campos_json), texto[f.id]
        fila = {"id": f.id, "modalidad": f.modalidad}
        if f.modalidad == "documento":
            for k in ["emisor", "fecha", "monto"]:
                fila[f"{k}_en_texto"] = None if c[k] is None else _norm(c[k]) in _norm(t)
        else:
            citas = c["citas_clave"] or []
            fila["citas"] = len(citas)
            fila["citas_literales"] = sum(_norm(q) in _norm(t) for q in citas)
            marcas = len(set(re.findall(r"\[HABLANTE_\d+\]", t)))
            fila["hablantes_marcados"] = marcas or None
            fila["hablantes_coinciden"] = None if c["numero_hablantes"] is None else c["numero_hablantes"] == (marcas or 1)
        filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    tabla = textos()
    extraccion = procesar(tabla)
    extraccion[["id", "modalidad", "campos_json", "modelo_llm", "fecha"]].to_csv(SALIDA, index=False)
    print(f"\n{len(extraccion)} extracciones en {SALIDA.relative_to(CARPETA)} · "
          f"{extraccion['segundos'].sum() / 60:.1f} min de LLM")
