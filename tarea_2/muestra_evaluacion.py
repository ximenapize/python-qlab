"""Muestra de evaluación de la Parte 2: 10 documentos y 10 clips, al azar estratificado.

    .venv/bin/python muestra_evaluacion.py

Se sortea SOLO con los metadatos de datos/fuentes.csv (tipo, fuente, descripción), sin mirar
ningún resultado de OCR ni de ASR, con una semilla fija. Escribe datos/muestra_evaluacion.csv.

Las transcripciones manuales van en datos/raw/referencias_manuales.xlsx (fuera de Git: se
transcribe literal, con nombres reales; se seudonimizan antes de publicar).

- Documentos: 3 impresos, 3 fotos y 4 manuscritos (proporcional a 8/8/9).
- Clips: estratos por fuente y condición de grabación (ver CONDICION). Los clips de más de
  RECORTE_SEG se recortan a sus primeros RECORTE_SEG segundos en datos/raw/audio/evaluacion/;
  ese mismo recorte es el que se transcribe a mano y el que se pasa a todos los modelos.
"""
from pathlib import Path

import pandas as pd
import soundfile as sf

CARPETA = Path(__file__).resolve().parent
SEMILLA = 2026
RECORTE_SEG = 60
EVAL_AUDIO = CARPETA / "datos" / "raw" / "audio" / "evaluacion"

N_DOCS = {"impreso": 3, "foto": 3, "manuscrito": 4}

# condición de grabación de cada clip, según la descripción de la fuente (no según el ASR)
CONDICION = {
    **{c: "lectura en estudio (OpenSLR 73)" for c in [f"clip_{i:03d}" for i in range(18, 26)]},
    "clip_001": "calle o marcha", "clip_002": "calle o marcha", "clip_013": "calle o marcha",
    "clip_014": "calle o marcha", "clip_015": "calle o marcha",
    "clip_003": "conferencia de prensa", "clip_004": "conferencia de prensa",
    "clip_005": "testimonio", "clip_006": "testimonio", "clip_007": "testimonio", "clip_008": "testimonio",
    "clip_009": "entrevista de estudiante (UNMSM)", "clip_010": "entrevista de estudiante (UNMSM)",
    "clip_011": "entrevista de estudiante (UNMSM)", "clip_012": "entrevista de estudiante (UNMSM)",
    "clip_016": "fiesta patronal (Áncash)", "clip_017": "fiesta patronal (Áncash)",
}
N_CLIPS = {"lectura en estudio (OpenSLR 73)": 3, "calle o marcha": 2, "testimonio": 2,
           "conferencia de prensa": 1, "entrevista de estudiante (UNMSM)": 1, "fiesta patronal (Áncash)": 1}


def sortear() -> pd.DataFrame:
    fuentes = pd.read_csv(CARPETA / "datos" / "fuentes.csv")
    docs = fuentes[fuentes["modalidad"] == "documento"]
    m_docs = pd.concat([docs[docs["tipo"] == t].sample(n, random_state=SEMILLA) for t, n in N_DOCS.items()])
    m_docs = m_docs.assign(estrato=m_docs["tipo"], archivo_evaluacion=m_docs["archivo"], recorte="página completa")

    clips = fuentes[fuentes["modalidad"] == "audio"].assign(estrato=lambda d: d["id"].map(CONDICION))
    m_clips = pd.concat([clips[clips["estrato"] == e].sample(n, random_state=SEMILLA) for e, n in N_CLIPS.items()])

    EVAL_AUDIO.mkdir(parents=True, exist_ok=True)
    rutas, recortes = [], []
    for f in m_clips.itertuples():
        x, sr = sf.read(CARPETA / f.archivo, dtype="float32")
        dur = len(x) / sr
        if dur > RECORTE_SEG:
            ruta = EVAL_AUDIO / f"{f.id}_primeros_{RECORTE_SEG}s.wav"
            sf.write(ruta, x[: RECORTE_SEG * sr], sr)
            recortes.append(f"primeros {RECORTE_SEG} s de {dur:.0f} s")
        else:
            ruta = CARPETA / f.archivo
            recortes.append(f"completo ({dur:.0f} s)")
        rutas.append(ruta.relative_to(CARPETA).as_posix())
    m_clips = m_clips.assign(archivo_evaluacion=rutas, recorte=recortes)

    muestra = pd.concat([m_docs, m_clips])[["id", "modalidad", "estrato", "descripcion", "archivo_evaluacion", "recorte"]]
    muestra.to_csv(CARPETA / "datos" / "muestra_evaluacion.csv", index=False)
    return muestra


if __name__ == "__main__":
    print(sortear().to_string(index=False))
