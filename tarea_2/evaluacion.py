"""Parte 2: corre cada modelo sobre la muestra de evaluación y guarda sus predicciones. Lo usa 02_evaluacion.ipynb.

    .venv/bin/python evaluacion.py linea_base     # PaddleOCR (config. base) + faster-whisper small en CPU
    .venv/bin/python evaluacion.py alternativos   # Tesseract + Qwen3-ASR 0.6B (+ Whisper turbo de la Parte 1)

Todos los modelos leen exactamente los mismos archivos: los 10 documentos y los 10 recortes de
audio de datos/muestra_evaluacion.csv. Las configuraciones están fijas aquí y se definieron antes
de ver ninguna métrica.

Salida: datos/raw/evaluacion/predicciones.csv (id, modalidad, modelo, texto, segundos,
duracion_seg). Va en datos/raw/ porque el texto aún no está seudonimizado.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

CARPETA = Path(__file__).resolve().parent
MUESTRA = CARPETA / "datos" / "muestra_evaluacion.csv"
SALIDA = CARPETA / "datos" / "raw" / "evaluacion"
PREDICCIONES = SALIDA / "predicciones.csv"
SR = 16_000

# ───────────────────────────────────────────────────────── OCR


def ocr_paddle(rutas: list[Path]) -> list[tuple[str, float]]:
    """Línea base: PaddleOCR tal como en la Parte 1 (PP-OCRv6_medium, lang=es, CPU, sin preprocesamiento)."""
    import ocr
    motor = ocr.crear_ocr()
    salida = []
    for r in rutas:
        res = ocr.leer(r, motor)
        salida.append((ocr.texto(res), res["segundos"]))
    return salida


def ocr_tesseract(rutas: list[Path]) -> list[tuple[str, float]]:
    """Alternativo: Tesseract 5 (LSTM), lang=spa, segmentación automática de página (--psm 3), sin preprocesamiento."""
    import pytesseract
    from PIL import Image
    salida = []
    for r in rutas:
        t0 = time.perf_counter()
        texto = pytesseract.image_to_string(Image.open(r), lang="spa", config="--oem 1 --psm 3")
        salida.append(("\n".join(l for l in texto.splitlines() if l.strip()), round(time.perf_counter() - t0, 2)))
    return salida


# ───────────────────────────────────────────────────────── ASR


def _whisper(rutas: list[Path], modelo: str) -> list[tuple[str, float]]:
    """faster-whisper en CPU (int8) con la misma decodificación de la Parte 1: VAD on, beam 5, lang=es."""
    from faster_whisper import WhisperModel
    m = WhisperModel(modelo, device="cpu", compute_type="int8", cpu_threads=min(8, os.cpu_count() or 4))
    salida = []
    for r in rutas:
        x, _ = sf.read(r, dtype="float32")
        t0 = time.perf_counter()
        segs, _ = m.transcribe(x, language="es", beam_size=5, vad_filter=True,
                               vad_parameters={"min_silence_duration_ms": 300}, condition_on_previous_text=False)
        texto = " ".join(s.text.strip() for s in segs)
        salida.append((texto, round(time.perf_counter() - t0, 2)))
    return salida


def asr_whisper_small(rutas):
    """Línea base del enunciado: faster-whisper small en CPU."""
    return _whisper(rutas, "small")


def asr_whisper_turbo(rutas):
    """El modelo de la Parte 1 (large-v3-turbo), sobre los mismos recortes."""
    import asr
    return _whisper(rutas, asr.MODELO_HF)


def asr_qwen3(rutas: list[Path]) -> list[tuple[str, float]]:
    """Alternativo: Qwen3-ASR 0.6B (Alibaba, Apache 2.0) en la GPU del Mac (MPS), language=es.

    Igual que voz.py: el VAD corta el audio en tramos de voz de hasta 25 s y se transcribe cada uno
    (así el silencio no llega al modelo, como con Whisper). El tiempo incluye el VAD.
    """
    import torch
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    rid = "Qwen/Qwen3-ASR-0.6B-hf"
    proc = AutoProcessor.from_pretrained(rid)
    m = Qwen3ASRForConditionalGeneration.from_pretrained(rid, dtype=dtype).to(dev)

    def una(x):
        with torch.inference_mode():
            inp = proc.apply_transcription_request(x, language="es").to(dev, dtype=dtype)
            out = m.generate(**inp, max_new_tokens=448)
        return proc.decode(out[0, inp["input_ids"].shape[1]:], return_format="transcription_only").strip()

    una(np.zeros(SR, dtype=np.float32))   # calentamiento: la primera llamada en MPS compila kernels
    opciones = VadOptions(threshold=0.5, min_silence_duration_ms=300, max_speech_duration_s=25, speech_pad_ms=200)
    salida = []
    for r in rutas:
        x, _ = sf.read(r, dtype="float32")
        t0 = time.perf_counter()
        tramos = get_speech_timestamps(x, opciones)
        texto = " ".join(t for t in (una(x[s["start"]:s["end"]]) for s in tramos) if t)
        salida.append((texto, round(time.perf_counter() - t0, 2)))
    return salida


MODELOS = {
    # nombre: (modalidad, función, papel)
    "PaddleOCR PP-OCRv6 (base)": ("documento", ocr_paddle, "línea base"),
    "Tesseract 5 (spa)": ("documento", ocr_tesseract, "alternativo"),
    "faster-whisper small (CPU)": ("audio", asr_whisper_small, "línea base"),
    "faster-whisper large-v3-turbo (CPU)": ("audio", asr_whisper_turbo, "modelo de la Parte 1"),
    "Qwen3-ASR 0.6B (MPS)": ("audio", asr_qwen3, "alternativo"),
}
GRUPOS = {
    "linea_base": ["PaddleOCR PP-OCRv6 (base)", "faster-whisper small (CPU)"],
    "alternativos": ["Tesseract 5 (spa)", "Qwen3-ASR 0.6B (MPS)", "faster-whisper large-v3-turbo (CPU)"],
}


def correr(nombre: str) -> pd.DataFrame:
    """Corre un modelo sobre su modalidad de la muestra y actualiza predicciones.csv."""
    modalidad, funcion, _ = MODELOS[nombre]
    muestra = pd.read_csv(MUESTRA)
    muestra = muestra[muestra["modalidad"] == modalidad]
    rutas = [CARPETA / a for a in muestra["archivo_evaluacion"]]
    resultados = funcion(rutas)
    filas = pd.DataFrame({
        "id": muestra["id"].values, "modalidad": modalidad, "modelo": nombre,
        "texto": [t for t, _ in resultados], "segundos": [s for _, s in resultados],
        "duracion_seg": [round(sf.info(r).duration, 2) if modalidad == "audio" else None for r in rutas],
    })
    SALIDA.mkdir(parents=True, exist_ok=True)
    if PREDICCIONES.exists():
        viejas = pd.read_csv(PREDICCIONES)
        filas = pd.concat([viejas[viejas["modelo"] != nombre], filas], ignore_index=True)
    filas.to_csv(PREDICCIONES, index=False)
    return filas[filas["modelo"] == nombre]


if __name__ == "__main__":
    for nombre in GRUPOS[sys.argv[1] if len(sys.argv) > 1 else "linea_base"]:
        t0 = time.perf_counter()
        r = correr(nombre)
        extra = (f" · factor de tiempo real {r['segundos'].sum() / r['duracion_seg'].sum():.3f}"
                 if r["modalidad"].iloc[0] == "audio" else "")
        print(f"{nombre}: {len(r)} archivos en {time.perf_counter() - t0:.0f} s{extra}", flush=True)
