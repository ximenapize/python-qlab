"""Transcripción de los 25 clips (Sesión 13): VAD + faster-whisper + diarización + seudonimización.

    .venv/bin/python asr.py                       # transcribe lo que falte y exporta todo
    .venv/bin/python asr.py --clips clip_002 clip_018
    .venv/bin/python asr.py --solo-exportar       # no transcribe: rehace seudonimización, SRT, TXT y CSV

Basado en sessions/13-audio-a-texto/voz.py del curso (d2cml-ai/Data-Science-Python): las
funciones de carga de audio, VAD, huellas de voz, diarización, SRT y seudonimización difusa están
copiadas y adaptadas aquí (no se importa voz.py para no tocar el repo de la clase y para que
este repo se ejecute solo).

Dos etapas:
1. Transcribir (lento). Por clip guarda datos/raw/audio/salidas/clip_XXX.json con los segmentos
   SIN seudonimizar. Si el JSON ya existe, no se vuelve a transcribir.
2. Exportar (segundos). Seudonimiza y escribe clip_XXX.srt y clip_XXX.txt en la misma carpeta,
   datos/transcripciones.csv y datos/descartes_audio.csv.

Seudonimización: los nombres reales van en datos/raw/seudonimos.json (fuera de Git, porque
contiene los nombres), agrupados por persona con las variantes que escribe Whisper
(«Elsa Barrientos», «Zavarrientos»...): todas reciben el mismo [PERSONA_N]. Además se detectan nombres tras «me llamo», «mi nombre es», «señora»,
etc., DNI (8 dígitos), celulares (9 dígitos que empiezan en 9) y cuentas @usuario. Es una ayuda: hay que revisar
a mano las transcripciones y agregar al JSON los nombres que se escapen.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
SALIDAS = CARPETA / "datos" / "raw" / "audio" / "salidas"
SEUDONIMOS = CARPETA / "datos" / "raw" / "seudonimos.json"
TRANSCRIPCIONES = CARPETA / "datos" / "transcripciones.csv"
DESCARTES = CARPETA / "datos" / "descartes_audio.csv"

SR = 16_000
IDIOMA = "es"
MODELO_HF = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"   # Whisper large-v3-turbo en CTranslate2
HILOS = int(os.environ.get("ASR_HILOS", 4))                   # pocos hilos: el OCR corre en paralelo
MIN_VOZ_SEG = 1.0      # menos de 1 s de voz según el VAD -> se descarta el clip como «sin voz»
UMBRAL_HABLANTES = 0.7 # distancia coseno para separar voces (valor por defecto de voz.py)
MAX_HABLANTES = 4      # tope razonable para entrevistas cortas

# ───────────────────────────────────────────────────────── adaptado de voz.py


@dataclass
class Segmento:
    inicio: float
    fin: float
    texto: str
    hablante: str | None = None


def cargar_audio(ruta: Path) -> np.ndarray:
    """Audio -> float32 mono 16 kHz (los WAV de descargar_corpus.py ya vienen así)."""
    import soundfile as sf
    x, sr = sf.read(str(ruta), dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        from faster_whisper import decode_audio
        x = decode_audio(str(ruta), sampling_rate=SR)
    return x


def tramos_de_voz(audio: np.ndarray, silencio_min=0.5) -> list[tuple[float, float]]:
    """Silero VAD (el mismo que usa faster-whisper): tramos (inicio, fin) donde alguien habla."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opciones = VadOptions(threshold=0.5, min_silence_duration_ms=int(silencio_min * 1000), speech_pad_ms=200)
    return [(t["start"] / SR, t["end"] / SR) for t in get_speech_timestamps(audio, opciones)]


@lru_cache(maxsize=1)
def modelo_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(MODELO_HF, device="cpu", compute_type="int8", cpu_threads=HILOS)


def transcribir(audio: np.ndarray) -> list[Segmento]:
    """faster-whisper con VAD: el silencio y la música sin voz no llegan al modelo (evita alucinaciones).

    condition_on_previous_text=False corta los bucles en que Whisper repite una frase.
    """
    segs, _ = modelo_whisper().transcribe(
        audio, language=IDIOMA, beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300}, condition_on_previous_text=False)
    return [Segmento(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segs if s.text.strip()]


@lru_cache(maxsize=1)
def _modelo_de_voces():
    """WeSpeaker ResNet34: un vector de 256 números por tramo de voz (la «huella» de la voz)."""
    import torch
    from pyannote.audio import Inference, Model
    modelo = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    return Inference(modelo, window="whole", device=torch.device("cpu"))


def huellas_de_voz(audio: np.ndarray, tramos: list[tuple[float, float]]) -> np.ndarray:
    import torch
    inf = _modelo_de_voces()
    filas = []
    for a, b in tramos:
        x = audio[int(a * SR):int(b * SR)]
        if len(x) < SR // 2:  # el modelo necesita al menos medio segundo
            x = np.pad(x, (0, SR // 2 - len(x)))
        filas.append(inf({"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": SR}))
    return np.vstack(filas)


def diarizar(audio: np.ndarray, segmentos: list[Segmento], umbral=UMBRAL_HABLANTES, min_dur=1.0) -> list[Segmento]:
    """Pone HABLANTE_1, HABLANTE_2... a cada segmento (clustering jerárquico de huellas de voz).

    El número de hablantes no se fija: lo decide el umbral de distancia. Los segmentos de menos
    de 1 s no votan en el agrupamiento y después se asignan al grupo más parecido.
    """
    from sklearn.cluster import AgglomerativeClustering
    if not segmentos:
        return []
    E = huellas_de_voz(audio, [(s.inicio, s.fin) for s in segmentos])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    largos = np.array([s.fin - s.inicio >= min_dur for s in segmentos])
    if largos.sum() < 2:
        largos[:] = True
    if largos.sum() < 2:
        etiquetas = np.zeros(len(segmentos), dtype=int)
    else:
        grupo = AgglomerativeClustering(n_clusters=None, distance_threshold=umbral,
                                        metric="cosine", linkage="average").fit(E[largos])
        if grupo.n_clusters_ > MAX_HABLANTES:
            grupo = AgglomerativeClustering(n_clusters=MAX_HABLANTES, metric="cosine",
                                            linkage="average").fit(E[largos])
        centros = np.vstack([E[largos][grupo.labels_ == k].mean(axis=0) for k in range(grupo.n_clusters_)])
        etiquetas = (E @ centros.T).argmax(axis=1)
        etiquetas[largos] = grupo.labels_
    # numerar por orden de aparición: HABLANTE_1 es quien habla primero
    orden = {k: i + 1 for i, k in enumerate(dict.fromkeys(etiquetas))}
    return [Segmento(s.inicio, s.fin, s.texto, f"HABLANTE_{orden[e]}") for s, e in zip(segmentos, etiquetas)]


def _hhmmss(t: float, sep=",") -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s % 1) * 1000)):03d}"


def a_srt(segmentos: list[Segmento]) -> str:
    bloques = []
    for i, s in enumerate(segmentos, 1):
        quien = f"[{s.hablante}] " if s.hablante else ""
        bloques.append(f"{i}\n{_hhmmss(s.inicio)} --> {_hhmmss(s.fin)}\n{quien}{s.texto}\n")
    return "\n".join(bloques)


def a_turnos(segmentos: list[Segmento]) -> list[tuple[float, str | None, str]]:
    """Une segmentos seguidos del mismo hablante en un solo turno."""
    turnos = []
    for s in segmentos:
        if turnos and turnos[-1][1] == s.hablante:
            turnos[-1][2].append(s.texto)
        else:
            turnos.append([s.inicio, s.hablante, [s.texto]])
    return [(t, h, " ".join(x)) for t, h, x in turnos]


def a_txt(segmentos: list[Segmento]) -> str:
    return "\n\n".join(f"[{_hhmmss(t, '.')[:8]}] {'[' + h + '] ' if h else ''}{x}" for t, h, x in a_turnos(segmentos))


def a_texto_asr(segmentos: list[Segmento], con_hablantes: bool) -> str:
    """Texto para el CSV: una sola línea, con [HABLANTE_N] al inicio de cada turno si hay varios."""
    if not con_hablantes:
        return " ".join(s.texto for s in segmentos)
    return " ".join(f"[{h}] {x}" for _, h, x in a_turnos(segmentos))


# ───────────────────────────────────────────────────────── seudonimización

CONTEXTO_NOMBRE = re.compile(
    r"\b(?i:me llamo|mi nombre es|mi nombre|soy la señora|soy el señor|señora|señor|doña|don|"
    r"hermana|hermano|compañera|compañero)\s+((?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})")
DNI = re.compile(r"(?<!\d)\d{8}(?!\d)")
CELULAR = re.compile(r"(?<!\d)9\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")
USUARIO = re.compile(r"@\w[\w.]*(?:\s[A-Z]{2,}\w*)?")   # cuentas de redes; el OCR a veces parte el nombre («@generO FCOm»)
NO_SON_NOMBRES = {"Dios", "Perú", "Lima", "Presidente", "Presidenta", "Alcalde", "Ministro", "Ministra"}


def _reemplazo_difuso(texto: str, nombre: str, seudonimo: str, parecido=85) -> tuple[str, int]:
    """Como voz.seudonimizar, pero cuenta cuántas veces reemplazó (atrapa nombres mal escritos)."""
    from rapidfuzz import fuzz
    n = len(re.findall(r"\w+", nombre))   # «Jean-Pierre Venegas» son 3 palabras
    palabras = re.findall(r"\w+|\W+", texto)
    idx = [i for i, p in enumerate(palabras) if re.match(r"\w", p)]
    cambios = []
    for j in range(len(idx) - n + 1):
        if cambios and idx[j] <= cambios[-1][1]:
            continue
        tramo = palabras[idx[j]:idx[j + n - 1] + 1]
        if fuzz.ratio("".join(tramo).lower(), nombre.lower()) >= parecido:
            cambios.append((idx[j], idx[j + n - 1]))
    for a, b in reversed(cambios):
        palabras[a:b + 1] = [seudonimo]
    return "".join(palabras), len(cambios)


class Seudonimizador:
    """Reemplaza nombres por [PERSONA_N] (el mismo N para el mismo nombre en todo el corpus)."""

    def __init__(self, personas: list[list[str]], por_contexto: bool = True):
        self.seudonimos: dict[str, str] = {}   # variante -> [PERSONA_N]
        self.n_personas = 0
        self.por_contexto = por_contexto       # en documentos históricos se apaga: «señor Presidente» no es dato personal
        for variantes in personas:
            seudonimo = self._nuevo()
            for v in variantes:
                self.seudonimos.setdefault(v, seudonimo)

    def _nuevo(self) -> str:
        self.n_personas += 1
        return f"[PERSONA_{self.n_personas}]"

    def _seudonimo(self, nombre: str) -> str:
        if nombre not in self.seudonimos:
            self.seudonimos[nombre] = self._nuevo()
        return self.seudonimos[nombre]

    def aplicar(self, texto: str) -> tuple[str, dict[str, int]]:
        cuenta = {"nombre": 0, "dni": 0, "celular": 0, "usuario": 0}
        # 1) nombres detectados por contexto («me llamo Rosa») pasan a la lista para todo el texto
        for m in (CONTEXTO_NOMBRE.finditer(texto) if self.por_contexto else []):
            nombre = m.group(1)
            if nombre.split()[0] not in NO_SON_NOMBRES:
                self._seudonimo(nombre)
        # 2) todos los nombres conocidos, de más largo a más corto («Luz Merida» antes que «Luz»)
        for nombre in sorted(self.seudonimos, key=len, reverse=True):
            texto, k = _reemplazo_difuso(texto, nombre, self.seudonimos[nombre])
            cuenta["nombre"] += k
        texto, cuenta["dni"] = DNI.subn("[DNI]", texto)
        texto, cuenta["celular"] = CELULAR.subn("[TELEFONO]", texto)
        texto, cuenta["usuario"] = USUARIO.subn("[USUARIO]", texto)
        return texto, cuenta


def cargar_personas(clave: str = "personas") -> list[list[str]]:
    """Lista de personas; cada una es la lista de variantes de su nombre.

    clave="personas" para el audio y "personas_documentos" para el OCR.
    """
    if SEUDONIMOS.exists():
        return json.loads(SEUDONIMOS.read_text(encoding="utf-8")).get(clave, [])
    return []


# ───────────────────────────────────────────────────────── etapas


def procesar_clip(fila) -> dict:
    """Etapa 1: VAD + transcripción + diarización de un clip. Devuelve y guarda el resultado crudo."""
    ruta = CARPETA / fila.archivo
    audio = cargar_audio(ruta)
    duracion = len(audio) / SR
    t0 = time.perf_counter()
    voz = tramos_de_voz(audio)
    seg_voz = sum(b - a for a, b in voz)
    resultado = {"id": fila.id, "duracion_seg": round(duracion, 1), "segundos_voz_vad": round(seg_voz, 1),
                 "segmentos": [], "diarizado": False, "motivo_descarte": None}
    if seg_voz < MIN_VOZ_SEG:
        resultado["motivo_descarte"] = f"sin voz: el VAD detectó {seg_voz:.1f} s de voz"
    else:
        segmentos = transcribir(audio)
        resultado["seg_transcripcion"] = round(time.perf_counter() - t0, 1)
        if not segmentos:
            resultado["motivo_descarte"] = "sin voz: Whisper no devolvió texto tras el VAD"
        # diarización en las entrevistas (las frases leídas de OpenSLR tienen un solo hablante)
        elif fila.tipo == "entrevista":
            t1 = time.perf_counter()
            segmentos = diarizar(audio, segmentos)
            resultado["diarizado"] = True
            resultado["seg_diarizacion"] = round(time.perf_counter() - t1, 1)
        resultado["segmentos"] = [asdict(s) for s in segmentos]
    resultado["segundos_total"] = round(time.perf_counter() - t0, 1)
    resultado["factor_tiempo_real"] = round(resultado["segundos_total"] / duracion, 3)
    (SALIDAS / f"{fila.id}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


def descripcion_modelo(diarizado: bool) -> str:
    import faster_whisper
    base = f"faster-whisper {faster_whisper.__version__} large-v3-turbo (int8, CPU, VAD Silero on, beam 5, lang=es)"
    return base + (" + diarización WeSpeaker ResNet34 + clustering coseno" if diarizado else "")


def exportar(fuentes: pd.DataFrame):
    """Etapa 2: seudonimiza, escribe SRT/TXT por clip, transcripciones.csv y descartes_audio.csv."""
    seud = Seudonimizador(cargar_personas())
    filas, descartes, reporte = [], [], []
    clips = fuentes[fuentes["modalidad"] == "audio"]
    crudos = {f.id: json.loads((SALIDAS / f"{f.id}.json").read_text(encoding="utf-8"))
              for f in clips.itertuples() if (SALIDAS / f"{f.id}.json").exists()}
    # primera pasada: recoger nombres por contexto en todo el corpus, para reemplazarlos en todos los clips
    for r in crudos.values():
        seud.aplicar(" ".join(s["texto"] for s in r["segmentos"]))
    for f in clips.itertuples():
        r = crudos.get(f.id)
        if r is None:
            continue
        if r["motivo_descarte"]:
            descartes.append({"id": f.id, "archivo_audio": f.archivo, "motivo": r["motivo_descarte"]})
            continue
        segmentos, total = [], {"nombre": 0, "dni": 0, "celular": 0, "usuario": 0}
        for s in r["segmentos"]:
            texto, cuenta = seud.aplicar(s["texto"])
            total = {k: total[k] + cuenta[k] for k in total}
            segmentos.append(Segmento(s["inicio"], s["fin"], texto, s["hablante"]))
        n_hab = len({s.hablante for s in segmentos}) if r["diarizado"] else 1
        (SALIDAS / f"{f.id}.srt").write_text(a_srt(segmentos), encoding="utf-8")
        (SALIDAS / f"{f.id}.txt").write_text(a_txt(segmentos), encoding="utf-8")
        filas.append({
            "id": f.id, "archivo_audio": f.archivo, "duracion_seg": r["duracion_seg"], "idioma": IDIOMA,
            "texto_asr": a_texto_asr(segmentos, con_hablantes=n_hab > 1),
            "modelo_asr": descripcion_modelo(r["diarizado"]), "n_hablantes": n_hab, "texto_referencia": None,
        })
        reporte.append({"id": f.id, **{f"reemplazos_{k}": v for k, v in total.items()},
                        "factor_tiempo_real": r["factor_tiempo_real"]})
    tabla = pd.DataFrame(filas, columns=["id", "archivo_audio", "duracion_seg", "idioma", "texto_asr",
                                         "modelo_asr", "n_hablantes", "texto_referencia"])
    tabla.to_csv(TRANSCRIPCIONES, index=False)
    pd.DataFrame(descartes, columns=["id", "archivo_audio", "motivo"]).to_csv(DESCARTES, index=False)
    reporte = pd.DataFrame(reporte)
    reporte.to_csv(SALIDAS / "reporte_seudonimizacion.csv", index=False)
    return tabla, pd.DataFrame(descartes), reporte, seud


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", nargs="*", help="solo estos ids (por defecto, todos los de fuentes.csv)")
    ap.add_argument("--solo-exportar", action="store_true", help="no transcribir; solo seudonimizar y exportar")
    args = ap.parse_args()

    SALIDAS.mkdir(parents=True, exist_ok=True)
    fuentes = pd.read_csv(CARPETA / "datos" / "fuentes.csv")
    clips = fuentes[fuentes["modalidad"] == "audio"]
    if args.clips:
        clips = clips[clips["id"].isin(args.clips)]
    if not args.solo_exportar:
        for f in clips.itertuples():
            if (SALIDAS / f"{f.id}.json").exists():
                continue
            r = procesar_clip(f)
            print(f"  {f.id}: {r['duracion_seg']:6.1f} s de audio -> {r['segundos_total']:6.1f} s "
                  f"(x{r['factor_tiempo_real']}) {'diarizado' if r['diarizado'] else ''} "
                  f"{r['motivo_descarte'] or ''}", flush=True)
    tabla, descartes, reporte, seud = exportar(fuentes)
    print(f"\n{len(tabla)} clips en {TRANSCRIPCIONES.relative_to(CARPETA)} · {len(descartes)} descartados")
    print(f"Entidades reemplazadas: {int(reporte.filter(like='reemplazos_').sum().sum()) if len(reporte) else 0} "
          f"({seud.n_personas} personas distintas seudonimizadas)")
