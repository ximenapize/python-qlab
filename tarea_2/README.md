# Voces y documentos peruanos: del archivo escaneado al texto buscable

**Pregunta guía:** ¿podemos convertir documentos escaneados y entrevistas en audio peruanos en un archivo de texto estructurado y buscable?

**Dashboard público:** _(pegar aquí el enlace de Streamlit Community Cloud)_ → `https://…streamlit.app`

Tarea 2 del curso Data Science con Python ([enunciado](https://github.com/d2cml-ai/Data-Science-Python/issues/190)). Trabajo individual de Ximena Pinillos.

## Qué hace

1. **OCR** de 25 documentos peruanos con **PaddleOCR 3.7** (PP-OCRv6, `lang="es"`).
2. **Transcripción** de 50 minutos de audio en español peruano con **faster-whisper** (large-v3-turbo), **VAD Silero** y **diarización** (WeSpeaker ResNet34), con marcas `[HABLANTE_N]`.
3. **Seudonimización** de nombres, DNI, celulares y cuentas `@` en todos los textos publicados (`[PERSONA_N]`).
4. **Evaluación** con CER y WER sobre 10 documentos y 10 clips transcritos a mano, contra una línea base y modelos alternativos (Tesseract, Qwen3-ASR).
5. **Extracción estructurada** con un LLM local (`qwen2.5:7b` en Ollama) a JSON, con auditoría humana de 20 casos.
6. **Dashboard** en Streamlit que lee solo los CSV: no ejecuta modelos.

## Corpus, fuentes y licencias

Fecha de extracción de todo el corpus: **25 de octubre de 2026**. La atribución archivo por archivo (URL, licencia, autor y cómo recuperarlo) está en [`datos/fuentes.csv`](datos/fuentes.csv).

| Corpus | Archivos | Fuente | Licencias |
|---|---|---|---|
| Documentos impresos | 8 | Wikimedia Commons (El Comercio 1839, 1842 y 1904; Mercurio Peruano 1791; Amauta 1926) | Dominio público |
| Fotos con sombras o inclinación | 8 | Wikimedia Commons (carteles, placas y pizarras de Lima, Islay, Junín y Pachacámac) | CC BY-SA 3.0/4.0, CC BY 3.0, dominio público |
| Manuscritos | 9 | Wikimedia Commons (2), Archivo José Carlos Mariátegui (1) y 6 fotos propias | Dominio público, CC BY 2.0, libre descarga del Archivo Mariátegui, captura propia sin datos personales |
| Entrevistas y testimonios | 17 clips (≈49 min) | Wikimedia Commons (Wikimedia Perú, WikiAcción Perú, UNMSM, Fenmucarinap, Áncash) | CC BY-SA 4.0 (14), CC BY 4.0 (3) |
| Frases leídas | 8 clips | [OpenSLR 73](https://www.openslr.org/73/), español peruano (Google) | CC BY-SA 4.0 |

**Descartes.** El cartel «Picante de cuy» (Commons) se reemplazó porque PaddleOCR no detectó ninguna línea. No hubo duplicados por hash ni audios sin voz.

## Resultados

| Modalidad | Línea base | Alternativos | Elegido |
|---|---|---|---|
| Documentos (CER, 10 docs) | PaddleOCR **0,17** | Tesseract 0,36 | PaddleOCR |
| Audio (WER, 10 clips de hasta 60 s) | faster-whisper small 0,39 (factor de tiempo real 0,36) | Qwen3-ASR 0,41 (0,16 con GPU) · Whisper turbo **0,37** (1,08) | Whisper turbo |

Acuerdo LLM-humano: _x_/20 extracciones completamente correctas (ver `02_evaluacion.ipynb`).

### Dos hallazgos

1. **En documentos, la letra a mano separa a los modelos.** En impresos y carteles los dos OCR rinden parecido; en los manuscritos fotografiados PaddleOCR mantiene un CER de 0,16–0,34 y Tesseract sube a 0,44–0,99. Por eso PaddleOCR equivoca la mitad de caracteres en total, aunque es unas 28 veces más lento (≈28 s por página en CPU). El peor documento para ambos es una pizarra de menú escrita con tiza (CER ≈ 0,66).
2. **En audio, la condición de grabación pesa más que el modelo, y el VAD puede borrar voz real.** Los tres ASR quedan a menos de 5 puntos de WER entre sí. En cambio, una misma herramienta va de WER ≈ 0 en lecturas de estudio a ≈ 0,35 en entrevistas en la calle, y a ≈ 0,85–0,92 en una fiesta patronal con música. Ahí el VAD solo detectó voz en el 18 % del recorte, y el modelo nunca recibió el 78 % de las palabras.

### Dos limitaciones

1. **Muestra pequeña.** 10 documentos y 10 recortes de hasta 60 s no permiten afirmar que diferencias de 2 a 4 puntos de WER sean reales. Además, por azar, los 4 manuscritos sorteados son fotos propias, así que no hay manuscritos históricos en la evaluación.
2. **La seudonimización es semiautomática y no alcanza al audio.** Se basa en una lista manual de nombres, una comparación difusa (la usada en la Sesión 13) y patrones de contexto. Una primera pasada dejó escapar nombres que se corrigieron a mano. Por eso las entrevistas originales no se publican en la app (en el audio se oyen los nombres): solo su transcripción seudonimizada. Los nombres de personajes históricos y públicos se conservan a propósito.

## Estructura

```
tarea_2/
├── 01_extraccion.ipynb      # Parte 1: OCR + ASR (ejecutados en el notebook), seudonimización, limpieza, EDA
├── 02_evaluacion.ipynb      # Parte 2: CER/WER, comparación de modelos, casos difíciles, LLM y auditoría
├── app.py                   # Parte 3: dashboard Streamlit (solo lee CSV)
├── requirements.txt         # dependencias de la app
├── requirements-pipeline.txt# dependencias de los notebooks (PaddleOCR, faster-whisper, etc.)
├── descargar_corpus.py      # baja el corpus a datos/raw/ y escribe datos/fuentes.csv
├── ocr.py · asr.py · llm.py · evaluacion.py · muestra_evaluacion.py · auditoria.py
├── datos/
│   ├── documentos.csv · transcripciones.csv · fuentes.csv · extraccion_llm.csv
│   ├── muestra_evaluacion.csv · metricas_modelos.csv · predicciones_evaluacion.csv · auditoria_llm.csv
│   ├── ejemplos/            # muestra pequeña y publicable para la app
│   └── raw/                 # corpus, salidas pesadas y referencias (no se versiona)
├── modelos/                 # (vacío: los pesos se descargan solos y no se versionan)
└── evidencias_ia/           # capturas de la conversación con Claude Code
```

## Cómo ejecutarlo en local

Solo la app (lo que necesita Streamlit Cloud):

```bash
cd tarea_2
python3 -m venv .venv-app && .venv-app/bin/pip install -r requirements.txt
.venv-app/bin/streamlit run app.py
```

Pipeline completo (macOS arm64, Python 3.12):

```bash
cd tarea_2
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-pipeline.txt
brew install tesseract tesseract-lang          # OCR alternativo de la Parte 2
ollama pull qwen2.5:7b                         # LLM local de la Parte 2
.venv/bin/python descargar_corpus.py           # corpus a datos/raw/ (~115 MB)
.venv/bin/python muestra_evaluacion.py         # muestra de evaluación y recortes de 60 s
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_extraccion.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_evaluacion.ipynb
.venv/bin/streamlit run app.py
```

Notas:

- Los modelos (~4 GB entre PaddleOCR, Whisper, WeSpeaker y Qwen3-ASR) se descargan solos la primera vez.
- Transcribir los 50 min en CPU toma alrededor de una hora; los notebooks retoman lo ya procesado.
- Las 6 fotos propias, la lista privada de nombres (`datos/raw/seudonimos.json`) y las referencias manuales no están en el repositorio, por privacidad.

## Despliegue

Streamlit Community Cloud (share.streamlit.io): repositorio `ximenapize/python-qlab`, rama `main`, archivo principal `tarea_2/app.py`.

## Uso de IA

El desarrollo se hizo con Claude Code. Las capturas de toda la conversación, en orden, están en [`evidencias_ia/`](evidencias_ia/). Las referencias manuales, la elección de las fuentes y la auditoría del LLM las hizo la autora.
