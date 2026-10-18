# 🗞️ Tarea 1 – Noticias peruanas: del titular al clasificador

## 1. Pregunta del proyecto

> **¿Pueden convertir titulares peruanos en un clasificador de noticias?**

Sí. Con 400 titulares reales y un Transformer en español (BETO) ajustado durante 3 épocas, el clasificador alcanza **F1 macro = 0.93** sobre un test reservado de 60 titulares, frente a **0.60** de un modelo zero-shot sin entrenamiento. Además, rinde igual que un LLM de 7B parámetros pero es ~300 veces más rápido.

| Modelo | F1 macro (test, 60) | Latencia por titular |
| :--- | :---: | :---: |
| Zero-shot `Recognai/bert-base-spanish-wwm-cased-xnli` | 0.60 *(40 titulares del test)* | ~70 ms |
| LLM débil `qwen2.5:7b` (Ollama, local) | 0.93 | ~3.4 s |
| **BETO fine-tuned** `dccuchile/bert-base-spanish-wwm-cased` | **0.93** | ~8–13 ms |

Auditoría humana de 20 etiquetas del LLM: **18/20 = 90 % de acuerdo** (los 2 desacuerdos son titulares economía↔política).

## 2. Medios y fechas de extracción

- **Opción C – Comparada:** 200 titulares de **La República** + 200 de **El Comercio** = 400, con 50 por categoría y medio (`politica`, `economia`, `deportes`, `ciencia_tecnologia`), elegidos al azar (`random_state=42`).
- **Fecha de descarga:** 18 de octubre de 2026.
- **El Comercio:** sitemaps de noticias por sección (`/sitemap/news/<seccion>/`); publicaciones del 6 al 18 de octubre de 2026.
- **La República:** RSS oficiales por sección (`/rss/<seccion>.xml`). Como cada RSS expone solo las últimas 10–20 notas y `robots.txt` prohíbe paginar y usar el buscador, se recuperaron **copias históricas de esos RSS en la Wayback Machine** (máx. 12 por sección); publicaciones de diciembre 2024 a octubre 2026.
- Buenas prácticas: `robots.txt` revisado, `User-Agent` descriptivo, pausas de 1–2 s, reintentos limitados, respaldo crudo en `datos/raw/` y validación HTTP de las 400 URLs.

## 3. Hallazgos analíticos

1. **La categoría económica es la frontera difícil.** Deportes se clasifica perfecto (vocabulario muy propio: clubes, "fecha, hora y canal"), mientras que economía es la categoría con menor F1: las medidas de gobierno con impacto económico (eliminación de feriados, Fonavi, CTS) se confunden con política, tanto en BETO como en el LLM (12 de los 100 titulares de economía fueron etiquetados como política por Qwen).
2. **La República titula más largo que El Comercio** (mediana de 106 vs. 98 caracteres), y un fine-tuning pequeño iguala al LLM: con solo 280 titulares de entrenamiento BETO logra el mismo F1 que Qwen 2.5 7B, con un modelo 10 veces más liviano y sin servidor externo.

### Bonificación (`03_bonus_semantico_sesgo.ipynb`)

- **Buscador semántico:** embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`) y similitud coseno; recupera noticias del mismo tema aunque no compartan palabras. Disponible en el dashboard y también bajo el probador ("noticias parecidas").
- **Cómo titula cada diario (opción C):**
  - *Palabras características por sección* (log-odds): en economía, La República titula sobre el bolsillo (*retiro, Fonavi, aportes*) y El Comercio sobre los mercados (*Wall Street, inversión, banca*); en deportes, La República sigue a la selección y el fútbol peruano y El Comercio los resultados y transmisiones (*video, goles, LaLiga*).
  - *¿Se adivina el diario por el titular?* TF-IDF + regresión logística acierta el **66 %** (azar 50 %): estilo propio moderado, en parte confundido con la diferencia de época.
  - *Misma noticia, dos diarios:* 13 pares de la misma noticia en ambos medios (similitud ≥ 0.82, mejor coincidencia mutua, ≤ 3 días). El clasificador asigna **la misma categoría en 13/13**; lo que cambia es la **confianza** cuando el encuadre difiere: la flor andina "bautizada en honor a Lionel Messi" (El Comercio, deportes 96 %) vs. "científicos descubren una flor albiceleste" (La República, deportes 77 %). En el test, el F1 es igual por medio (0.93).

## 4. Limitaciones técnicas

1. **Ventanas temporales distintas entre medios.** El Comercio cubre ~12 días y La República ~22 meses (por depender de Wayback). Parte de la diferencia entre medios puede deberse a la época y no a la redacción; además las etiquetas vienen de la sección editorial, que a veces no coincide con el tema (p. ej. notas de minería ilegal publicadas en *Tecnología/Ecología*).
2. **Test pequeño y modelo fuera del repositorio.** Con 60 titulares en test, un error cambia ~1.7 puntos de accuracy, así que la diferencia BETO vs. Qwen no es significativa. Los pesos del modelo (~420 MB) no se suben a GitHub (límite de 100 MB), por lo que hay que re-entrenar con `02_clasificacion.ipynb` (≈1–2 min en Apple M2) antes de usar el probador. La "confianza" que reporta el LLM es casi siempre 0.95 y no sirve como probabilidad calibrada.

## 5. Cómo ejecutar en local

```bash
cd tarea_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. `01_recoleccion.ipynb` – reconstruye `datos/noticias.csv` desde `datos/raw/` (con `DESCARGAR = True` vuelve a descargar).
2. `02_clasificacion.ipynb` – zero-shot, etiquetas LLM (reutiliza `datos/etiquetas_llm.csv`; para regenerarlas: instalar [Ollama](https://ollama.com) y `ollama pull qwen2.5:7b`), entrenamiento y evaluación. Guarda el modelo en `modelo/`.
3. `03_bonus_semantico_sesgo.ipynb` – embeddings, buscador semántico y pares LR vs. EC (necesita el modelo de la Parte 2).
4. Dashboard (pestañas: probador en vivo, explorar datos, buscador semántico, sesgo LR vs. EC y modelo):

```bash
streamlit run app.py
```

## Estructura

```text
tarea_1/
├── 01_recoleccion.ipynb      # Scraping, limpieza y EDA
├── 02_clasificacion.ipynb    # Zero-shot, LLM, fine-tuning y evaluación
├── 03_bonus_semantico_sesgo.ipynb  # Buscador semántico y sesgo LR vs. EC
├── app.py                    # Dashboard Streamlit
├── scraping.py               # Funciones de descarga (RSS, sitemap, Wayback)
├── requirements.txt
├── datos/
│   ├── noticias.csv          # 400 titulares, 7 columnas
│   ├── etiquetas_llm.csv     # Etiquetas de Qwen 2.5 7B
│   ├── auditoria_llm_20.xlsx # Auditoría humana (también en .csv)
│   ├── embeddings_titulares.npy, pares_misma_noticia.csv, predicciones_test.csv  # Bonus
│   └── raw/                  # Respaldos crudos XML + caché de validación (bonus/ = RSS home LR)
└── modelo/                   # config, tokenizer y metricas.json (pesos ignorados)
```
