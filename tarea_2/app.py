"""Voces y documentos peruanos: explorador de los resultados de OCR, ASR y LLM (Tarea 2).

    streamlit run app.py

Solo lee los CSV de datos/ y los ejemplos publicables de datos/ejemplos/. No carga ningún modelo:
todo se calculó antes en 01_extraccion.ipynb y 02_evaluacion.ipynb.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "datos"
EJEMPLOS = DATOS / "ejemplos"
COLORES = {"impreso": "#2a78d6", "foto": "#eb6834", "manuscrito": "#0ca30c",
           "Wikimedia Commons": "#7b4fd6", "OpenSLR 73": "#898781"}

st.set_page_config(page_title="Voces y documentos peruanos", page_icon="📜", layout="wide")


# ───────────────────────────────────────────────────────── datos

@st.cache_data
def leer(nombre: str) -> pd.DataFrame | None:
    ruta = DATOS / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data
def cargar():
    fuentes = leer("fuentes.csv")
    docs = leer("documentos.csv").merge(
        fuentes[["id", "descripcion", "fuente_url", "licencia"]], on="id", how="left")
    docs["texto_ocr"] = docs["texto_ocr"].fillna("")
    docs["caracteres"] = docs["texto_ocr"].str.len()

    clips = leer("transcripciones.csv").merge(
        fuentes[["id", "descripcion", "fuente_url", "licencia"]], on="id", how="left")
    clips["texto_asr"] = clips["texto_asr"].fillna("")
    clips["origen"] = clips["fuente_url"].map(lambda u: "OpenSLR 73" if "openslr" in str(u) else "Wikimedia Commons")
    clips["minutos"] = clips["duracion_seg"] / 60
    clips["palabras"] = clips["texto_asr"].str.replace(r"\[HABLANTE_\d+\]", "", regex=True).str.split().str.len()

    llm = leer("extraccion_llm.csv")
    campos = {} if llm is None else dict(zip(llm["id"], llm["campos_json"].map(json.loads)))
    modelo_llm = None if llm is None else llm["modelo_llm"].iloc[0]
    return docs, clips, campos, modelo_llm


docs, clips, campos_llm, modelo_llm = cargar()
metricas = leer("metricas_modelos.csv")          # Parte 2: CER/WER por modelo (02_evaluacion.ipynb)
predicciones = leer("predicciones_evaluacion.csv")  # Parte 2: texto y error por archivo y modelo
auditoria = leer("auditoria_llm.csv")            # Parte 2: auditoría humana de 20 extracciones


def ejemplo(id_: str, ext: str) -> Path | None:
    ruta = EJEMPLOS / f"{id_}.{ext}"
    return ruta if ruta.exists() else None


def resaltar(texto: str, buscar: str) -> str:
    """Marca en amarillo la palabra buscada (para st.markdown)."""
    texto = texto.replace("[", "\\[").replace("]", "\\]")   # que [HABLANTE_1] no se lea como enlace
    if not buscar:
        return texto
    return re.sub(f"({re.escape(buscar)})", r":orange-background[\1]", texto, flags=re.IGNORECASE)


def mostrar_campos(campos: dict | None):
    if not campos:
        st.info("Sin extracción del LLM para este registro.")
        return
    filas = []
    for k, v in campos.items():
        if isinstance(v, list):
            v = " · ".join(f"«{x}»" for x in v)
        filas.append({"campo": k, "valor": "null" if v is None else str(v)})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")


# ───────────────────────────────────────────────────────── encabezado

st.title("Voces y documentos peruanos")
st.subheader("¿Pueden los documentos escaneados y las entrevistas en audio convertirse en un archivo de texto buscable?")
st.markdown(
    "Pipeline de IA con modelos abiertos y locales: **PaddleOCR** lee 25 documentos peruanos (impresos, fotos y "
    "manuscritos), **faster-whisper + VAD + diarización** transcribe 50 minutos de entrevistas y lecturas en "
    "español peruano, y un **LLM local (qwen2.5:7b)** extrae campos estructurados. Todos los textos están "
    "**seudonimizados** (`[PERSONA_N]`). Esta app solo muestra resultados calculados antes en los notebooks; "
    "no ejecuta modelos."
)


def valor_metrica(modalidad: str) -> str:
    if metricas is None:
        return "—"
    m = metricas[(metricas["modalidad"] == modalidad) & (metricas["elegido"] == True)]   # noqa: E712
    return "—" if m.empty else f"{m['valor'].iloc[0]:.2f}"


def valor_auditoria() -> str:
    if auditoria is None:
        return "—"
    ok = int(auditoria["completamente_correcta"].sum())
    return f"{ok}/{len(auditoria)} ({100 * ok / len(auditoria):.0f}%)"


PENDIENTE = "«—» = pendiente: se calcula en 02_evaluacion.ipynb"
c = st.columns(5)
c[0].metric("Documentos", len(docs), help="Documentos procesados con OCR: 8 impresos, 8 fotos y 9 manuscritos")
c[1].metric("Min. de audio", f"{clips['minutos'].sum():.1f}", help=f"Minutos transcritos en {len(clips)} clips")
c[2].metric("CER docs", valor_metrica("documento"), help="Error por carácter del OCR elegido sobre 10 documentos con transcripción manual. " + PENDIENTE)
c[3].metric("WER audio", valor_metrica("audio"), help="Error por palabra del ASR elegido sobre 10 clips con transcripción manual. " + PENDIENTE)
c[4].metric("Acuerdo LLM", valor_auditoria(), help="Extracciones del LLM completamente correctas en la auditoría humana de 20 casos. " + PENDIENTE)

# ───────────────────────────────────────────────────────── filtros (barra lateral)

with st.sidebar:
    st.header("Buscar y filtrar")
    buscar = st.text_input("Palabra en los textos OCR y ASR", placeholder="p. ej. Lima, fútbol, mujeres")
    tipos = st.multiselect("Tipo de documento", ["impreso", "foto", "manuscrito"], default=["impreso", "foto", "manuscrito"])
    origenes = st.multiselect("Fuente del audio", ["Wikimedia Commons", "OpenSLR 73"], default=["Wikimedia Commons", "OpenSLR 73"])
    dur_max = float(clips["minutos"].max())
    rango = st.slider("Duración del clip (min)", 0.0, round(dur_max + 0.1, 1), (0.0, round(dur_max + 0.1, 1)), step=0.1)
    modelos_disponibles = sorted(set(docs["modelo_ocr"]) | set(clips["modelo_asr"])
                                 | (set(predicciones["modelo"]) if predicciones is not None else set()))
    modelos = st.multiselect("Modelo", modelos_disponibles, default=modelos_disponibles)

f_docs = docs[docs["tipo"].isin(tipos) & docs["modelo_ocr"].isin(modelos)]
f_clips = clips[clips["origen"].isin(origenes) & clips["minutos"].between(*rango) & clips["modelo_asr"].isin(modelos)]
if buscar:
    f_docs = f_docs[f_docs["texto_ocr"].str.contains(buscar, case=False, regex=False)]
    f_clips = f_clips[f_clips["texto_asr"].str.contains(buscar, case=False, regex=False)]
st.caption(f"Filtros activos: {len(f_docs)} de {len(docs)} documentos · {len(f_clips)} de {len(clips)} clips")

tab_res, tab_docs, tab_audio, tab_eval, tab_datos = st.tabs(
    ["Resumen", "Documentos", "Audios", "Evaluación de modelos", "Datos y descarga"])

# ───────────────────────────────────────────────────────── resumen

with tab_res:
    col1, col2 = st.columns(2)
    por_tipo = f_docs.groupby("tipo").agg(documentos=("id", "size"), caracteres=("caracteres", "sum")).reset_index()
    fig = px.bar(por_tipo, x="tipo", y="documentos", color="tipo", color_discrete_map=COLORES,
                 hover_data=["caracteres"], title="Documentos por tipo")
    fig.update_layout(showlegend=False)
    col1.plotly_chart(fig, width="stretch")

    fig = px.bar(f_clips.sort_values("duracion_seg", ascending=False), x="id", y="minutos", color="origen",
                 color_discrete_map=COLORES, hover_data=["descripcion", "n_hablantes"],
                 title="Duración por clip (min)")
    col2.plotly_chart(fig, width="stretch")

    largo = pd.concat([
        pd.DataFrame({"modalidad": "Documentos (caracteres OCR)", "longitud": f_docs["caracteres"], "grupo": f_docs["tipo"]}),
        pd.DataFrame({"modalidad": "Audio (palabras ASR)", "longitud": f_clips["palabras"], "grupo": f_clips["origen"]}),
    ])
    fig = px.histogram(largo, x="longitud", color="grupo", facet_col="modalidad", nbins=20, color_discrete_map=COLORES,
                       title="Longitud de los textos extraídos")
    fig.update_xaxes(matches=None)
    st.plotly_chart(fig, width="stretch")

# ───────────────────────────────────────────────────────── documentos

with tab_docs:
    if f_docs.empty:
        st.warning("Ningún documento cumple los filtros.")
    else:
        opciones = {f"{r.id} · {r.tipo} · {r.descripcion}": r.id for r in f_docs.itertuples()}
        elegido = opciones[st.selectbox("Documento", list(opciones))]
        d = docs.set_index("id").loc[elegido]
        izq, der = st.columns([2, 3])
        with izq:
            img = ejemplo(elegido, "jpg")
            if img:
                st.image(str(img), caption=f"{d['descripcion']} — {d['fuente']}", width="stretch")
            else:
                st.info("Imagen no publicada (captura propia con posibles datos personales). "
                        "Se muestran solo el texto seudonimizado y sus metadatos.")
            st.markdown(f"**Tipo:** {d['tipo']}  \n**Fuente:** {d['fuente']}  \n**Modelo:** {d['modelo_ocr']}")
            if isinstance(d["fuente_url"], str):
                st.markdown(f"[Ver en la fuente original]({d['fuente_url']})")
        with der:
            st.subheader("Texto OCR")
            with st.container(height=320):
                st.markdown(resaltar(d["texto_ocr"], buscar).replace("\n", "  \n"))
            st.subheader("Campos extraídos por el LLM")
            mostrar_campos(campos_llm.get(elegido))
            if modelo_llm:
                st.caption(modelo_llm)
            if isinstance(d.get("texto_referencia"), str) and d["texto_referencia"]:
                st.subheader("Transcripción manual (referencia)")
                st.markdown(d["texto_referencia"].replace("\n", "  \n"))
            if predicciones is not None:
                p = predicciones[predicciones["id"] == elegido]
                if not p.empty:
                    st.subheader("CER por modelo en este documento")
                    st.dataframe(p[["modelo", "error"]].rename(columns={"error": "CER"}).round(3), hide_index=True)

# ───────────────────────────────────────────────────────── audios

with tab_audio:
    if f_clips.empty:
        st.warning("Ningún clip cumple los filtros.")
    else:
        opciones = {f"{r.id} · {r.minutos:.1f} min · {r.descripcion}": r.id for r in f_clips.itertuples()}
        elegido = opciones[st.selectbox("Clip", list(opciones))]
        a = clips.set_index("id").loc[elegido]
        izq, der = st.columns([2, 3])
        with izq:
            audio = ejemplo(elegido, "wav")
            if audio:
                st.audio(str(audio))
            else:
                st.info("Audio no publicado: en la entrevista se escuchan nombres reales que no se pueden "
                        "seudonimizar en el audio. Se muestra la transcripción seudonimizada; el original está "
                        "en la fuente con su licencia.")
            st.markdown(f"**Duración:** {a['duracion_seg']:.0f} s  \n**Hablantes detectados:** {a['n_hablantes']}  \n"
                        f"**Idioma:** {a['idioma']}  \n**Licencia:** {a['licencia']}  \n**Modelo:** {a['modelo_asr']}")
            st.markdown(f"[Ver en la fuente original]({a['fuente_url']})")
        with der:
            st.subheader("Transcripción (con marcas de hablante)")
            texto = re.sub(r"\s*(\[HABLANTE_\d+\])", r"\n\n\1", a["texto_asr"]).strip()
            with st.container(height=320):
                st.markdown(resaltar(texto, buscar))
            st.subheader("Campos extraídos por el LLM")
            mostrar_campos(campos_llm.get(elegido))
            if predicciones is not None:
                p = predicciones[predicciones["id"] == elegido]
                if not p.empty:
                    st.subheader("Comparación de modelos en los primeros 60 s")
                    st.dataframe(p[["modelo", "error", "texto"]].rename(columns={"error": "WER"}).round(3),
                                 hide_index=True, width="stretch")

# ───────────────────────────────────────────────────────── evaluación

with tab_eval:
    if metricas is None:
        st.info("Las métricas de la Parte 2 (CER/WER sobre 10 documentos y 10 clips con transcripción manual) "
                "aparecerán aquí cuando 02_evaluacion.ipynb genere datos/metricas_modelos.csv.")
    else:
        m = metricas[metricas["modelo"].isin(modelos)]
        fig = px.bar(m, x="modelo", y="valor", color="modalidad", facet_col="modalidad", text_auto=".2f",
                     hover_data=[c for c in ["segundos", "factor_tiempo_real"] if c in m.columns],
                     title="Error por modelo sobre las mismas muestras (CER en documentos, WER en audio; menos es mejor)")
        fig.update_xaxes(matches=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(m, hide_index=True, width="stretch")
    if auditoria is not None:
        st.subheader("Auditoría humana del LLM (20 extracciones)")
        st.dataframe(auditoria, hide_index=True, width="stretch")

# ───────────────────────────────────────────────────────── datos

with tab_datos:
    st.markdown("Textos revisados y seudonimizados. La descarga respeta los filtros de la barra lateral.")
    st.subheader("Documentos")
    vista_docs = f_docs[["id", "tipo", "fuente", "texto_ocr", "modelo_ocr"]]
    st.dataframe(vista_docs, hide_index=True, width="stretch")
    st.download_button("Descargar documentos (CSV)", vista_docs.to_csv(index=False).encode("utf-8"),
                       "documentos_filtrados.csv", "text/csv")
    st.subheader("Transcripciones")
    vista_clips = f_clips[["id", "origen", "duracion_seg", "n_hablantes", "texto_asr", "modelo_asr"]]
    st.dataframe(vista_clips, hide_index=True, width="stretch")
    st.download_button("Descargar transcripciones (CSV)", vista_clips.to_csv(index=False).encode("utf-8"),
                       "transcripciones_filtradas.csv", "text/csv")
    fuentes = leer("fuentes.csv")
    st.subheader("Fuentes y licencias")
    st.dataframe(fuentes[["id", "modalidad", "descripcion", "licencia", "fuente_url"]], hide_index=True,
                 width="stretch")

st.divider()
st.caption("Tarea 2 · Data Science con Python · Fuentes: Wikimedia Commons, Archivo José Carlos Mariátegui, "
           "OpenSLR 73 (Google) y capturas propias. Licencias en la pestaña «Datos y descarga».")
