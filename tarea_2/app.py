"""Voces y documentos peruanos: los resultados de OCR, ASR y LLM presentados como un periódico.

    streamlit run app.py

Solo lee los CSV de datos/ y unos pocos ejemplos publicables de datos/ejemplos/. No carga ningún
modelo: todo se calculó antes en 01_extraccion.ipynb y 02_evaluacion.ipynb. El tema (serif, tinta
negra, un acento) está en .streamlit/config.toml, en la raíz del repositorio.
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
TINTA, GRIS, ACENTO = "#111111", "#8a877f", "#9b1c1c"
COLORES = {"impreso": TINTA, "foto": GRIS, "manuscrito": ACENTO,
           "Wikimedia Commons": TINTA, "OpenSLR 73": GRIS}

st.set_page_config(page_title="Voces y documentos peruanos", layout="wide")


# ───────────────────────────────────────────────────────── datos

@st.cache_data
def leer(nombre: str) -> pd.DataFrame | None:
    ruta = DATOS / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data
def cargar():
    fuentes = leer("fuentes.csv")
    meta = fuentes[["id", "descripcion", "fuente_url", "licencia"]]
    docs = leer("documentos.csv").merge(meta, on="id", how="left")
    docs["texto_ocr"] = docs["texto_ocr"].fillna("")
    docs["caracteres"] = docs["texto_ocr"].str.len()
    clips = leer("transcripciones.csv").merge(meta, on="id", how="left")
    clips["texto_asr"] = clips["texto_asr"].fillna("")
    clips["origen"] = clips["fuente_url"].map(lambda u: "OpenSLR 73" if "openslr" in str(u) else "Wikimedia Commons")
    clips["minutos"] = clips["duracion_seg"] / 60
    clips["palabras"] = clips["texto_asr"].str.replace(r"\[HABLANTE_\d+\]", "", regex=True).str.split().str.len()
    llm = leer("extraccion_llm.csv")
    campos = {} if llm is None else dict(zip(llm["id"], llm["campos_json"].map(json.loads)))
    return fuentes, docs, clips, campos


fuentes, docs, clips, campos_llm = cargar()
metricas = leer("metricas_modelos.csv")             # 02_evaluacion.ipynb
predicciones = leer("predicciones_evaluacion.csv")  # 02_evaluacion.ipynb
auditoria = leer("auditoria_llm.csv")               # 02_evaluacion.ipynb


def ejemplo(id_: str, ext: str) -> Path | None:
    ruta = EJEMPLOS / f"{id_}.{ext}"
    return ruta if ruta.exists() else None


def escapar(texto: str) -> str:
    """Que [HABLANTE_1] y [PERSONA_2] no se lean como enlaces de Markdown."""
    return texto.replace("[", "\\[").replace("]", "\\]")


def resaltar(texto: str, buscar: str) -> str:
    texto = escapar(texto)
    if not buscar:
        return texto
    return re.sub(f"({re.escape(buscar)})", r":red-background[\1]", texto, flags=re.IGNORECASE)


def metrica(modalidad: str) -> str:
    if metricas is None:
        return "—"
    m = metricas[(metricas["modalidad"] == modalidad) & (metricas["elegido"] == True)]  # noqa: E712
    return "—" if m.empty else f"{m['valor'].iloc[0]:.2f}"


def acuerdo() -> str:
    if auditoria is None:
        return "—"
    return f"{int(auditoria['completamente_correcta'].sum())}/{len(auditoria)}"


def ficha_llm(campos: dict | None):
    if not campos:
        st.caption("Sin extracción del LLM para este registro.")
        return
    for k, v in campos.items():
        if k == "modalidad":
            continue
        if isinstance(v, list):
            v = " · ".join(f"«{x}»" for x in v)
        st.markdown(f"**{k.replace('_', ' ')}:** {escapar('—' if v is None else str(v))}")


def grafico(fig):
    fig.update_layout(font_family="Georgia, serif", font_color=TINTA, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10), title_font_size=16, legend_title_text="")
    fig.update_xaxes(showgrid=False, linecolor=TINTA)
    fig.update_yaxes(gridcolor="#e6e3dc")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ───────────────────────────────────────────────────────── cabecera del periódico

st.caption("Lima, domingo 25 de octubre de 2026 · Edición única · Tarea 2, Data Science con Python")
st.title("Voces y documentos peruanos")
st.markdown("##### ¿Pueden los documentos escaneados y las entrevistas en audio convertirse en un archivo de texto buscable?")
st.divider()

# ───────────────────────────────────────────────────────── filtros

with st.sidebar:
    st.subheader("Buscar en el archivo")
    buscar = st.text_input("Palabra en los textos OCR y ASR", placeholder="Lima, fútbol, mujeres…")
    tipos = st.multiselect("Tipo de documento", ["impreso", "foto", "manuscrito"], default=["impreso", "foto", "manuscrito"])
    origenes = st.multiselect("Fuente del audio", ["Wikimedia Commons", "OpenSLR 73"], default=["Wikimedia Commons", "OpenSLR 73"])
    tope = round(float(clips["minutos"].max()) + 0.1, 1)
    rango = st.slider("Duración del clip (min)", 0.0, tope, (0.0, tope), step=0.1)
    modelos_todos = sorted(set(docs["modelo_ocr"]) | set(clips["modelo_asr"])
                           | (set(predicciones["modelo"]) if predicciones is not None else set()))
    modelos = st.multiselect("Modelo", modelos_todos, default=modelos_todos)

f_docs = docs[docs["tipo"].isin(tipos) & docs["modelo_ocr"].isin(modelos)]
f_clips = clips[clips["origen"].isin(origenes) & clips["minutos"].between(*rango) & clips["modelo_asr"].isin(modelos)]
if buscar:
    f_docs = f_docs[f_docs["texto_ocr"].str.contains(buscar, case=False, regex=False)]
    f_clips = f_clips[f_clips["texto_asr"].str.contains(buscar, case=False, regex=False)]

portada, sec_docs, sec_audio, sec_eval, sec_archivo = st.tabs(
    ["Portada", "Documentos", "Audios", "Evaluación", "Archivo y descargas"])

# ───────────────────────────────────────────────────────── portada

with portada:
    c = st.columns(5)
    c[0].metric("Documentos leídos", len(docs), help="8 impresos, 8 fotos y 9 manuscritos, con PaddleOCR")
    c[1].metric("Minutos transcritos", f"{clips['minutos'].sum():.0f}", help=f"{len(clips)} clips con faster-whisper + VAD + diarización")
    c[2].metric("CER documentos", metrica("documento"), help="Error por carácter del OCR elegido sobre 10 documentos con transcripción manual")
    c[3].metric("WER audio", metrica("audio"), help="Error por palabra del ASR elegido sobre 10 clips con transcripción manual")
    c[4].metric("Acuerdo LLM-humano", acuerdo(), help="Extracciones del LLM completamente correctas en una auditoría humana de 20 casos")
    st.divider()

    principal, lateral = st.columns([3, 2], gap="large")
    with principal:
        st.header("La letra a mano y el ruido, no el modelo, deciden cuánto se entiende")
        st.markdown(
            "Un pipeline con modelos abiertos que corren en una laptop leyó 25 documentos peruanos, desde el "
            "*Mercurio Peruano* de 1791 hasta carteles de Pachacámac y cuadernos de clase, y transcribió 50 minutos "
            "de entrevistas en calles de Lima, conferencias de dirigentes y fiestas patronales de Áncash. "
            "Con impresos nítidos y voz de estudio el error es casi nulo; con letra a mano, tiza o música de fondo, "
            "se dispara.\n\n"
            "Al medir contra transcripciones hechas a mano, PaddleOCR cometió la mitad de errores que Tesseract, "
            "sobre todo en manuscritos. En audio, tres modelos distintos quedaron a menos de cinco puntos de WER "
            "entre sí. Lo que más separó los resultados fue la grabación: en una fiesta con música, el detector de "
            "voz descartó casi ocho de cada diez palabras antes de que el modelo pudiera oírlas."
        )
        st.caption("Todos los nombres de personas privadas aparecen seudonimizados como [PERSONA_N].")
    with lateral:
        por_tipo = f_docs.groupby("tipo").agg(documentos=("id", "size"), caracteres=("caracteres", "sum")).reset_index()
        fig = px.bar(por_tipo, x="tipo", y="caracteres", color="tipo", color_discrete_map=COLORES,
                     hover_data=["documentos"], labels={"tipo": "", "caracteres": "caracteres extraídos"},
                     title="Los impresos concentran el texto leído")
        fig.update_layout(showlegend=False)
        grafico(fig)

    st.divider()
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.subheader("Cómo se hizo")
        st.markdown("OCR con **PaddleOCR 3.7**; transcripción con **faster-whisper** turbo, detector de voz "
                    "**Silero** y diarización **WeSpeaker**; extracción de campos con **qwen2.5:7b** en local. "
                    "Ningún texto salió de la computadora.")
    with col2:
        st.subheader("De dónde salen")
        st.markdown("Wikimedia Commons, el Archivo José Carlos Mariátegui, el corpus **OpenSLR 73** de español "
                    "peruano y seis fotos propias. Cada archivo tiene su licencia en la sección *Archivo*.")
    with col3:
        st.subheader("Lo que no se publica")
        st.markdown("Las fotos propias con datos personales y el audio de las entrevistas, donde se oyen nombres "
                    "reales. De ellos se muestra solo el texto seudonimizado.")

    st.divider()
    fig = px.bar(f_clips.sort_values("duracion_seg", ascending=False), x="id", y="minutos", color="origen",
                 color_discrete_map=COLORES, hover_data=["descripcion", "n_hablantes"],
                 labels={"id": "", "minutos": "minutos"},
                 title="Diecisiete entrevistas aportan casi todo el audio; las frases leídas duran segundos")
    grafico(fig)

# ───────────────────────────────────────────────────────── documentos

with sec_docs:
    st.header("Documentos")
    st.caption(f"{len(f_docs)} de {len(docs)} documentos según la búsqueda y los filtros")
    if f_docs.empty:
        st.warning("Ningún documento cumple los filtros.")
    else:
        opciones = {f"{r.id} · {r.tipo} · {r.descripcion}": r.id for r in f_docs.itertuples()}
        elegido = opciones[st.selectbox("Elige un documento", list(opciones))]
        d = docs.set_index("id").loc[elegido]
        izq, der = st.columns([2, 3], gap="large")
        with izq:
            img = ejemplo(elegido, "jpg")
            if img:
                st.image(str(img), caption=f"{d['descripcion']}. {d['fuente']}", width="stretch")
            else:
                st.caption("Imagen no incluida en esta edición: la app muestra solo unos pocos ejemplos con "
                           "licencia de publicación y sin datos personales. El texto y los metadatos están completos.")
            st.markdown(f"**Tipo:** {d['tipo']}  \n**Fuente:** {d['fuente']}  \n**Modelo:** {d['modelo_ocr']}")
            if isinstance(d["fuente_url"], str):
                st.markdown(f"[Ver el original]({d['fuente_url']})")
        with der:
            st.subheader("Texto leído por el OCR")
            with st.container(height=300):
                st.markdown(resaltar(d["texto_ocr"], buscar).replace("\n", "  \n"))
            st.subheader("Ficha del LLM")
            ficha_llm(campos_llm.get(elegido))
            if isinstance(d.get("texto_referencia"), str):
                st.subheader("Transcripción manual")
                with st.container(height=200):
                    st.markdown(escapar(d["texto_referencia"]).replace("\n", "  \n"))
            if predicciones is not None and elegido in set(predicciones["id"]):
                p = predicciones[(predicciones["id"] == elegido) & predicciones["modelo"].isin(modelos)]
                st.dataframe(p[["modelo", "error"]].rename(columns={"error": "CER"}).round(3), hide_index=True)

# ───────────────────────────────────────────────────────── audios

with sec_audio:
    st.header("Audios")
    st.caption(f"{len(f_clips)} de {len(clips)} clips según la búsqueda y los filtros")
    if f_clips.empty:
        st.warning("Ningún clip cumple los filtros.")
    else:
        opciones = {f"{r.id} · {r.minutos:.1f} min · {r.descripcion}": r.id for r in f_clips.itertuples()}
        elegido = opciones[st.selectbox("Elige un clip", list(opciones))]
        a = clips.set_index("id").loc[elegido]
        izq, der = st.columns([2, 3], gap="large")
        with izq:
            audio = ejemplo(elegido, "wav")
            if audio:
                st.audio(str(audio))
            else:
                st.caption("Audio no incluido en esta edición: en las entrevistas se oyen nombres reales que no se "
                           "pueden seudonimizar en la grabación. El original está en la fuente, con su licencia.")
            st.markdown(f"**Duración:** {a['duracion_seg']:.0f} s  \n**Hablantes detectados:** {a['n_hablantes']}  \n"
                        f"**Idioma:** {a['idioma']}  \n**Licencia:** {a['licencia']}  \n**Modelo:** {a['modelo_asr']}")
            st.markdown(f"[Escuchar el original]({a['fuente_url']})")
        with der:
            st.subheader("Transcripción con marcas de hablante")
            texto = re.sub(r"\s*(\[HABLANTE_\d+\])", r"\n\n\1", a["texto_asr"]).strip()
            with st.container(height=300):
                st.markdown(resaltar(texto, buscar))
            st.subheader("Ficha del LLM")
            ficha_llm(campos_llm.get(elegido))
            if predicciones is not None and elegido in set(predicciones["id"]):
                st.subheader("Tres modelos, los mismos 60 segundos")
                p = predicciones[(predicciones["id"] == elegido) & predicciones["modelo"].isin(modelos)]
                st.dataframe(p[["modelo", "error", "texto"]].rename(columns={"error": "WER"}).round(3),
                             hide_index=True, width="stretch")

# ───────────────────────────────────────────────────────── evaluación

with sec_eval:
    st.header("Evaluación de los modelos")
    if metricas is None:
        st.info("Las métricas aparecerán cuando 02_evaluacion.ipynb genere datos/metricas_modelos.csv.")
    else:
        m = metricas[metricas["modelo"].isin(modelos)]
        st.markdown("Error medido sobre **10 documentos y 10 clips** con transcripción manual. Todos los modelos "
                    "leyeron los mismos archivos; menos es mejor.")
        fig = px.bar(m, x="modelo", y="valor", color="elegido", facet_col="metrica", text_auto=".2f",
                     color_discrete_map={True: ACENTO, False: GRIS},
                     labels={"valor": "error", "modelo": "", "elegido": "elegido"},
                     hover_data=["segundos_por_archivo", "factor_tiempo_real"],
                     title="PaddleOCR y Whisper turbo, los elegidos, tienen el menor error de su modalidad")
        fig.update_xaxes(matches=None)
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
        grafico(fig)
        st.dataframe(m.round(3), hide_index=True, width="stretch")
    if auditoria is not None:
        st.subheader("Auditoría humana del LLM")
        st.dataframe(auditoria, hide_index=True, width="stretch")

# ───────────────────────────────────────────────────────── archivo

with sec_archivo:
    st.header("Archivo y descargas")
    st.markdown("Textos revisados y seudonimizados. Las descargas respetan la búsqueda y los filtros.")
    vista_docs = f_docs[["id", "tipo", "fuente", "texto_ocr", "modelo_ocr"]]
    st.dataframe(vista_docs, hide_index=True, width="stretch")
    st.download_button("Descargar documentos (CSV)", vista_docs.to_csv(index=False).encode("utf-8"),
                       "documentos.csv", "text/csv")
    vista_clips = f_clips[["id", "origen", "duracion_seg", "n_hablantes", "texto_asr", "modelo_asr"]]
    st.dataframe(vista_clips, hide_index=True, width="stretch")
    st.download_button("Descargar transcripciones (CSV)", vista_clips.to_csv(index=False).encode("utf-8"),
                       "transcripciones.csv", "text/csv")
    st.subheader("Fuentes y licencias")
    st.dataframe(fuentes[["id", "modalidad", "descripcion", "licencia", "fuente_url"]], hide_index=True, width="stretch")

st.divider()
st.caption("Fuentes: Wikimedia Commons, Archivo José Carlos Mariátegui, OpenSLR 73 (Google) y capturas propias. "
           "Licencias en «Archivo y descargas». Código y notebooks en github.com/ximenapize/python-qlab/tree/main/tarea_2")
