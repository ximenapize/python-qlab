"""Dashboard Tarea 1: ¿Pueden convertir titulares peruanos en un clasificador de noticias?

Ejecutar desde la carpeta tarea_1/:  streamlit run app.py
"""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BASE = Path(__file__).parent
DATOS = BASE / "datos"
MODELO = BASE / "modelo"
EMB_MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ETIQUETAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRE = {"politica": "Política", "economia": "Economía", "deportes": "Deportes",
          "ciencia_tecnologia": "Ciencia y tecnología"}
MEDIO = {"La Republica": "La República", "El Comercio": "El Comercio"}

# Paleta de periódico: tinta, vino, pizarra y ocre
TINTA, VINO, PIZARRA, OCRE, PAPEL = "#1A1A1A", "#8B1E1E", "#5C6B7A", "#A8823F", "#FAF7F0"
COLOR_CAT = {"Política": TINTA, "Economía": OCRE, "Deportes": VINO, "Ciencia y tecnología": PIZARRA}
COLOR_MEDIO = {"La República": VINO, "El Comercio": TINTA}
SERIF = "Source Serif 4, Georgia, serif"

pio.templates["periodico"] = go.layout.Template(layout=dict(
    font=dict(family=SERIF, color=TINTA, size=13),
    title=dict(font=dict(family="Playfair Display, Georgia, serif", size=18)),
    paper_bgcolor=PAPEL, plot_bgcolor=PAPEL, colorway=[TINTA, VINO, PIZARRA, OCRE],
    xaxis=dict(showgrid=False, linecolor=TINTA, ticks="outside"),
    yaxis=dict(gridcolor="#E2DCCD", zeroline=False),
    legend=dict(orientation="h", y=1.08, x=0, title=None),
    margin=dict(l=10, r=10, t=60, b=10)))
pio.templates.default = "periodico"

st.set_page_config(page_title="AgendaPerú · Buscador y clasificador de noticias", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,800;0,900;1,600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, .stMarkdown, p, li, label, input, textarea, button {font-family: 'Source Serif 4', Georgia, serif;}
h1, h2, h3, h4 {font-family: 'Playfair Display', Georgia, serif !important; letter-spacing: -.01em;}
.block-container {padding-top: 1.2rem; max-width: 1250px;}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
.cabecera {text-align: center; border-bottom: 4px double #1A1A1A; padding-bottom: .4rem; margin-bottom: .5rem;}
.cabecera .lema {font-size: .8rem; letter-spacing: .25em; text-transform: uppercase; color: #5C6B7A;}
.cabecera .nombre {font-family: 'Playfair Display', serif; font-weight: 900; font-size: 4rem; line-height: 1; margin: .2rem 0;}
.fecha {display: flex; justify-content: space-between; border-bottom: 1px solid #1A1A1A;
        font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; padding: .25rem 0 .35rem; margin-bottom: 1rem;}
.titular-principal {font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 800; line-height: 1.15; margin: .3rem 0 .4rem;}
.bajada-principal {font-size: 1.1rem; color: #3A3A3A; font-style: italic; margin-bottom: .6rem;}
.antetitulo {font-size: .75rem; letter-spacing: .18em; text-transform: uppercase; color: #8B1E1E; font-weight: 600;}
.nota {border-top: 1px solid #1A1A1A; padding: .7rem 0 .8rem;}
.nota h4 {font-size: 1.25rem; font-weight: 800; margin: .15rem 0 .25rem; line-height: 1.25;}
.nota h4 a {color: #1A1A1A; text-decoration: none;}
.nota h4 a:hover {text-decoration: underline;}
.nota p {margin: 0 0 .3rem; color: #3A3A3A;}
.nota .meta {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #5C6B7A;}
.veredicto {border: 2px solid #1A1A1A; padding: 1rem 1.1rem; background: #F0EBDF;}
.veredicto .cat {font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 900; line-height: 1.1;}
.barra {display: flex; align-items: center; gap: .5rem; margin: .25rem 0; font-size: .9rem;}
.barra .pista {flex: 1; height: 9px; background: #E2DCCD;}
.barra .relleno {height: 9px;}
.cifras {display: grid; grid-template-columns: repeat(4, 1fr); border-top: 3px solid #1A1A1A; border-bottom: 1px solid #1A1A1A; margin: 1.2rem 0;}
.cifra {padding: .7rem .9rem; border-right: 1px solid #C9C1AE;}
.cifra:last-child {border-right: none;}
.cifra .valor {font-family: 'Playfair Display', serif; font-size: 2.3rem; font-weight: 900; line-height: 1;}
.cifra .rotulo {font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; color: #5C6B7A; margin-top: .3rem;}
.cifra .nota-cifra {font-size: .85rem; font-style: italic; color: #3A3A3A;}
.recuadro {border-left: 4px solid #8B1E1E; padding: .6rem 1rem; background: #F0EBDF; margin: .8rem 0; font-style: italic;}
div[data-testid="stForm"] {border: 2px solid #1A1A1A; border-radius: 0; background: #FFFDF8; padding: 1rem 1.2rem;}
div[data-testid="stForm"] input {font-size: 1.15rem;}
.stTabs [data-baseweb="tab-list"] {border-bottom: 1px solid #1A1A1A; gap: 1.6rem;}
.stTabs [data-baseweb="tab"] {font-family: 'Playfair Display', serif; font-size: 1.05rem; text-transform: uppercase; letter-spacing: .06em;}
button[kind="primary"], button[kind="primaryFormSubmit"] {border-radius: 0;}
@media (max-width: 800px) {.cifras {grid-template-columns: repeat(2, 1fr);} .cabecera .nombre {font-size: 2.6rem;}}
</style>
""", unsafe_allow_html=True)


# --- Carga de datos y modelos ---------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATOS / "noticias.csv")
    df["categoria"] = df["etiqueta"].map(NOMBRE)
    df["diario"] = df["medio"].map(MEDIO)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["largo_titular"] = df["titular"].str.len()
    return df


@st.cache_data
def cargar_json(ruta):
    return json.loads(ruta.read_text()) if ruta.exists() else {}


@st.cache_data
def cargar_csv(ruta):
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


@st.cache_resource
def cargar_clasificador():
    from transformers import pipeline
    return pipeline("text-classification", model=str(MODELO), tokenizer=str(MODELO), top_k=None)


@st.cache_resource
def cargar_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODELO)


@st.cache_data
def cargar_embeddings():
    ruta = DATOS / "embeddings_titulares.npy"
    return np.load(ruta) if ruta.exists() else None


def buscar_similares(texto, base, k):
    E = cargar_embeddings()
    q = cargar_encoder().encode([texto], normalize_embeddings=True)[0]
    sim = E[base.index] @ q
    orden = np.argsort(-sim)[:k]
    return base.iloc[orden].assign(similitud=sim[orden])


def nota_html(fila):
    e = html.escape
    return (f'<div class="nota"><div class="antetitulo">{e(fila.categoria)}</div>'
            f'<h4><a href="{e(fila.url)}" target="_blank">{e(fila.titular)}</a></h4>'
            f'<p>{e(str(fila.bajada))}</p>'
            f'<div class="meta">{e(fila.diario)} · {fila.fecha:%d/%m/%Y} · similitud {fila.similitud:.2f}</div></div>')


def veredicto_html(probs):
    mejor = probs.iloc[0]
    barras = "".join(
        f'<div class="barra"><span style="width:9.5rem">{r.categoria}</span>'
        f'<div class="pista"><div class="relleno" style="width:{r.probabilidad:.0%};background:{COLOR_CAT[r.categoria]}"></div></div>'
        f'<span style="width:3.2rem;text-align:right">{r.probabilidad:.0%}</span></div>'
        for r in probs.itertuples())
    aviso = ('<p style="font-size:.85rem;margin:.5rem 0 0"><i>Confianza baja: el titular podría pertenecer a más de una sección.</i></p>'
             if mejor.probabilidad < 0.6 else "")
    return (f'<div class="veredicto"><div class="antetitulo">Veredicto del clasificador</div>'
            f'<div class="cat">{mejor.categoria}</div>'
            f'<div style="margin-bottom:.6rem">Probabilidad <b>{mejor.probabilidad:.1%}</b> · BETO ajustado</div>'
            f'{barras}{aviso}</div>')


df = cargar_datos()
met = cargar_json(MODELO / "metricas.json")
pares = cargar_csv(DATOS / "pares_misma_noticia.csv")
pred_test = cargar_csv(DATOS / "predicciones_test.csv")
palabras = cargar_csv(DATOS / "palabras_caracteristicas.csv")
resumen = cargar_json(DATOS / "sesgo_resumen.json")
hay_modelo = (MODELO / "model.safetensors").exists()
hay_embeddings = cargar_embeddings() is not None

# --- Cabecera -------------------------------------------------------------------
st.markdown(f"""
<div class="cabecera">
  <div class="lema">Diplomado QLAB · Tarea 1 · Ciencia de datos con Python</div>
  <div class="nombre">AgendaPerú</div>
  <div class="lema">Buscador y clasificador de noticias</div>
</div>
<div class="fecha"><span>Lima, domingo 18 de octubre de 2026</span>
<span>{len(df)} titulares · La República y El Comercio</span><span>Edición única</span></div>
""", unsafe_allow_html=True)

# --- Filtros horizontales -------------------------------------------------------
f1, f2, f3 = st.columns([1.1, 2.2, 1.3])
diarios = f1.pills("Diario", list(MEDIO.values()), selection_mode="multi", default=list(MEDIO.values()))
secciones = f2.pills("Sección", list(NOMBRE.values()), selection_mode="multi", default=list(NOMBRE.values()))
fmin, fmax = df["fecha"].min().date(), df["fecha"].max().date()
rango = f3.date_input("Fechas de publicación", (fmin, fmax), min_value=fmin, max_value=fmax, format="DD/MM/YYYY")
filtrado = df[df["diario"].isin(diarios or []) & df["categoria"].isin(secciones or [])]
if isinstance(rango, tuple) and len(rango) == 2:
    filtrado = filtrado[filtrado["fecha"].dt.date.between(*rango)]
st.caption(f"Mostrando {len(filtrado)} de {len(df)} noticias. Los filtros se aplican al buscador y a la sección Datos.")

portada, datos, sesgo, modelo = st.tabs(["Portada", "Datos", "Cómo titula cada diario", "El modelo"])

# --- Portada: buscador + clasificador -------------------------------------------
with portada:
    st.markdown('<div class="antetitulo">La pregunta del proyecto</div>'
                '<div class="titular-principal">¿Pueden convertir titulares peruanos en un clasificador de noticias?</div>'
                f'<div class="bajada-principal">Sí: con 400 titulares de La República y El Comercio, un BETO ajustado acierta '
                f'la sección con F1 macro {met.get("f1_macro_test", float("nan")):.2f} en titulares que nunca vio. '
                'Pruébelo: escriba un titular o un tema y verá las noticias más parecidas del archivo '
                '(búsqueda por significado, no por palabras exactas) y en qué sección lo publicaría el modelo.</div>',
                unsafe_allow_html=True)
    with st.form("buscador", border=True):
        c1, c2, c3 = st.columns([5, 1, 1.1], vertical_alignment="bottom")
        consulta = c1.text_input("Titular o tema", placeholder="Ej.: Congreso aprueba nuevo retiro de fondos de AFP",
                                 label_visibility="collapsed")
        k = c2.number_input("Notas", 3, 15, 6, label_visibility="collapsed")
        enviar = c3.form_submit_button("Buscar y clasificar", type="primary", width="stretch")
    st.caption("Pruebe con: *Perú vs Uruguay por Eliminatorias* · *BCR mantiene la tasa de referencia* · "
               "*Científicos descubren una nueva especie en la Amazonía* · *Senado debate facultades legislativas*")

    if enviar and consulta.strip():
        izq, der = st.columns([2, 1], gap="large")
        with der:
            if hay_modelo:
                with st.spinner("Clasificando..."):
                    res = cargar_clasificador()(consulta.strip())[0]
                probs = (pd.DataFrame(res).rename(columns={"label": "etiqueta", "score": "probabilidad"})
                         .assign(categoria=lambda d: d["etiqueta"].map(NOMBRE))
                         .sort_values("probabilidad", ascending=False))
                st.markdown(veredicto_html(probs), unsafe_allow_html=True)
            else:
                st.error("Falta `modelo/model.safetensors`: ejecute `02_clasificacion.ipynb`.")
        with izq:
            st.markdown('<div class="antetitulo">Noticias relacionadas en el archivo</div>', unsafe_allow_html=True)
            if not hay_embeddings:
                st.error("Falta `datos/embeddings_titulares.npy`: ejecute `03_bonus_semantico_sesgo.ipynb`.")
            elif filtrado.empty:
                st.info("No hay noticias con los filtros elegidos.")
            else:
                res = buscar_similares(consulta.strip(), filtrado, int(k))
                st.markdown("".join(nota_html(r) for r in res.itertuples()), unsafe_allow_html=True)
                if res["similitud"].max() < 0.45:
                    st.caption("Similitudes bajas: el archivo no tiene noticias muy cercanas a esta consulta.")

    st.markdown(f"""
<div class="cifras">
  <div class="cifra"><div class="valor">{len(df)}</div><div class="rotulo">Noticias recolectadas</div>
    <div class="nota-cifra">200 de cada diario, 100 por sección</div></div>
  <div class="cifra"><div class="valor">{met.get('f1_macro_test', float('nan')):.2f}</div><div class="rotulo">F1 macro en test · BETO</div>
    <div class="nota-cifra">sobre 60 titulares reservados</div></div>
  <div class="cifra"><div class="valor">{met.get('acuerdo_llm_humano_pct', float('nan')):.0%}</div><div class="rotulo">Acuerdo LLM vs. auditoría humana</div>
    <div class="nota-cifra">{met.get('acuerdo_llm_humano', '')} titulares revisados a mano</div></div>
  <div class="cifra"><div class="valor">{met.get('f1_macro_zero_shot', float('nan')):.2f}</div><div class="rotulo">F1 macro zero-shot</div>
    <div class="nota-cifra">el piso a superar, sin entrenamiento</div></div>
</div>""", unsafe_allow_html=True)

    with st.expander("Metodología"):
        st.markdown(
            "1. **Recolección** (opción C): 200 titulares por diario desde RSS, sitemaps oficiales y copias históricas de "
            "los RSS en la Wayback Machine, respetando `robots.txt`, con pausas de 1–2 s y las 400 URLs validadas.\n"
            "2. **Etiquetado**: la sección editorial se mapea a política, economía, deportes o ciencia y tecnología; "
            "50 noticias por sección y diario, elegidas al azar.\n"
            "3. **Supervisión débil**: Qwen 2.5 7B (Ollama, local) etiqueta los 400 titulares; 20 se auditan a mano.\n"
            "4. **Modelado**: split estratificado 70/15/15 (`random_state=42`) y BETO ajustado 3 épocas, eligiendo la mejor "
            "por F1 macro en validación. Las métricas se reportan sobre el test reservado.\n"
            "5. **Extras**: buscador semántico con embeddings multilingües y análisis de sesgo con pares de la misma noticia.")

# --- Datos ----------------------------------------------------------------------
with datos:
    if filtrado.empty:
        st.info("No hay noticias con los filtros elegidos.")
    else:
        a, b = st.columns(2, gap="large")
        with a:
            dist = filtrado.groupby(["categoria", "diario"]).size().reset_index(name="noticias")
            fig = px.bar(dist, x="categoria", y="noticias", color="diario", barmode="group", text="noticias",
                         color_discrete_map=COLOR_MEDIO, title="Cuántas noticias hay por sección",
                         labels={"categoria": "", "noticias": "Noticias", "diario": ""})
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = px.box(filtrado, x="diario", y="largo_titular", color="diario", points="all",
                         color_discrete_map=COLOR_MEDIO, hover_data=["titular"],
                         title="La República titula más largo que El Comercio",
                         labels={"largo_titular": "Caracteres del titular", "diario": ""})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        fig = px.histogram(filtrado, x="fecha", color="diario", nbins=50, color_discrete_map=COLOR_MEDIO,
                           title="El Comercio aporta notas de octubre; La República, del histórico de sus RSS",
                           labels={"fecha": "", "diario": ""})
        fig.update_layout(yaxis_title="Noticias")
        st.plotly_chart(fig, width="stretch")

        st.markdown("### El archivo completo")
        texto = st.text_input("Filtrar por palabra en el titular", "")
        tabla = filtrado[filtrado["titular"].str.contains(texto, case=False, regex=False)] if texto else filtrado
        st.dataframe(tabla[["titular", "bajada", "diario", "categoria", "seccion_original", "fecha", "url"]],
                     width="stretch", hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("Enlace", display_text="leer nota"),
                                    "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                                    "diario": "Diario", "categoria": "Sección", "seccion_original": "Sección original"})
        st.download_button("Descargar selección en CSV",
                           tabla[["titular", "bajada", "medio", "seccion_original", "url", "fecha", "etiqueta"]]
                           .to_csv(index=False).encode("utf-8"),
                           file_name="noticias_filtradas.csv", mime="text/csv")

# --- Cómo titula cada diario ----------------------------------------------------
with sesgo:
    st.markdown('<div class="antetitulo">Análisis comparado</div>'
                '<div class="titular-principal">Cómo titula cada diario</div>'
                '<div class="bajada-principal">Mismas cuatro secciones, dos redacciones. Qué palabras prefiere cada una, '
                'si un modelo puede adivinar el diario solo con el titular y qué pasa cuando ambos cuentan la misma noticia.</div>',
                unsafe_allow_html=True)

    # 1. Palabras características
    if not palabras.empty:
        st.markdown("### Las palabras de cada diario")
        LECTURA = {
            "Economía": "La República titula sobre el bolsillo del ciudadano (retiro, Fonavi, aportes); El Comercio, sobre los mercados (Wall Street, inversión, banca).",
            "Deportes": "La República sigue al fútbol peruano y a la selección (Eliminatorias, Cueva, Ruidíaz); El Comercio, resultados y transmisiones (video, goles, LaLiga, canales).",
            "Política": "El Comercio titula desde las instituciones (facultades, Senado, comisión); La República, desde los personajes y el conflicto (Boluarte, censura, candidatos). Aquí pesa la diferencia de época.",
            "Ciencia y tecnología": "La República prefiere la ciencia y el espacio (científicos, eclipse, microplásticos); El Comercio, la tecnología y sus empresas (Google, lanza, datos).",
        }
        sec = st.segmented_control("Sección", list(LECTURA), default="Economía", label_visibility="collapsed") or "Economía"
        g = (palabras[palabras["etiqueta"].map(NOMBRE) == sec]
             .assign(diario=lambda d: d["diario"].map(MEDIO)).sort_values("log_odds"))
        a, b = st.columns([1.4, 1], gap="large")
        with a:
            fig = px.bar(g, x="log_odds", y="palabra", color="diario", orientation="h", color_discrete_map=COLOR_MEDIO,
                         hover_data={"usos_lr": True, "usos_ec": True, "log_odds": ":.2f"},
                         labels={"log_odds": "← más El Comercio · más La República →", "palabra": "", "diario": "",
                                 "usos_lr": "Usos en La República", "usos_ec": "Usos en El Comercio"},
                         title=f"{sec}: palabras más propias de cada diario")
            fig.update_layout(height=430)
            st.plotly_chart(fig, width="stretch")
        with b:
            st.markdown(f'<div class="recuadro" style="margin-top:3.2rem">{LECTURA[sec]}</div>', unsafe_allow_html=True)
            st.caption("Log-odds entre la frecuencia relativa de cada palabra en uno y otro diario, sobre los 100 titulares "
                       "de la sección (50 por diario). Se ignoran palabras vacías y de menos de 4 letras.")

    # 2. ¿Se adivina el diario?
    if resumen:
        st.markdown("### ¿Se puede adivinar el diario solo con el titular?")
        ps = resumen["acc_adivinar_diario_por_seccion"]
        mx, mn = max(ps.values()), min(ps.values())
        st.markdown(f"""
<div class="cifras">
  <div class="cifra"><div class="valor">{resumen['acc_adivinar_diario_400']:.0%}</div><div class="rotulo">Aciertos al adivinar el diario</div>
    <div class="nota-cifra">± {resumen['acc_adivinar_diario_400_std']:.0%} en 5 particiones; el azar es 50 %</div></div>
  <div class="cifra"><div class="valor">{mx:.0%}</div><div class="rotulo">Sección más reconocible</div>
    <div class="nota-cifra">{' y '.join(NOMBRE[k].lower() for k, v in ps.items() if v == mx)}</div></div>
  <div class="cifra"><div class="valor">{mn:.0%}</div><div class="rotulo">Sección menos reconocible</div>
    <div class="nota-cifra">{' y '.join(NOMBRE[k].lower() for k, v in ps.items() if v == mn)}</div></div>
  <div class="cifra"><div class="valor">{resumen['acc_adivinar_diario_ventana']:.0%}</div><div class="rotulo">Mismas fechas (5–18 oct.)</div>
    <div class="nota-cifra">solo {resumen['n_ventana'].get('La Republica', 0)} notas de La República: poco confiable</div></div>
</div>""", unsafe_allow_html=True)
        st.markdown("Un modelo sencillo (TF-IDF + regresión logística) reconoce al diario en 2 de cada 3 titulares: "
                    "cada redacción tiene un **estilo propio, pero moderado**. Parte de la señal puede venir de la época "
                    "(los titulares de La República van de 2024 a 2026 y los de El Comercio son de octubre de 2026) "
                    "y no solo de la manera de escribir.")

    # 3. Misma noticia, dos diarios
    if not pares.empty:
        st.markdown(f"### Misma noticia, dos diarios: el clasificador coincide en {pares['misma_prediccion'].sum()} de {len(pares)} casos")
        st.caption("Pares de titulares que cuentan el mismo hecho: similitud coseno ≥ 0,82, mejor coincidencia mutua y ≤ 3 días de diferencia.")
        orden = pares.assign(etq=lambda d: d["titular_ec"].str[:60] + "…").sort_values("dif_prob")
        fig = go.Figure()
        for _, r in orden.iterrows():
            fig.add_trace(go.Scatter(x=[r.prob_ec, r.prob_lr], y=[r.etq, r.etq], mode="lines",
                                     line=dict(color="#C9C1AE", width=3), showlegend=False, hoverinfo="skip"))
        for nombre, col_t, col_p, col_c in [("El Comercio", "titular_ec", "prob_ec", "pred_ec"),
                                            ("La República", "titular_lr", "prob_lr", "pred_lr")]:
            fig.add_trace(go.Scatter(x=orden[col_p], y=orden.etq, mode="markers", name=nombre,
                                     marker=dict(size=12, color=COLOR_MEDIO[nombre]),
                                     customdata=np.stack([orden[col_t], orden[col_c].map(NOMBRE)], axis=-1),
                                     hovertemplate="%{customdata[0]}<br>%{customdata[1]}: %{x:.1%}<extra></extra>"))
        fig.update_layout(title="La sección no cambia entre diarios; la confianza sí, cuando cambia el encuadre",
                          xaxis=dict(title="Probabilidad de la sección predicha", tickformat=".0%"), yaxis_title="",
                          height=140 + 36 * len(orden))
        st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="recuadro">El caso de la flor andina: El Comercio titula «…la bautizan en honor a Lionel Messi» '
                    '(deportes, 96 %) y La República «Científicos descubren una nueva especie de flor albiceleste…» '
                    '(deportes, 77 %). Misma noticia y misma sección —errónea: es ciencia—, pero 19 puntos menos de '
                    'confianza. El gancho de la celebridad arrastra al modelo.</div>', unsafe_allow_html=True)

        with st.expander("Ver los pares lado a lado"):
            vista = pares.assign(pred_ec=pares["pred_ec"].map(NOMBRE), pred_lr=pares["pred_lr"].map(NOMBRE))
            st.dataframe(vista[["titular_ec", "pred_ec", "prob_ec", "titular_lr", "pred_lr", "prob_lr", "similitud"]],
                         width="stretch", hide_index=True,
                         column_config={"titular_ec": "El Comercio", "titular_lr": "La República",
                                        "pred_ec": "Sección EC", "pred_lr": "Sección LR",
                                        "prob_ec": st.column_config.NumberColumn("Prob. EC", format="%.2f"),
                                        "prob_lr": st.column_config.NumberColumn("Prob. LR", format="%.2f"),
                                        "similitud": st.column_config.NumberColumn("Similitud", format="%.2f")})

        if not pred_test.empty:
            from sklearn.metrics import accuracy_score, f1_score
            st.markdown("### ¿El clasificador acierta igual con ambos diarios?")
            pm = pred_test.groupby("medio").apply(lambda g: pd.Series({
                "Titulares en test": len(g), "Accuracy": accuracy_score(g.etiqueta, g.pred),
                "F1 macro": f1_score(g.etiqueta, g.pred, average="macro"), "Confianza media": g.prob.mean()}),
                include_groups=False).rename(index=MEDIO)
            st.dataframe(pm.style.format({"Titulares en test": "{:.0f}", "Accuracy": "{:.2f}",
                                          "F1 macro": "{:.2f}", "Confianza media": "{:.2f}"}), width="stretch")

# --- El modelo ------------------------------------------------------------------
with modelo:
    st.markdown('<div class="antetitulo">Comparativa</div>'
                '<div class="titular-principal">Un modelo pequeño y ajustado iguala al LLM y es 300 veces más rápido</div>',
                unsafe_allow_html=True)
    comp = pd.DataFrame([
        {"Modelo": "Zero-shot (BETO-XNLI)", "F1 macro": met.get("f1_macro_zero_shot"), "Evaluado en": "40 titulares de test",
         "Latencia por titular": "~80 ms", "Recursos": "440 MB"},
        {"Modelo": "LLM Qwen 2.5 7B (Ollama)", "F1 macro": met.get("f1_macro_llm_test"), "Evaluado en": "test (60)",
         "Latencia por titular": "~3,4 s", "Recursos": "4,7 GB + servidor"},
        {"Modelo": "BETO ajustado (elegido)", "F1 macro": met.get("f1_macro_test"), "Evaluado en": "test (60)",
         "Latencia por titular": "~10 ms", "Recursos": "440 MB"}])
    st.dataframe(comp, width="stretch", hide_index=True,
                 column_config={"F1 macro": st.column_config.ProgressColumn("F1 macro", min_value=0, max_value=1, format="%.2f")})
    if not pred_test.empty:
        a, b = st.columns(2, gap="large")
        with a:
            cm = pd.crosstab(pred_test["etiqueta"].map(NOMBRE), pred_test["pred"].map(NOMBRE)) \
                   .reindex(index=NOMBRE.values(), columns=NOMBRE.values(), fill_value=0)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, PAPEL], [1, TINTA]],
                            labels=dict(x="Predicción", y="Sección real", color="Titulares"),
                            title="Matriz de confusión en test")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        with b:
            st.markdown("### Dónde se equivoca")
            err = pred_test[pred_test["etiqueta"] != pred_test["pred"]]
            st.markdown("".join(
                f'<div class="nota"><div class="antetitulo">Real: {NOMBRE[r.etiqueta]} · Predicha: {NOMBRE[r.pred]}</div>'
                f'<h4>{html.escape(r.titular)}</h4><div class="meta">{MEDIO[r.medio]}</div></div>'
                for r in err.itertuples()), unsafe_allow_html=True)
            st.markdown('<div class="recuadro">"Real" es la sección donde el diario publicó la nota. En la revisión humana, '
                        'la nota de minería ilegal en Madre de Dios se lee más como economía que como ciencia (El Comercio '
                        'la publicó en Tecnología/Ecología): el modelo no se equivoca tanto como dice la métrica. '
                        'No se corrigió la etiqueta del test para no inflar el F1 a posteriori.</div>',
                        unsafe_allow_html=True)
            st.caption("Los otros errores están en la frontera economía–política (feriados, Fonavi) y en una nota de "
                       "Tecnología sobre estafas digitales.")

st.markdown('<div class="fecha" style="border-top:4px double #1A1A1A;border-bottom:none;margin-top:2rem">'
            '<span>AgendaPerú · Tarea 1</span><span>Fuentes: La República y El Comercio</span>'
            '<span>Descarga del 18/10/2026</span></div>', unsafe_allow_html=True)
