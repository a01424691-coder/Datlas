import json
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import h3
except ImportError:
    h3 = None

st.set_page_config(page_title="Oportunidades de servicios en Mérida", page_icon="📍", layout="wide")

# Colores (cambia aquí los hex si quieres igualarlos exactamente a la marca de Datlas)
NAVY = "#0B2A4A"
AZUL_LOGO = "#325596"
AZUL = "#1F5FBF"
AMARILLO = "#F5B400"
CORAL = "#E4572E"
TEXTO = "#1F2937"
GRIS = "#6B7280"
BORDE = "#E3E7ED"
FONDO = "#F6F7F9"

TRES = ["Mascotas", "Cafeterías", "Antros y bares"]
INDUSTRIAS = {
    "Mascotas": ["465911", "541941"],
    "Cafeterías": ["722515"],
    "Antros y bares": ["722411", "722412"],
}
COLORES = {"Mascotas": AMARILLO, "Cafeterías": AZUL, "Antros y bares": CORAL}
RGB = {"Mascotas": [245, 180, 0], "Cafeterías": [31, 95, 191], "Antros y bares": [228, 87, 46]}
ARCHIVO_DENUE = "denue_inegi_31_.csv"
CENTRO_LAT, CENTRO_LON = 20.9674, -89.6237

ORDEN_TAM = ["0 a 5 personas", "6 a 10 personas", "11 a 30 personas", "31 a 50 personas",
             "51 a 100 personas", "101 a 250 personas", "251 y más personas"]
ZONAS_ORDEN = ["Centro", "Norte", "Noreste", "Oriente", "Sureste", "Sur", "Suroeste", "Poniente", "Noroeste"]
SECTORES = {
    "11": "Agricultura y pesca", "21": "Minería", "22": "Energía y agua", "23": "Construcción",
    "31": "Manufacturas", "32": "Manufacturas", "33": "Manufacturas", "43": "Comercio al por mayor",
    "46": "Comercio al por menor", "48": "Transportes", "49": "Transportes", "51": "Medios masivos",
    "52": "Finanzas y seguros", "53": "Inmobiliarios y alquiler", "54": "Servicios profesionales",
    "55": "Corporativos", "56": "Apoyo a negocios", "61": "Educación", "62": "Salud y asistencia social",
    "71": "Esparcimiento y cultura", "72": "Alojamiento y alimentos", "81": "Otros servicios",
    "93": "Gobierno y organismos",
}

# Mapas base. Mapa usa el estilo de CARTO Voyager, el más parecido a Google Maps. Las demás vistas usan mosaicos de Esri.
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"
IMAGEN = f"{ESRI}/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}"
ETIQUETAS = f"{ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}"
RELIEVE = f"{ESRI}/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}"


def estilo_raster(*urls):
    fuentes = {f"capa{i}": {"type": "raster", "tiles": [u], "tileSize": 256, "maxzoom": 19} for i, u in enumerate(urls)}
    capas = [{"id": f"capa{i}", "type": "raster", "source": f"capa{i}"} for i in range(len(urls))]
    estilo = {"version": 8, "sources": fuentes, "layers": capas}
    return "data:application/json;charset=utf-8," + quote(json.dumps(estilo))


BASES = {
    "Mapa": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "Satélite": estilo_raster(IMAGEN),
    "Híbrido": estilo_raster(IMAGEN, ETIQUETAS),
    "Relieve": estilo_raster(RELIEVE),
}

ESTILO = f"""
<style>
.stApp {{ background: {FONDO}; color: {TEXTO}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDE}; min-width: 230px !important; width: 230px !important; }}
[data-testid="stSidebarHeader"] {{ height: 2.4rem; min-height: 2.4rem; padding-bottom: 0; }}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding: 0 0.9rem 1rem 0.9rem; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 8px; padding: 0.35rem 0.7rem; margin-bottom: 2px; color: {TEXTO}; }}
[data-testid="stSidebarNavLink"] span {{ font-size: 0.93rem; }}
[data-testid="stSidebarNavLink"]:hover {{ background: #EAF0FA; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {AZUL_LOGO}; }}
[data-testid="stSidebarNavLink"], [data-testid="stSidebarNavLink"] * {{ color: {NAVY} !important; }}
[data-testid="stSidebarNavLink"][aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] * {{ color: #FFFFFF !important; font-weight: 600; }}
.block-container {{ padding-top: 0.8rem; max-width: 1250px; }}
h1, h2, h3, h4 {{ color: {NAVY} !important; letter-spacing: -0.01em; }}
h1 {{ font-weight: 750 !important; padding-top: 0.3rem !important; }}
[data-testid="stHeading"] {{ padding-bottom: 0.45rem; }}
h2, h3 {{ padding-bottom: 0.35rem !important; margin-top: 1.1rem !important; }}
[data-testid="stMain"] [data-testid="stVerticalBlock"] {{ gap: 1.3rem; }}
.stApp p, .stApp li, .stApp label {{ color: {TEXTO}; }}
[data-testid="stCaptionContainer"] p {{ color: {GRIS}; }}
[data-tag] {{ background: #E3ECFA !important; border: 1px solid #C5D5F0 !important; }}
[data-tag] span, [data-tag] button {{ color: {AZUL_LOGO} !important; }}
[data-testid="stMultiSelect"] .react-aria-ComboBox > div, [data-testid="stSelectbox"] .react-aria-ComboBox > div,
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] {{ background: #FFFFFF !important; border: 1px solid #CBD5E1 !important; border-radius: 8px; }}
[data-testid="stMultiSelect"] .react-aria-ComboBox > div:focus-within, [data-testid="stSelectbox"] .react-aria-ComboBox > div:focus-within,
[data-testid="stTextInputRootElement"]:focus-within {{ border-color: {AZUL_LOGO} !important; box-shadow: 0 0 0 1px {AZUL_LOGO}; }}
[data-testid="stTextInputField"], [data-testid="stMultiSelect"] input, [data-testid="stSelectbox"] input {{ color: {TEXTO} !important; background: transparent !important; }}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {{ background: {AZUL_LOGO} !important; border-color: {AZUL_LOGO} !important; }}
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div {{ background: #FFFFFF !important; }}
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type, [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {{ background: {AZUL_LOGO} !important; border-color: {AZUL_LOGO} !important; }}
[data-testid="stSliderThumbValue"] {{ color: {AZUL_LOGO} !important; }}
[data-testid="stSlider"] [role="group"] > div > div:first-child {{ }}
button[data-baseweb="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"] {{ color: {AZUL_LOGO} !important; }}
[data-baseweb="tab-highlight"] {{ background-color: {AZUL_LOGO} !important; }}
.react-aria-SelectionIndicator {{ background-color: {AZUL_LOGO} !important; }}
[data-testid="stTab"][data-selected="true"], [data-testid="stTab"][data-selected="true"] p {{ color: {AZUL_LOGO} !important; }}
.rejilla {{ display: grid; gap: 1rem; align-items: stretch; margin-bottom: 0.9rem; }}
.rejilla .tarjeta {{ display: flex; flex-direction: column; height: auto; min-height: 0; box-sizing: border-box; }}
.rejilla .tarjeta .t {{ line-height: 1.3; min-height: 2.6em; }}
.rejilla .tarjeta .v {{ margin-top: 0.15rem; }}
.rejilla .tarjeta .n {{ margin-top: auto; padding-top: 0.45rem; }}
.sub-filtro {{ font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {GRIS}; margin: 0.6rem 0 0.2rem 0; padding-top: 0.7rem; border-top: 1px solid {BORDE}; }}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stTextInput"] input {{ background: #FFFFFF !important; border-color: #CBD5E1 !important; color: {TEXTO} !important; }}
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within {{ border-color: {AZUL_LOGO} !important; box-shadow: 0 0 0 1px {AZUL_LOGO}; }}
[data-baseweb="select"] input, [data-baseweb="select"] div[value], [data-baseweb="select"] span {{ color: {TEXTO}; }}
[data-baseweb="select"] svg {{ fill: {GRIS}; }}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {{ background: #FFFFFF !important; }}
[data-baseweb="popover"] li {{ color: {TEXTO} !important; background: #FFFFFF !important; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {{ background: #EAF0FA !important; }}
[data-baseweb="tag"] {{ background: #E3ECFA !important; border: 1px solid #C5D5F0; }}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {{ color: {AZUL_LOGO} !important; fill: {AZUL_LOGO} !important; }}
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{ border-color: #94A3B8 !important; background-color: #FFFFFF !important; }}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {{ background-color: {AZUL_LOGO} !important; border-color: {AZUL_LOGO} !important; }}
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span:first-child {{ border-color: #94A3B8 !important; background-color: #FFFFFF !important; }}
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {{ background-color: {AZUL_LOGO} !important; border-color: {AZUL_LOGO} !important; }}
[data-testid="stSlider"] [role="slider"] {{ background: {AZUL_LOGO} !important; }}
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: {AZUL_LOGO} !important; }}
[data-testid="stBaseButton-secondary"] {{ background: #FFFFFF; border: 1px solid {AZUL_LOGO}; color: {AZUL_LOGO}; }}
[data-testid="stBaseButton-secondary"]:hover {{ background: #EAF0FA; border-color: {AZUL_LOGO}; color: {AZUL_LOGO}; }}
.tarjeta {{ background: #FFFFFF; border: 1px solid {BORDE}; border-radius: 14px; padding: 16px 18px; height: 100%; min-height: 118px; }}
.tarjeta .t {{ font-size: 0.82rem; color: {GRIS}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
.tarjeta .v {{ font-size: 1.9rem; color: {NAVY}; font-weight: 750; line-height: 1.25; }}
.tarjeta .n {{ font-size: 0.85rem; color: {GRIS}; }}
.tarjeta.fuentes {{ min-height: 0; padding: 0.9rem 1.15rem; }}
.tarjeta.fuentes .v {{ font-size: 2.4rem; line-height: 1.2; }}
.tarjeta.fuentes .n {{ font-size: 0.74rem; color: #374151; margin-top: 0.45rem; }}
.tarjeta.grande .v {{ font-size: 2.7rem; line-height: 1.15; }}
.tarjeta.grande .n {{ font-size: 0.78rem; }}
.nota {{ font-size: 0.74rem; color: #374151; margin: 0.4rem 0 1rem 0; }}
table.claro {{ width: 100%; border-collapse: collapse; background: #FFFFFF; border: 1px solid {BORDE}; border-radius: 10px; overflow: hidden; font-size: 0.9rem; }}
table.claro th {{ background: #F3F5F9; color: {NAVY}; text-align: left; font-weight: 650; padding: 10px 14px; border-bottom: 1px solid {BORDE}; }}
table.claro td {{ padding: 10px 14px; border-bottom: 1px solid #EEF1F5; color: {TEXTO}; vertical-align: top; }}
table.claro tr:last-child td {{ border-bottom: none; }}
table.claro td:first-child {{ font-weight: 600; color: {NAVY}; white-space: nowrap; }}
.etiqueta {{ display: inline-block; padding: 3px 12px; margin: 0 6px 6px 0; border-radius: 999px; font-size: 0.85rem; color: #fff; }}
.evento {{ background: #FFFFFF; border: 1px solid {BORDE}; border-left: 6px solid {AZUL}; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; }}
.evento.proyeccion {{ border-style: dashed; border-left-style: dashed; }}
.evento .f {{ font-size: 0.8rem; font-weight: 700; color: {GRIS}; text-transform: uppercase; letter-spacing: 0.05em; }}
.evento .h {{ font-size: 1.05rem; font-weight: 700; color: {NAVY}; margin: 2px 0 4px 0; }}
.evento .d {{ font-size: 0.92rem; color: {TEXTO}; }}
.evento .s {{ font-size: 0.82rem; color: {GRIS}; margin-top: 6px; }}
.evento .s a {{ color: {AZUL}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFFFFF; border-radius: 14px; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
"""
st.markdown(ESTILO, unsafe_allow_html=True)

CSS_BP = f"""
<style>
.bp-kicker {{ font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; margin-top: 0.4rem; }}
.bp-nombre {{ font-size: 2rem; font-weight: 750; color: {NAVY}; line-height: 1.2; }}
.bp-perfil {{ font-size: 0.92rem; color: {GRIS}; margin-bottom: 0.4rem; }}
.bp-oscura {{ background: {NAVY}; border-radius: 16px; padding: 1.6rem 1.7rem; }}
.bp-oscura .bp-et {{ font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: {AMARILLO}; font-weight: 700; }}
.bp-frase {{ font-size: 1.45rem; font-weight: 650; line-height: 1.3; margin-top: 0.7rem; color: #FFFFFF; }}
.bp-texto {{ font-size: 0.9rem; line-height: 1.55; color: #E5EAF2; margin-top: 0.6rem; }}
.bp-clara {{ background: #FFFFFF; border: 1px solid {BORDE}; border-radius: 16px; padding: 1.6rem 1.7rem; }}
.bp-rejilla {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem 1.5rem; }}
.bp-et2 {{ font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: {GRIS}; font-weight: 700; }}
.bp-dato {{ font-size: 1.3rem; font-weight: 700; color: {NAVY}; margin-top: 0.25rem; }}
.bp-dato2 {{ font-size: 1rem; font-weight: 650; color: {NAVY}; margin-top: 0.35rem; }}
.bp-col {{ background: #FFFFFF; border: 1px solid {BORDE}; border-left: 6px solid {AZUL}; border-radius: 12px; padding: 1.1rem 1.3rem; min-height: 190px; margin-bottom: 1.3rem; }}
.bp-tit {{ font-weight: 700; color: {NAVY}; margin-bottom: 0.6rem; }}
.bp-texto2 {{ font-size: 0.9rem; line-height: 1.6; color: {TEXTO}; }}
.bp-clara2 {{ background: #FFFFFF; border: 1px solid {BORDE}; border-radius: 16px; padding: 1.4rem 1.6rem; min-height: 250px; }}
.bp-op {{ font-size: 1.2rem; font-weight: 700; color: {NAVY}; margin-top: 1rem; }}
.bp-total {{ font-size: 2.2rem; font-weight: 750; margin: 0.3rem 0 1.2rem 0; }}
.bp-fila {{ display: flex; align-items: center; gap: 1rem; margin: 0.9rem 0; font-size: 0.92rem; }}
.bp-fila span {{ flex: 0 0 140px; color: {TEXTO}; }}
.bp-barra {{ flex: 1; height: 10px; background: #E3E7ED; border-radius: 6px; overflow: hidden; }}
.bp-barra div {{ height: 100%; border-radius: 6px; }}
.bp-fila b {{ flex: 0 0 64px; text-align: right; color: {NAVY}; }}
.bp-puntos {{ display: flex; gap: 5px; }}
.bp-puntos i {{ display: inline-block; width: 22px; height: 22px; border-radius: 5px; }}
.bp-nombre2 {{ font-size: 1.3rem; font-weight: 700; color: {NAVY}; }}
.bp-total2 {{ font-size: 1.9rem; font-weight: 750; color: {NAVY}; margin-top: 1rem; }}
</style>
"""
st.markdown(CSS_BP, unsafe_allow_html=True)


# Ayudas
def grafica(fig, alto=None):
    eje = dict(showline=True, linecolor="black", linewidth=1, ticks="outside", tickcolor="black",
               tickfont=dict(color="black", size=12), title_font=dict(color="black", size=13), gridcolor="#E5E7EB")
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="black"), margin=dict(l=10, r=10, t=50, b=10),
                      title_font=dict(size=15, color="black"), legend=dict(font=dict(color="black")))
    fig.update_xaxes(**eje)
    fig.update_yaxes(**eje)
    if alto:
        fig.update_layout(height=alto)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def tabla(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def mapa(deck, alto=620):
    try:
        st.pydeck_chart(deck, width="stretch", height=alto)
    except TypeError:
        try:
            st.pydeck_chart(deck, use_container_width=True, height=alto)
        except TypeError:
            st.pydeck_chart(deck, use_container_width=True)


def _celda(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"
    return escape(str(v))


def tabla_clara(df):
    filas = "".join("<tr>" + "".join(f"<td>{_celda(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    cabecera = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    st.markdown(f"<table class='claro'><thead><tr>{cabecera}</tr></thead><tbody>{filas}</tbody></table>", unsafe_allow_html=True)


def tarjetas(lista, grande=False, fuentes=False, igual=False):
    clase = f"tarjeta{' grande' if grande else ''}{' fuentes' if fuentes else ''}"
    if igual:
        html = "".join(f"<div class='{clase}'><div class='t'>{t}</div><div class='v'>{v}</div><div class='n'>{n}</div></div>" for t, v, n in lista)
        st.markdown(f"<div class='rejilla' style='grid-template-columns: repeat({len(lista)}, 1fr)'>{html}</div>", unsafe_allow_html=True)
        return
    cols = st.columns(len(lista))
    for c, (titulo, valor, nota) in zip(cols, lista):
        c.markdown(f"<div class='{clase}'><div class='t'>{titulo}</div><div class='v'>{valor}</div><div class='n'>{nota}</div></div>", unsafe_allow_html=True)


def asignar_zona(df):
    dy = df["latitud"] - CENTRO_LAT
    dx = (df["longitud"] - CENTRO_LON) * np.cos(np.radians(CENTRO_LAT))
    ang = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    sector = np.round(ang.fillna(0) / 45).astype(int) % 8
    nombres = np.array(["Norte", "Noreste", "Oriente", "Sureste", "Sur", "Suroeste", "Poniente", "Noroeste"])
    zona = pd.Series(nombres[sector], index=df.index)
    zona[df["nomb_asent"].str.upper() == "CENTRO"] = "Centro"
    zona[df["latitud"].isna()] = "Sin ubicación"
    return zona


# Datos
@st.cache_data(show_spinner="Cargando el DENUE...")
def cargar_denue(origen):
    df = pd.read_csv(origen, encoding="latin1", low_memory=False)
    df = df[df["municipio"].astype(str).str.strip().str.lower() == "mérida"].copy()
    df["codigo_act"] = df["codigo_act"].astype(str)

    def clasificar(codigo):
        for nombre, prefijos in INDUSTRIAS.items():
            if codigo.startswith(tuple(prefijos)):
                return nombre
        return "Otras industrias"

    df["industria"] = df["codigo_act"].apply(clasificar)
    df["sector"] = df["codigo_act"].str[:2].map(SECTORES).fillna("Otros")
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df["anio_alta"] = pd.to_datetime(df["fecha_alta"], errors="coerce").dt.year
    df["nomb_asent"] = df["nomb_asent"].astype(str).str.strip()
    df["per_ocu"] = df["per_ocu"].astype(str).str.strip()
    df["zona"] = asignar_zona(df)
    return df


def obtener_denue():
    ruta = Path(__file__).parent / ARCHIVO_DENUE
    if ruta.exists():
        return cargar_denue(str(ruta))
    st.warning(f"No encontré {ARCHIVO_DENUE} junto a app.py. Súbelo aquí para continuar.")
    archivo = st.file_uploader("Archivo del DENUE (csv)", type="csv")
    if archivo is None:
        st.stop()
    return cargar_denue(archivo)


def competidor_cercano(g):
    g = g.dropna(subset=["latitud", "longitud"])
    if len(g) < 2:
        return np.nan, np.nan
    coords = np.radians(g[["latitud", "longitud"]].values)
    dist, _ = NearestNeighbors(n_neighbors=2, metric="haversine").fit(coords).kneighbors(coords)
    metros = dist[:, 1] * 6371000
    return float(np.median(metros)), float((metros < 100).mean() * 100)


def hhi(serie):
    p = serie / serie.sum() * 100
    return float((p ** 2).sum())


# Zonas H3 y modelos
RES_H3 = 8
MIN_EST = 30
PALETA_GRUPOS = [AZUL, AMARILLO, CORAL, "#12A4A0", "#7B5CD6", "#6B7280", "#8D5524", "#E377C2"]
VARS_PERFIL = {
    "Actividad comercial": "Logaritmo del total de establecimientos de la zona",
    "% mascotas": "Porcentaje de establecimientos de mascotas",
    "% cafeterías": "Porcentaje de establecimientos de cafeterías",
    "% antros y bares": "Porcentaje de establecimientos de antros y bares",
    "% micro": "Porcentaje de negocios con 0 a 5 personas",
    "Distancia al Centro (km)": "Distancia en línea recta a la Plaza Grande",
}
CRITERIOS = {
    "actividad": "Actividad comercial (más es mejor)",
    "competencia": "Baja competencia local (menos competidores es mejor)",
    "centralidad": "Cercanía al Centro (más cerca es mejor)",
}
PESOS_BASE = {"Mascotas": (0.3, 0.6, 0.1), "Cafeterías": (0.4, 0.4, 0.2), "Antros y bares": (0.3, 0.4, 0.3)}


def a_celda(lat, lon):
    return h3.latlng_to_cell(lat, lon, RES_H3) if hasattr(h3, "latlng_to_cell") else h3.geo_to_h3(lat, lon, RES_H3)


def centro_celda(celda):
    return h3.cell_to_latlng(celda) if hasattr(h3, "cell_to_latlng") else h3.h3_to_geo(celda)


def vecinos(celda):
    return h3.grid_disk(celda, 1) if hasattr(h3, "grid_disk") else h3.k_ring(celda, 1)


def exigir_h3():
    if h3 is None:
        st.error("Falta la librería h3. Instálala en la terminal con python -m pip install h3 y vuelve a correr la app.")
        st.stop()


def km_al_centro(lat, lon):
    dy = (np.asarray(lat) - CENTRO_LAT) * 111.2
    dx = (np.asarray(lon) - CENTRO_LON) * 111.2 * np.cos(np.radians(CENTRO_LAT))
    return np.hypot(dx, dy)


@st.cache_data(show_spinner="Armando las zonas H3...")
def construir_zonas(_denue, clave):
    d = _denue.dropna(subset=["latitud", "longitud"]).copy()
    d["zone_id"] = [a_celda(a, b) for a, b in zip(d["latitud"], d["longitud"])]
    g = d.groupby("zone_id")
    z = pd.DataFrame({"total": g.size()})
    for i in TRES:
        z[i] = d[d["industria"] == i].groupby("zone_id").size().reindex(z.index).fillna(0).astype(int)
    z["micro"] = (d["per_ocu"] == "0 a 5 personas").groupby(d["zone_id"]).mean()
    z["colonia"] = g["nomb_asent"].agg(lambda s: s.mode().iat[0])
    centros = [centro_celda(h) for h in z.index]
    z["lat"] = [c[0] for c in centros]
    z["lon"] = [c[1] for c in centros]
    z["dist"] = km_al_centro(z["lat"], z["lon"])
    for i in TRES:
        z["comp_" + i] = [sum(z[i].get(n, 0) for n in vecinos(h)) for h in z.index]
    z["elegible"] = z["total"] >= MIN_EST
    z["zona"] = asignar_zona(pd.DataFrame({"latitud": z["lat"], "longitud": z["lon"], "nomb_asent": z["colonia"]}))
    return z.reset_index()


def matriz_perfil(e):
    return pd.DataFrame(
        {
            "Actividad comercial": np.log1p(e["total"]),
            "% mascotas": e["Mascotas"] / e["total"] * 100,
            "% cafeterías": e["Cafeterías"] / e["total"] * 100,
            "% antros y bares": e["Antros y bares"] / e["total"] * 100,
            "% micro": e["micro"] * 100,
            "Distancia al Centro (km)": e["dist"],
        },
        index=e.index,
    )


@st.cache_data(show_spinner=False)
def silhouettes(_z, clave):
    e = _z[_z["elegible"]]
    X = StandardScaler().fit_transform(matriz_perfil(e))
    filas = []
    for k in range(2, 9):
        etiquetas = KMeans(k, n_init=10, random_state=42).fit_predict(X)
        filas.append({"k": k, "Silhouette": round(float(silhouette_score(X, etiquetas)), 3)})
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False)
def agrupar(_z, clave, k):
    e = _z[_z["elegible"]]
    X = StandardScaler().fit_transform(matriz_perfil(e))
    base = KMeans(k, n_init=10, random_state=42).fit_predict(X)
    otros = [adjusted_rand_score(base, KMeans(k, n_init=1, random_state=s).fit_predict(X)) for s in range(10)]
    orden = pd.Series(base).value_counts().index.tolist()
    nombre = {c: f"Grupo {chr(65 + i)}" for i, c in enumerate(orden)}
    return pd.Series([nombre[c] for c in base], index=e["zone_id"].values), float(np.mean(otros))


def describir_grupo(medias_z):
    frases = {
        ("Actividad comercial", 1): "mucha actividad comercial", ("Actividad comercial", -1): "poca actividad comercial",
        ("% mascotas", 1): "más mascotas de lo normal", ("% mascotas", -1): "menos mascotas de lo normal",
        ("% cafeterías", 1): "más cafeterías de lo normal", ("% cafeterías", -1): "menos cafeterías de lo normal",
        ("% antros y bares", 1): "más antros y bares de lo normal", ("% antros y bares", -1): "menos antros y bares de lo normal",
        ("% micro", 1): "más negocios micro", ("% micro", -1): "menos negocios micro",
        ("Distancia al Centro (km)", 1): "lejos del Centro", ("Distancia al Centro (km)", -1): "cerca del Centro",
    }
    top = medias_z.abs().sort_values(ascending=False).index[:2]
    return " y ".join(frases[(v, 1 if medias_z[v] > 0 else -1)] for v in top)


@st.cache_data(show_spinner=False)
def puntuar(_z, clave, industria, pesos):
    e = _z[_z["elegible"]].copy()
    C = np.column_stack([
        e["total"].rank(pct=True),
        1 - e["comp_" + industria].rank(pct=True),
        1 - e["dist"].rank(pct=True),
    ])
    w = np.array(pesos, dtype=float)
    w = w / w.sum()
    e["actividad"], e["competencia"], e["centralidad"] = C[:, 0], C[:, 1], C[:, 2]
    e["score"] = C @ w * 100
    e["rank"] = e["score"].rank(ascending=False, method="min").astype(int)
    sorteos = np.random.default_rng(0).dirichlet([1, 1, 1], 1000)
    rangos = (-(C @ sorteos.T)).argsort(0).argsort(0) + 1
    e["robustez"] = (rangos <= 10).mean(1) * 100
    e["rango_mediano"] = np.median(rangos, axis=1)
    return e


def color_continuo(valores, alfa=215):
    v = np.clip(np.nan_to_num(np.asarray(valores, dtype=float), nan=0.0), 0, 1)
    pasos = np.array([[219, 234, 254], [147, 197, 253], [59, 130, 246], [30, 64, 175], [11, 42, 74]], dtype=float)
    pos = v * (len(pasos) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(pasos) - 2)
    f = (pos - i)[:, None]
    rgb = pasos[i] * (1 - f) + pasos[i + 1] * f
    return [[int(a), int(b), int(c), alfa] for a, b, c in rgb]


def capa_zonas(datos, extruido, id_capa="zonas"):
    return pdk.Layer("H3HexagonLayer", id=id_capa, data=datos, get_hexagon="zone_id", get_fill_color="color",
                     get_elevation="altura", elevation_scale=1, extruded=extruido, pickable=True, auto_highlight=True,
                     stroked=True, get_line_color=[255, 255, 255, 230], line_width_min_pixels=1, coverage=0.97)


def capa_seleccion(zone_id):
    return pdk.Layer("H3HexagonLayer", id="sel", data=pd.DataFrame({"zone_id": [zone_id]}), get_hexagon="zone_id",
                     filled=False, stroked=True, get_line_color=[0, 0, 0, 255], line_width_min_pixels=4, extruded=False, pickable=False)


def zona_clic(evento):
    try:
        objetos = evento.selection.objects
    except Exception:
        return None
    zona = objetos.get("zonas") if hasattr(objetos, "get") else None
    if zona:
        return zona[0].get("zone_id")
    punto = objetos.get("estab") if hasattr(objetos, "get") else None
    if punto:
        return a_celda(punto[0]["latitud"], punto[0]["longitud"])
    return None



# Inicio
def pagina_inicio():
    st.title("Oportunidades de expansión en Mérida")
    st.markdown(
        "El cliente de Datlas, empresa mexicana de inteligencia de negocios geo referenciada, busca identificar nuevas oportunidades de crecimiento dentro del mercado mexicano, "
        "con interés particular en expandir su presencia en distintas regiones del país. Sin embargo, enfrenta una limitante central y es que no cuenta con claridad sobre qué industrias "
        "de servicios ofrecen las mejores condiciones de crecimiento en cada territorio que está considerando. Como parte de la primera fase de este análisis se seleccionaron tres regiones piloto, "
        "Torreón, Querétaro y Mérida, de las cuales este equipo de trabajo se enfoca específicamente en la región de Mérida, Yucatán."
    )

    st.subheader("La problemática")
    st.markdown(
        "Hasta el momento, las decisiones de expansión se han basado principalmente en indicadores descriptivos, estudios de mercado y la experiencia de especialistas. "
        "Si bien estos elementos proporcionan información relevante, la empresa considera que la integración de múltiples fuentes de información y el uso de técnicas de inteligencia artificial "
        "pueden permitir una evaluación más profunda, objetiva y sistemática de las oportunidades existentes en cada territorio."
    )

    st.subheader("La solución")
    st.markdown(
        "El equipo construyó un diagnóstico por industria y por zona que junta fuentes que normalmente se ven por separado. "
        "Con el DENUE se sabe cuántos establecimientos hay y dónde están, con el Censo se aproxima la demanda de cada AGEB, "
        "con los Censos Económicos, la matrícula universitaria y los perfiles de turista se entiende quién consume, "
        "y con las reseñas de Google Maps y las noticias se entiende qué opinan los clientes y qué pasa en el entorno. "
        "Con todo eso se calcula qué zonas están saturadas, cuáles están subatendidas y dónde hay espacio para nuevos participantes."
    )

    st.subheader("Las tres industrias")
    tarjetas([
        ("Mascotas", "592", "comercio y veterinarias"),
        ("Cafeterías", "958", "cafeterías, neverías y similares"),
        ("Antros y bares", "211", "bares, cantinas y centros nocturnos"),
    ], grande=True)
    st.markdown("<div class='nota'>* Establecimientos en el DENUE de Mérida.</div>", unsafe_allow_html=True)

    st.subheader("Alcance y límites")
    st.markdown(
        "El análisis cubre el municipio de Mérida, que tenía 995,129 habitantes en 2020 y 1,053,914 en 2025, y se centra en tres industrias, mascotas, cafeterías y antros y bares. "
        "La base principal es el DENUE, con 56,909 establecimientos del municipio que tienen ubicación, de los cuales 1,761 son de las tres industrias, con fechas de alta entre 2010 y 2026. "
        "Esa base se complementa con los Censos Económicos, el Censo de Población 2020, la matrícula universitaria, los perfiles de turista, 600 reseñas de Google Maps y notas de prensa."
    )
    st.markdown(
        "Hay límites que conviene tener presentes. El DENUE es un directorio y no mide ventas, además puede traer negocios que ya cerraron o que están mal ubicados. "
        "Las zonas del mapa son celdas H3 de unos 0.7 kilómetros cuadrados calculadas solo con el DENUE, así que no incluyen población ni ingresos, y esos datos por AGEB solo se usaron en el análisis de Colab. "
        "Las reseñas son únicamente de cafeterías y son pocas, y los pesos del ranking de oportunidad son supuestos del equipo. "
        "Por eso los resultados sirven para priorizar zonas y validarlas en campo y no para predecir ventas."
    )

    st.subheader("Cómo usar la app")
    st.markdown(
        "Conviene empezar por Ciudad, que describe quién vive, estudia y visita Mérida y cómo es su actividad económica. Después, en Mapa se recorre el mapa con los filtros de la izquierda por industria, zona, tamaño del establecimiento, "
        "actividad SCIAN y colonia, se cambia el mapa base entre mapa, satélite, híbrido y relieve, y se elige si se colorean las zonas, se ven los puntos o se inclina el mapa en 3D. "
        "Al hacer clic en una zona aparece su detalle con sus negocios, su grupo y su puntaje de oportunidad."
    )
    st.markdown(
        "Perfiles agrupa las zonas con K-means y deja elegir cuántos grupos usar. Industrias describe las tres industrias, mide qué tan intensa es su competencia con filtros por actividad SCIAN y zona, las reseñas de los clientes de las cafeterías y los buyer personas de sus públicos. "
        "Ciudad también incluye una cronología de inversiones, cambios y riesgos del entorno con el enlace a cada fuente."
    )
    st.markdown(
        "En Oportunidades se pueden cambiar los pesos de cada criterio para ver cómo se mueve el ranking de zonas y qué tan estable es, y Decisión resume la industria y las zonas que se recomiendan, "
        "con sus alternativas, incertidumbres y próximos pasos."
    )

    st.subheader("Fuentes")
    tabla_clara(pd.DataFrame(
        {
            "Fuente": ["DENUE, INEGI", "Censos Económicos, SAIC", "Censo de Población y Vivienda, INEGI", "Encuesta Intercensal, INEGI",
                       "ANUIES", "Perfiles de turista", "Reseñas de Google Maps", "Notas de prensa"],
            "Qué aporta": ["Ubicación, actividad y tamaño de cada establecimiento", "Unidades económicas por industria",
                           "Población por edad y sexo, y datos por AGEB", "Población total del municipio", "Matrícula universitaria",
                           "Edad de los visitantes", "Opinión de los clientes de 20 cafeterías, 600 reseñas públicas",
                           "Inversiones, cambios y riesgos del entorno"],
            "Periodo": ["Altas de 2010 a 2026", "2013, 2018 y 2023", "2020", "2025", "Ciclo 2025 a 2026",
                        "Aéreo 2023 y terrestre ADO 2024", "Hasta la fecha de descarga", "2023 a 2026"],
        }
    ))


# Comparación de industrias y filtro SCIAN
def comparacion_industrias(denue):
    m4 = denue[denue["industria"].isin(TRES)]
    st.subheader("Comparación de industrias")
    por_col = pd.crosstab(m4["nomb_asent"], m4["industria"])
    filas = []
    for ind in TRES:
        g = m4[m4["industria"] == ind]
        mediana, cerca = competidor_cercano(g)
        filas.append({
            "Industria": ind,
            "Establecimientos": len(g),
            "% en el Centro": round((g["nomb_asent"].str.upper() == "CENTRO").mean() * 100, 1),
            "Mediana a la Plaza Grande (km)": round(float(np.nanmedian(km_al_centro(g["latitud"], g["longitud"]))), 1),
            "Mediana al competidor más cercano (m)": round(mediana),
            "% con competidor a menos de 100 m": round(cerca, 1),
            "HHI por colonia": round(hhi(por_col[ind])),
        })
    tabla_clara(pd.DataFrame(filas))
    st.caption("La distancia a la Plaza Grande se usa como medida simple de acceso al Centro. Un HHI alto quiere decir que la industria está concentrada en pocas colonias.")

    st.subheader("Explora por actividad SCIAN")
    acts = m4.drop_duplicates("codigo_act").set_index("codigo_act")["nombre_act"].sort_index()
    opciones = {f"{c} · {n}": c for c, n in acts.items()}
    c1, c2 = st.columns(2)
    sel = c1.multiselect("Actividad SCIAN", list(opciones), default=list(opciones))
    zonas_sel = c2.multiselect("Zona", ZONAS_ORDEN, placeholder="Todas")
    d = m4[m4["codigo_act"].isin([opciones[s] for s in sel])]
    if zonas_sel:
        d = d[d["zona"].isin(zonas_sel)]
    if d.empty:
        st.info("No hay establecimientos con esos filtros.")
        return
    filas = []
    for codigo, g in d.groupby("codigo_act"):
        mediana, cerca = competidor_cercano(g)
        filas.append({
            "Código": codigo,
            "Actividad": acts[codigo],
            "Establecimientos": len(g),
            "% en el Centro": round((g["zona"] == "Centro").mean() * 100, 1),
            "% micro": round((g["per_ocu"] == "0 a 5 personas").mean() * 100, 1),
            "Mediana al competidor más cercano (m)": None if np.isnan(mediana) else round(mediana),
        })
    tabla_clara(pd.DataFrame(filas))
    z = d.groupby(["zona", "industria"]).size().reset_index(name="Establecimientos")
    fig = px.bar(z, x="zona", y="Establecimientos", color="industria", barmode="group", color_discrete_map=COLORES,
                 category_orders={"zona": ZONAS_ORDEN}, labels={"zona": "Zona", "industria": "Industria"})
    grafica(fig, 340)



# Página 2
EDADES_2020 = pd.DataFrame(
    {
        "Edad": ["0 a 5", "6 a 11", "12 a 17", "18 a 24", "25 a 29", "30 a 34", "35 a 39", "40 a 44", "45 a 49",
                 "50 a 54", "55 a 59", "60 a 64", "65 a 69", "70 a 74", "75 a 79", "80 a 84", "85 o más"],
        "Mujeres": [37509, 40093, 42549, 59047, 41993, 41142, 39166, 37365, 36703, 32669, 27654, 23661, 18883, 14576, 9519, 6451, 5513],
        "Hombres": [38790, 41172, 43734, 58923, 40893, 39297, 36715, 33681, 32270, 27771, 23667, 19574, 15176, 11455, 7460, 4358, 3168],
    }
)


# Buyer personas
# Los personajes son ilustrativos. Los montos y hábitos son supuestos de escenario y no equivalen a una encuesta de mercado.
PERSONAS = {
    "Antros y bares": [
        {
            "segmento": "Estudiante local", "nombre": "Jesús Gomez Lopez", "perfil": "22 años · Estudiante de UADY · Nació y vive en Mérida",
            "frase": "Me gusta mucho salir con mis amigos, pero no quiero gastar demasiado.",
            "contexto": "Actualmente está en los últimos semestres de su carrera, combina clases con un empleo parcial y acostumbra decidir sus planes con amigos por WhatsApp. Compara promociones antes de salir y prioriza un regreso sencillo y seguro.",
            "ingreso": "$8,000 / mes", "presupuesto": "$600 por salida", "frecuencia": "2 salidas / mes", "grupo": "4 a 6 amigos",
            "origen": "Trabajo de medio tiempo y apoyo familiar, no paga renta ya que sigue viviendo con sus padres",
            "busca": ["Convivir y escuchar música sin pagar de más", "Encontrar promociones atractivas", "Regresar a casa sin traslados muy largos"],
            "aleja": ["Precios muy elevados", "Traslado de regreso caro o complicado", "Filas, cobros de acceso y poca disponibilidad"],
            "canal": "Instagram y TikTok", "criterio": "Precio total, ubicación y opinión de su grupo de amigos",
            "ritual": [("Días", "Viernes o sábado"), ("Horario", "21:00 a 01:00"), ("Movilidad", "Transporte por app compartido"), ("Área preferida", "Centro y sus alrededores")],
            "oportunidad": "Paquetes para grupos y promociones de consumo temprano",
            "gasto": [("Acceso o cover", 0), ("Bebidas y consumo", 400), ("Alimentos", 0), ("Movilidad", 100)], "total": 500,
            "factores": [("Precio", 5), ("Ambiente", 4), ("Ubicación", 5), ("Exclusividad", 1)],
            "valor": "Paquetes para grupos y promociones de consumo temprano",
        },
        {
            "segmento": "Estudiante foráneo", "nombre": "Marco Garza Ayala", "perfil": "20 años · Estudiante de Anáhuac Mayab · Originario de Ciudad de México y vive en Mérida",
            "frase": "Me gusta mucho salir con mis amig@s a lugares trending.",
            "contexto": "Vive con roomies cerca de su red universitaria. Sale con amistades de la universidad y suele reservar con anticipación si el lugar es popular. Privilegia comodidad, estética y seguridad.",
            "ingreso": "$32,000 / mes", "presupuesto": "$2,200 por salida", "frecuencia": "4 a 6 salidas / mes", "grupo": "8 a 12 amigos",
            "origen": "Mensualidad disponible de su familia",
            "busca": ["Experiencias memorables con su círculo social", "Servicio ágil, reservación y buena música", "Mesas cómodas"],
            "aleja": ["Servicio lento pese al precio premium", "Saturación sin reservación garantizada", "Mala relación entre precio y experiencia"],
            "canal": "Instagram, TikTok y recomendación social", "criterio": "Ambiente, reputación, servicio y mesa disponible",
            "ritual": [("Días", "Jueves a sábado"), ("Horario", "22:30 a 03:00"), ("Movilidad", "Transporte privado o por app"), ("Área preferida", "Norte de Mérida y zonas con oferta aspiracional")],
            "oportunidad": "Reservaciones, mesas y experiencia premium curada",
            "gasto": [("Acceso o cover", 200), ("Bebidas y consumo", 1600), ("Alimentos", 200), ("Movilidad", 200)], "total": 2200,
            "factores": [("Precio", 2), ("Ambiente", 5), ("Ubicación", 4), ("Exclusividad", 5)],
            "valor": "Reservaciones, mesas y experiencia premium",
        },
        {
            "segmento": "Turista", "nombre": "Daniel Torres García", "perfil": "26 años · Profesional en viaje de ocio · Reside en Guadalajara y visita Mérida",
            "frase": "Quiero una noche divertida en un lugar que valga la pena conocer.",
            "contexto": "Visita la ciudad por cuatro noches con 5 amigos. Planea algunas actividades, pero descubre bares mediante mapas, reseñas y recomendaciones del hotel. No conoce bien las distancias.",
            "ingreso": "$40,000 / mes", "presupuesto": "$2,000 por noche", "frecuencia": "2 noches en su viaje", "grupo": "2 a 4 personas",
            "origen": "Sueldo personal mensual",
            "busca": ["Vivir una experiencia divertida en Mérida", "Conocer un lugar confiable y fácil de encontrar", "Reservar o entrar sin sorpresas"],
            "aleja": ["No saber qué zonas convienen de noche", "Información poco clara sobre horarios y acceso", "Precios ocultos o dificultad para volver al hotel"],
            "canal": "Google Maps, reseñas, TikTok e Instagram", "criterio": "Reseñas, cercanía, autenticidad y seguridad",
            "ritual": [("Días", "Viernes o sábado del viaje"), ("Horario", "21:00 a 02:00"), ("Movilidad", "Caminata corta o transporte por app"), ("Área preferida", "Centro y corredores cercanos al hospedaje")],
            "oportunidad": "Coctelería local, atención bilingüe y alianzas hoteleras",
            "gasto": [("Acceso o cover", 100), ("Bebidas y consumo", 1500), ("Alimentos", 200), ("Movilidad", 100)], "total": 2000,
            "factores": [("Precio", 3), ("Ambiente", 5), ("Ubicación", 5), ("Exclusividad", 4)],
            "valor": "Coctelería local, atención premium y alianzas hoteleras",
        },
    ],
    "Cafeterías": [
        {
            "segmento": "Estudiante local", "nombre": "Valeria Canul Pech", "perfil": "21 años · Estudiante de UADY · Nació y vive en Mérida",
            "frase": "Busco un café donde pueda estudiar con calma sin gastar de más.",
            "contexto": "Pasa buena parte del día entre clases y trabajos en equipo. Va a la cafetería para estudiar o ver a sus amigas, compara precios antes de pedir y prefiere lugares cerca de su universidad o de su casa.",
            "ingreso": "$6,000 / mes", "presupuesto": "$120 por visita", "frecuencia": "3 visitas / semana", "grupo": "1 a 3 personas",
            "origen": "Apoyo familiar y trabajo de medio tiempo, vive con sus padres",
            "busca": ["Un lugar tranquilo para estudiar o convivir", "Bebidas y postres ricos a buen precio", "Que esté cerca y sea fácil llegar"],
            "aleja": ["Esperas largas para que le entreguen su pedido", "Precios altos para lo que ofrece", "Limpieza y baños descuidados"],
            "canal": "Instagram y Google Maps", "criterio": "Precio, ambiente y cercanía a su universidad",
            "ritual": [("Días", "Lunes a viernes"), ("Horario", "Entre clases y por la tarde"), ("Movilidad", "Camión o caminando"), ("Área preferida", "Cerca de su universidad y del Centro")],
            "oportunidad": "Combos de bebida y postre y promociones para estudiantes",
            "gasto": [("Bebidas", 70), ("Alimentos y postres", 40), ("Movilidad", 10), ("Otros", 0)], "total": 120,
            "factores": [("Precio", 5), ("Ambiente", 4), ("Ubicación", 5), ("Calidad", 3)],
            "valor": "Combos de bebida y postre y promociones para estudiantes",
        },
        {
            "segmento": "Profesional local", "nombre": "Andrea Pacheco Cetina", "perfil": "29 años · Diseñadora independiente · Nació y vive en Mérida",
            "frase": "Trabajo desde cafeterías porque ahí me concentro y me reúno con mis clientes.",
            "contexto": "Trabaja por proyectos y alterna su casa con cafeterías del norte y del Centro. Busca internet estable, buen café y un ambiente cómodo para pasar varias horas, y recomienda los lugares que le gustan.",
            "ingreso": "$25,000 / mes", "presupuesto": "$160 por visita", "frecuencia": "4 visitas / semana", "grupo": "1 a 2 personas",
            "origen": "Ingresos propios por proyectos independientes",
            "busca": ["Un espacio cómodo para trabajar varias horas", "Buen café y opciones para desayunar o comer", "Internet estable y lugares donde conectar su computadora"],
            "aleja": ["Ruido y lugares saturados", "Que no toleren quedarse por horas", "Servicio lento o poco atento"],
            "canal": "Instagram, Google Maps y recomendación de colegas", "criterio": "Ambiente, comodidad para trabajar y calidad del café",
            "ritual": [("Días", "Lunes a viernes"), ("Horario", "09:00 a 14:00"), ("Movilidad", "Auto o transporte por app"), ("Área preferida", "Norte de Mérida y Centro")],
            "oportunidad": "Espacios para trabajar con buen café, internet y paquetes de desayuno",
            "gasto": [("Bebidas", 90), ("Alimentos y postres", 60), ("Movilidad", 10), ("Otros", 0)], "total": 160,
            "factores": [("Precio", 3), ("Ambiente", 5), ("Ubicación", 4), ("Calidad", 5)],
            "valor": "Espacios para trabajar con buen café, internet y paquetes de desayuno",
        },
        {
            "segmento": "Turista", "nombre": "Mariana Soto Vega", "perfil": "29 años · Profesional en viaje de ocio · Reside en Ciudad de México y visita Mérida",
            "frase": "Quiero probar un buen café local y descansar un rato del calor.",
            "contexto": "Visita la ciudad por cuatro noches con una amiga. Combina caminatas y paseos con paradas para desayunar o tomar algo, y elige dónde ir con mapas, reseñas y recomendaciones del hotel.",
            "ingreso": "$38,000 / mes", "presupuesto": "$250 por visita", "frecuencia": "4 visitas en su viaje", "grupo": "2 a 3 personas",
            "origen": "Sueldo personal mensual",
            "busca": ["Probar un café y un desayuno locales", "Un lugar confiable y fácil de encontrar", "Un sitio fresco donde descansar"],
            "aleja": ["No saber qué cafeterías valen la pena", "Esperas largas con poco tiempo de viaje", "Precios más altos de lo esperado"],
            "canal": "Google Maps, reseñas, Instagram y TikTok", "criterio": "Reseñas, cercanía al hospedaje y autenticidad",
            "ritual": [("Días", "Todos los días del viaje"), ("Horario", "08:30 a 12:00"), ("Movilidad", "Caminata corta o transporte por app"), ("Área preferida", "Centro y corredores cercanos al hospedaje")],
            "oportunidad": "Café local con desayunos, atención bilingüe y alianzas hoteleras",
            "gasto": [("Bebidas", 110), ("Alimentos y postres", 110), ("Movilidad", 30), ("Otros", 0)], "total": 250,
            "factores": [("Precio", 3), ("Ambiente", 5), ("Ubicación", 5), ("Calidad", 4)],
            "valor": "Café local con desayunos, atención bilingüe y alianzas hoteleras",
        },
    ],
    "Mascotas": [
        {
            "segmento": "Dueño de perro", "nombre": "Carlos Pech Mena", "perfil": "34 años · Ingeniero · Nació y vive en Mérida con su pareja y su perro",
            "frase": "Mi perro es parte de la familia y quiero darle lo mejor sin complicarme.",
            "contexto": "Vive en una casa del norte de Mérida con su pareja y un perro mediano. Compra el alimento en tienda o en línea, lleva a su perro con el veterinario que le recomendaron y le importa que el lugar sea de confianza y esté cerca de su casa.",
            "ingreso": "$28,000 / mes", "presupuesto": "$1,800 al mes", "frecuencia": "2 compras o visitas / mes", "grupo": "Con su pareja y su perro",
            "origen": "Sueldo personal mensual",
            "busca": ["Alimento y productos de calidad", "Un veterinario de confianza cerca de casa", "Estética, guardería y accesorios en un mismo lugar"],
            "aleja": ["Precios altos sin explicación", "Mala atención o poca paciencia con los animales", "Tener que ir a varios lugares para lo mismo"],
            "canal": "Recomendación de otros dueños, Google Maps e Instagram", "criterio": "Confianza, calidad y cercanía a su casa",
            "ritual": [("Días", "Sábados y algunas tardes entre semana"), ("Horario", "10:00 a 13:00 y 17:00 a 19:00"), ("Movilidad", "Auto propio"), ("Área preferida", "Norte de Mérida")],
            "oportunidad": "Veterinaria con estética y venta de alimento en un mismo lugar, con suscripción mensual",
            "gasto": [("Alimento", 900), ("Veterinario y medicinas", 450), ("Estética y guardería", 250), ("Accesorios y juguetes", 200)], "total": 1800,
            "factores": [("Precio", 3), ("Calidad", 5), ("Ubicación", 4), ("Confianza", 5)],
            "valor": "Veterinaria con estética y venta de alimento en un mismo lugar, con suscripción mensual",
        },
        {
            "segmento": "Dueña de gato", "nombre": "Daniela Cauich Tun", "perfil": "27 años · Diseñadora · Nació y vive en Mérida con sus dos gatos",
            "frase": "Mis gatos son mi compañía y busco productos buenos a un precio razonable.",
            "contexto": "Renta un departamento cerca del Centro y vive con dos gatos. Compara precios de alimento y arena, compra seguido en línea o en tiendas cercanas y va al veterinario solo cuando es necesario.",
            "ingreso": "$18,000 / mes", "presupuesto": "$1,100 al mes", "frecuencia": "1 a 2 compras / mes", "grupo": "Vive con sus dos gatos",
            "origen": "Sueldo personal mensual",
            "busca": ["Alimento y arena a buen precio", "Una tienda cercana a su casa o a su trabajo", "Consulta veterinaria accesible y atención amable"],
            "aleja": ["Precios más altos que en línea", "Tiendas lejanas o con poco surtido", "Cobros extra que no se anunciaron"],
            "canal": "Instagram, Google Maps y grupos de dueños de mascotas", "criterio": "Precio, cercanía y surtido",
            "ritual": [("Días", "Entre semana y fines de semana"), ("Horario", "Después del trabajo"), ("Movilidad", "Caminando, bicicleta o transporte por app"), ("Área preferida", "Centro y colonias cercanas")],
            "oportunidad": "Tienda de barrio con entrega a domicilio, promociones por volumen y consulta veterinaria accesible",
            "gasto": [("Alimento y arena", 650), ("Veterinario y medicinas", 250), ("Accesorios y juguetes", 120), ("Otros", 80)], "total": 1100,
            "factores": [("Precio", 5), ("Calidad", 4), ("Ubicación", 5), ("Confianza", 4)],
            "valor": "Tienda de barrio con entrega a domicilio, promociones por volumen y consulta veterinaria accesible",
        },
    ],
}
ACENTOS = [AZUL, AMARILLO, CORAL]


def persona_html(texto):
    return escape(str(texto))


def buyer_personas(denue):
    st.markdown(
        "Esta pestaña junta cada industria con las personas que la consumen. Primero se ven las cifras reales de cada público y después los personajes de cada industria, "
        "con lo que buscan, lo que los aleja, cómo deciden y en qué gastan."
    )
    industria = st.radio("Industria", ["Antros y bares", "Cafeterías", "Mascotas"], horizontal=True, key="bp_industria")
    n_ind = int((denue["industria"] == industria).sum())

    st.subheader("Los públicos en cifras")
    e = EDADES_2020
    jov = int(e.loc[3:4, ["Mujeres", "Hombres"]].sum().sum())
    tarjetas([
        (f"{industria}", f"{n_ind:,}", "* establecimientos en el DENUE"),
        ("Población de 18 a 29 años", f"{jov:,}", "* Censo de Población y Vivienda 2020, INEGI"),
        ("Estudiantes de 18 a 24 años", "57,933", "* ANUIES, ciclo 2025 a 2026"),
        ("Visitantes en el primer trimestre de 2026", "194,477", "* IMEF, nota de El Momento Yucatán"),
    ], fuentes=True, igual=True)
    if industria != "Mascotas":
        st.markdown(
            "Los estudiantes de 18 a 24 años son 80.7 por ciento de la matrícula universitaria, y entre los turistas que llegan en avión 51.1 por ciento tiene entre 18 y 39 años. "
            "Esos son los públicos que describen los personajes de abajo."
        )

    st.subheader("Buyer personas")
    lista = PERSONAS[industria]
    nombres = [p["segmento"] for p in lista]
    elegido = st.radio("Público", nombres, horizontal=True, key=f"bp_persona_{industria}")
    idx = nombres.index(elegido)
    p, color = lista[idx], ACENTOS[idx]

    st.markdown(f"<div class='bp-kicker' style='color:{color}'>{persona_html(p['segmento'].upper())}</div><div class='bp-nombre'>{persona_html(p['nombre'])}</div>"
                f"<div class='bp-perfil'>{persona_html(p['perfil'])}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1:
        st.markdown(
            f"<div class='bp-oscura'><div class='bp-et'>En una frase</div><div class='bp-frase'>“{persona_html(p['frase'])}”</div>"
            f"<div class='bp-et' style='margin-top:1.4rem'>Contexto personal</div><div class='bp-texto'>{persona_html(p['contexto'])}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        datos = [("Ingreso mensual", p["ingreso"]), ("Presupuesto", p["presupuesto"]), ("Frecuencia", p["frecuencia"]), ("Grupo típico", p["grupo"])]
        celdas = "".join(f"<div><div class='bp-et2'>{persona_html(k)}</div><div class='bp-dato'>{persona_html(v)}</div></div>" for k, v in datos)
        st.markdown(
            f"<div class='bp-clara'><div class='bp-rejilla'>{celdas}</div>"
            f"<div class='bp-et2' style='margin-top:1.2rem'>Origen del ingreso</div><div class='bp-dato2'>{persona_html(p['origen'])}</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader(f"Cómo conquistar a {p['nombre'].split()[0]}")
    col = st.columns(3, gap="medium")
    bloques = [
        ("Qué busca", "<br>".join("• " + persona_html(x) for x in p["busca"])),
        ("Qué lo aleja", "<br>".join("• " + persona_html(x) for x in p["aleja"])),
        ("Cómo decide", f"<b>Canal</b> {persona_html(p['canal'])}<br><b>Criterio</b> {persona_html(p['criterio'])}"),
    ]
    for c, (t, cuerpo) in zip(col, bloques):
        c.markdown(f"<div class='bp-col' style='border-left-color:{color}'><div class='bp-tit'>{t}</div><div class='bp-texto2'>{cuerpo}</div></div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="medium")
    ritual = "<br>".join(f"<b>{persona_html(k)}</b> {persona_html(v)}" for k, v in p["ritual"])
    d1.markdown(f"<div class='bp-clara2'><div class='bp-tit'>{'RITUAL DE COMPRA' if industria == 'Mascotas' else 'RITUAL DE SALIDA'}</div><div class='bp-texto2'>{ritual}</div></div>", unsafe_allow_html=True)
    d2.markdown(f"<div class='bp-clara2' style='background:{color}1F'><div class='bp-tit'>OPORTUNIDAD COMERCIAL</div><div class='bp-op'>{persona_html(p['oportunidad'])}</div></div>", unsafe_allow_html=True)

    etiq_pres = "PRESUPUESTO MENSUAL" if industria == "Mascotas" else "PRESUPUESTO POR VISITA O NOCHE"
    st.subheader(f"En qué gasta {p['nombre'].split()[0]}")
    g1, g2 = st.columns([1.3, 1], gap="medium")
    mayor = max(m for _, m in p["gasto"]) or 1
    filas = "".join(
        f"<div class='bp-fila'><span>{persona_html(k)}</span><div class='bp-barra'><div style='width:{m / mayor * 100:.0f}%;background:{color}'></div></div><b>${m:,}</b></div>"
        for k, m in p["gasto"]
    )
    g1.markdown(f"<div class='bp-clara2'><div class='bp-tit'>{etiq_pres}</div><div class='bp-total' style='color:{color}'>${p['total']:,} MXN</div>{filas}</div>", unsafe_allow_html=True)
    puntos = "".join(
        f"<div class='bp-fila'><span>{persona_html(k)}</span><div class='bp-puntos'>" + "".join(f"<i style='background:{color if i < v else '#E3E7ED'}'></i>" for i in range(5)) + "</div></div>"
        for k, v in p["factores"]
    )
    g2.markdown(f"<div class='bp-clara2' style='background:{color}1F'><div class='bp-tit'>FACTORES DE ELECCIÓN</div>{puntos}</div>", unsafe_allow_html=True)

    etiq_gasto = "Gasto mensual, escenario" if industria == "Mascotas" else "Gasto por visita o noche, escenario"
    st.subheader("Dos públicos, dos propuestas de valor" if len(lista) == 2 else "Tres públicos, tres propuestas de valor")
    cols = st.columns(len(lista), gap="medium")
    for c, q, col_ in zip(cols, lista, ACENTOS):
        c.markdown(
            f"<div class='bp-clara2' style='background:{col_}1F'><div class='bp-nombre2'>{persona_html(' '.join(q['nombre'].split()[:2]))}</div>"
            f"<div class='bp-perfil'>{persona_html(q['segmento'])}</div><div class='bp-total2'>${q['total']:,}</div>"
            f"<div class='bp-perfil'>{etiq_gasto}</div><div class='bp-texto2' style='margin-top:1rem'><b>{persona_html(q['valor'])}</b></div></div>",
            unsafe_allow_html=True,
        )
    nota("Personajes ilustrativos del equipo. Los montos son supuestos de escenario y no equivalen a una encuesta de mercado.")


def nota(texto):
    st.markdown(f"<div class='nota'>* {texto}</div>", unsafe_allow_html=True)


POB_SERIE = pd.DataFrame({"Año": [2010, 2020, 2025], "Habitantes": [830732, 995129, 1053914],
                          "Fuente": ["Censo 2010", "Censo 2020", "Encuesta Intercensal 2025"]})


def pagina_2(denue):
    st.title("Mérida en cifras")
    st.markdown(
        "Mérida, capital y ciudad más poblada del estado de Yucatán, destaca por su actividad económica, universitaria y turística. "
        "Esta página ofrece una visión general de la ciudad a través de sus principales indicadores demográficos, económicos y turísticos."
    )
    tab1, tab2, tab3 = st.tabs(["La ciudad", "Actividad económica y características del mercado", "Cronología"])

    with tab1:
        tarjetas([
            ("Población 2025", "1,053,914", "* Encuesta Intercensal 2025, INEGI"),
            ("Establecimientos", f"{len(denue):,}", "* DENUE, INEGI"),
            ("Estudiantes de 18 a 24 años", "57,933", "* ANUIES, ciclo 2025 a 2026"),
        ], fuentes=True)

        st.subheader("Cuánto ha crecido")
        fig = px.line(POB_SERIE, x="Año", y="Habitantes", markers=True, color_discrete_sequence=[AZUL],
                      title="Población del municipio de Mérida", text="Habitantes")
        fig.update_traces(texttemplate="%{text:,}", textposition="top center", line=dict(width=3), marker=dict(size=10))
        fig.update_xaxes(tickvals=POB_SERIE["Año"].tolist())
        fig.update_yaxes(range=[700000, 1150000], tickformat=",")
        grafica(fig, 380)
        nota("Censos de Población y Vivienda 2010 y 2020 y Encuesta Intercensal 2025, INEGI.")
        c10, c20, c25 = POB_SERIE["Habitantes"]
        st.markdown(
            f"Entre 2010 y 2020 el municipio ganó {c20 - c10:,} habitantes, un crecimiento de {(c20 / c10 - 1) * 100:.1f} por ciento y de {((c20 / c10) ** 0.1 - 1) * 100:.1f} por ciento anual. "
            f"Entre 2020 y 2025 sumó {c25 - c20:,} más, {(c25 / c20 - 1) * 100:.1f} por ciento en total y {((c25 / c20) ** 0.2 - 1) * 100:.1f} por ciento anual, "
            "así que la ciudad sigue creciendo pero a un ritmo menor."
        )

        st.subheader("Quién vive en Mérida")
        e = EDADES_2020.copy()
        total_m, total_h = e["Mujeres"].sum(), e["Hombres"].sum()
        fig = go.Figure()
        fig.add_bar(y=e["Edad"], x=-e["Hombres"], orientation="h", name="Hombres", marker_color=AZUL,
                    customdata=e["Hombres"], hovertemplate="%{y} años<br>Hombres %{customdata:,}<extra></extra>")
        fig.add_bar(y=e["Edad"], x=e["Mujeres"], orientation="h", name="Mujeres", marker_color=AMARILLO,
                    customdata=e["Mujeres"], hovertemplate="%{y} años<br>Mujeres %{customdata:,}<extra></extra>")
        marcas = [-60000, -40000, -20000, 0, 20000, 40000, 60000]
        fig.update_layout(barmode="relative", title="Población por edad y sexo, 2020", bargap=0.08)
        fig.update_xaxes(tickvals=marcas, ticktext=[f"{abs(m):,}" for m in marcas], title="Personas")
        fig.update_yaxes(categoryorder="array", categoryarray=list(e["Edad"]), title="Edad")
        grafica(fig, 520)
        nota("Censo de Población y Vivienda 2020, INEGI.")

        bloques = pd.DataFrame(
            {
                "Grupo": ["0 a 17", "18 a 29", "30 a 59", "60 o más"],
                "Personas": [
                    e.loc[:2, ["Mujeres", "Hombres"]].sum().sum(),
                    e.loc[3:4, ["Mujeres", "Hombres"]].sum().sum(),
                    e.loc[5:10, ["Mujeres", "Hombres"]].sum().sum(),
                    e.loc[11:, ["Mujeres", "Hombres"]].sum().sum(),
                ],
            }
        )
        bloques["% de la población"] = (bloques["Personas"] / 995129 * 100).round(1)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(bloques, x="Grupo", y="% de la población", color_discrete_sequence=[AZUL],
                         title="Población por grupo de edad (%)", text="% de la población")
            fig.update_traces(textposition="outside")
            grafica(fig, 360)
            nota("Censo de Población y Vivienda 2020, INEGI.")
        with c2:
            sexo = pd.DataFrame({"Sexo": ["Mujeres", "Hombres"], "Personas": [515760, 479369]})
            fig = px.bar(sexo, x="Sexo", y="Personas", color="Sexo", color_discrete_map={"Mujeres": AMARILLO, "Hombres": AZUL},
                         title="Población por sexo, 2020", text="Personas")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(showlegend=False)
            grafica(fig, 360)
        st.markdown(
            f"En 2020 vivían 995,129 personas en el municipio, 51.8 por ciento mujeres. El grupo de 18 a 24 años es el más grande con 117,970 personas, "
            f"y {bloques.loc[1, '% de la población']:.1f} por ciento de la población tiene entre 18 y 29 años. "
            f"Entre 30 y 59 años está {bloques.loc[2, '% de la población']:.1f} por ciento y solo {bloques.loc[3, '% de la población']:.1f} por ciento tiene 60 años o más, "
            "así que es una ciudad joven con mucha gente en edad de salir a consumir. No se muestran 2,532 personas con edad desconocida."
        )

        st.subheader("Quién estudia")
        est = pd.DataFrame({"Edad": ["18 a 24 años", "Otras edades"], "Estudiantes": [57933, 71812 - 57933]})
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(est, x="Edad", y="Estudiantes", color="Edad", text="Estudiantes",
                         color_discrete_map={"18 a 24 años": AZUL, "Otras edades": AMARILLO},
                         title="Estudiantes universitarios, ciclo 2025 a 2026")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(showlegend=False)
            grafica(fig, 360)
        with c2:
            st.markdown(
                "<br><br>De 71,812 estudiantes universitarios, 57,933 tienen entre 18 y 24 años, es decir 80.7 por ciento. "
                "Es una población joven que sale a comer, tomar café y convivir, y por eso pesa tanto en cafeterías, antros y bares.",
                unsafe_allow_html=True,
            )
        nota("ANUIES, matrícula de Mérida, ciclo 2025 a 2026.")
        st.subheader("Quién visita")
        edades = pd.DataFrame(
            {
                "Rango": ["18 a 29", "30 a 39", "40 a 49", "50 a 59", "60 a 69", "70 o más"],
                "Turista aéreo 2023": [22.8, 28.3, 23.5, 15.5, 7.8, 2.0],
                "Turista terrestre ADO 2024": [32.6, 24.3, 18.5, 13.1, 8.5, 3.1],
            }
        ).melt("Rango", var_name="Perfil", value_name="% de encuestados")
        fig = px.bar(edades, x="Rango", y="% de encuestados", color="Perfil", barmode="group",
                     color_discrete_sequence=[AZUL, AMARILLO], title="Edad de los turistas")
        grafica(fig, 380)
        nota("Perfiles de turista, aéreo 2023 y terrestre ADO 2024.")
        st.markdown(
            "51.1 por ciento de los turistas que llegan en avión tiene entre 18 y 39 años, y entre los que llegan por tierra el grupo más grande es el de 18 a 29. "
            "Por eso cafeterías, antros y bares se leen como industrias para estudiantes y visitantes, "
            "y mascotas como una industria para los hogares que viven en cada zona."
        )

    with tab2:
        economia(denue)

    with tab3:
        senales()


def economia(denue):
    sect_n = denue["sector"].value_counts()
    micro = (denue["per_ocu"] == "0 a 5 personas").mean() * 100
    centro = (denue["zona"] == "Centro").mean() * 100
    tarjetas([
        ("Establecimientos", f"{len(denue):,}", "* DENUE, INEGI"),
        ("Negocios micro", f"{micro:.1f}%", "* 0 a 5 personas, DENUE, INEGI"),
        (sect_n.index[0], f"{sect_n.iloc[0] / sect_n.sum() * 100:.1f}%", "* Sector más grande, DENUE, INEGI"),
        ("Negocios en el Centro", f"{centro:.1f}%", "* DENUE, INEGI"),
    ], fuentes=True)

    st.subheader("A qué se dedica la ciudad")
    sect = denue["sector"].value_counts()
    top = sect.head(10).sort_values()
    graf = pd.DataFrame({"Sector": top.index, "Establecimientos": top.values})
    fig = px.bar(graf, x="Establecimientos", y="Sector", orientation="h", color_discrete_sequence=[AZUL],
                 title="Los diez sectores con más establecimientos")
    grafica(fig, 420)
    nota("DENUE, INEGI.")
    pct = sect / sect.sum() * 100
    st.markdown(
        f"El comercio al por menor es el sector más grande, con {pct['Comercio al por menor']:.1f} por ciento de los establecimientos, "
        f"seguido de otros servicios con {pct['Otros servicios']:.1f} por ciento y de alojamiento y alimentos con {pct['Alojamiento y alimentos']:.1f} por ciento. "
        "Es una ciudad de comercio y servicios, y eso explica por qué las industrias que se estudian en este proyecto son todas de servicios."
    )


    st.subheader("Qué tan grandes son los negocios")
    tam = denue["per_ocu"].value_counts().reindex(ORDEN_TAM).fillna(0)
    graf = pd.DataFrame({"Personal ocupado": tam.index, "% de establecimientos": (tam / tam.sum() * 100).round(1).values})
    fig = px.bar(graf, x="Personal ocupado", y="% de establecimientos", color_discrete_sequence=[AMARILLO],
                 title="Establecimientos según el personal que ocupan")
    fig.update_xaxes(categoryorder="array", categoryarray=ORDEN_TAM)
    grafica(fig, 380)
    nota("DENUE, INEGI.")
    chicos = tam.iloc[0] / tam.sum() * 100
    st.markdown(
        f"{chicos:.1f} por ciento de los establecimientos ocupa a cinco personas o menos. La ciudad vive de negocios pequeños, "
        "así que una cafetería, una tienda de mascotas o un bar nuevo compite casi siempre contra negocios de su mismo tamaño."
    )

    st.subheader("Dónde están los negocios")
    zz = denue["zona"].value_counts().reindex(ZONAS_ORDEN).dropna()
    c1, c2 = st.columns(2)
    with c1:
        g = pd.DataFrame({"Zona": zz.index, "Establecimientos": zz.values})
        fig = px.bar(g, x="Zona", y="Establecimientos", color_discrete_sequence=[AZUL], title="Establecimientos por zona de la ciudad")
        fig.update_xaxes(categoryorder="array", categoryarray=ZONAS_ORDEN)
        grafica(fig, 380)
        nota("DENUE, INEGI. Zonas según la posición respecto a la Plaza Grande.")
    with c2:
        col = denue["nomb_asent"].str.title().value_counts().head(10).sort_values()
        fig = px.bar(x=col.values, y=col.index, orientation="h", color_discrete_sequence=[AMARILLO],
                     labels={"x": "Establecimientos", "y": "Colonia"}, title="Las diez colonias con más establecimientos")
        grafica(fig, 380)
        nota("DENUE, INEGI.")
    centro = (denue["zona"] == "Centro").mean() * 100
    st.markdown(
        f"El Centro concentra {centro:.1f} por ciento de los establecimientos de todo el municipio y la colonia con más negocios es {col.index[-1]}. "
        "Esto importa porque una parte de la demanda del Centro viene de gente que no vive ahí."
    )

    st.subheader("Cuándo abrieron los negocios")
    anios_c = [2014, 2019, 2024]
    altas_anio = denue[denue["anio_alta"].isin(anios_c)].groupby("anio_alta").size().reset_index(name="Altas")
    altas_anio["anio_alta"] = altas_anio["anio_alta"].astype(int)
    fig = px.bar(altas_anio, x="anio_alta", y="Altas", color_discrete_sequence=[AZUL],
                 labels={"anio_alta": "Año del Censo Económico"}, title="Altas en el DENUE en los años de Censo Económico")
    fig.update_xaxes(type="category")
    grafica(fig, 360)
    nota("DENUE, INEGI.")
    st.markdown(
        "El DENUE se actualiza alrededor de los Censos Económicos y en esos años se registran muchos negocios que ya existían, "
        "por eso solo se muestran los años de censo y la fecha de alta no equivale a la fecha de apertura."
    )

    st.subheader("Entorno económico")
    tarjetas([
        ("Inflación en Mérida, junio de 2026", "4.15%", "* IMEF, Diario de Yucatán. La nacional fue de 3.37%"),
        ("Salario mínimo diario 2026", "$315.04", "* Diario de Yucatán, aumento de 13%"),
        ("Ocupación hotelera, marzo de 2026", "59.7%", "* IMEF, El Momento Yucatán. Un año antes fue de 60.8%"),
    ], fuentes=True)


def industrias_tres(denue):
    m4 = denue[denue["industria"].isin(TRES)]
    st.subheader("Industrias")
    tabla_ind = pd.DataFrame(
        {
            "Industria": ["Mascotas", "Mascotas", "Cafeterías", "Antros y bares", "Antros y bares"],
            "Código": ["465911", "541941", "722515", "722411", "722412"],
            "Actividad": [
                "Comercio al por menor de mascotas",
                "Servicios veterinarios para mascotas",
                "Cafeterías, fuentes de sodas, neverías, refresquerías y similares",
                "Centros nocturnos, discotecas y similares",
                "Bares, cantinas y similares",
            ],
        }
    )
    tabla_ind["Establecimientos"] = tabla_ind["Código"].map(m4.groupby("codigo_act").size()).fillna(0).astype(int)
    tabla_clara(tabla_ind)
    tot = len(denue)
    part = m4.groupby("industria").size().reindex(TRES)
    tarjetas([(i, f"{part[i]:,}", f"* {part[i] / tot * 100:.1f}% de los establecimientos del DENUE") for i in TRES], grande=True)

    st.subheader("Crecimiento frente a la ciudad")
    anios = [2014, 2019, 2024]
    total_ciudad = denue.groupby("anio_alta").size().reindex(anios)
    altas = m4.groupby(["industria", "anio_alta"]).size().unstack(fill_value=0).reindex(columns=anios, fill_value=0)
    indice_ciudad = total_ciudad / total_ciudad[2014] * 100

    filas = []
    for ind in ["Cafeterías", "Mascotas", "Antros y bares"]:
        indice = altas.loc[ind] / altas.loc[ind, 2014] * 100
        for a in anios:
            filas.append({"Año": a, "Serie": ind, "Índice": indice[a]})
    for a in anios:
        filas.append({"Año": a, "Serie": "Toda la ciudad", "Índice": indice_ciudad[a]})
    graf = pd.DataFrame(filas)
    cmap = {**COLORES, "Toda la ciudad": "#6B7280"}
    fig = px.line(graf, x="Año", y="Índice", color="Serie", markers=True, color_discrete_map=cmap,
                  title="Establecimientos registrados por año del Censo Económico (2014 igual a 100)")
    fig.update_xaxes(tickvals=anios)
    grafica(fig, 420)
    nota("DENUE, INEGI. Años del Censo Económico.")
    vs = (altas[2024] / total_ciudad[2024]) / (altas[2014] / total_ciudad[2014])
    st.markdown(
        f"El índice de la ciudad en 2024 es de {indice_ciudad[2024]:.1f}. Cafeterías llega a "
        f"{altas.loc['Cafeterías', 2024] / altas.loc['Cafeterías', 2014] * 100:.1f} con una razón de {vs['Cafeterías']:.2f} contra la ciudad, "
        f"mascotas a {altas.loc['Mascotas', 2024] / altas.loc['Mascotas', 2014] * 100:.1f} con {vs['Mascotas']:.2f} y "
        f"antros y bares a {altas.loc['Antros y bares', 2024] / altas.loc['Antros y bares', 2014] * 100:.1f} con {vs['Antros y bares']:.2f}. "
        "La fecha de alta del DENUE se infla en los años de Censo Económico, por eso el crecimiento se compara siempre contra el total de la ciudad."
    )


    st.subheader("Tamaño de los negocios por industria")
    t = m4.groupby(["industria", "per_ocu"]).size().reset_index(name="n")
    t["%"] = t["n"] / t.groupby("industria")["n"].transform("sum") * 100
    fig = px.bar(t, x="per_ocu", y="%", color="industria", barmode="group", color_discrete_map=COLORES,
                 category_orders={"per_ocu": ORDEN_TAM}, labels={"per_ocu": "Personal ocupado", "%": "% de establecimientos", "industria": "Industria"},
                 title="Establecimientos según el personal que ocupan")
    grafica(fig, 380)
    nota("DENUE, INEGI.")

    st.subheader("Dónde está cada industria")
    zi = m4.groupby(["zona", "industria"]).size().reset_index(name="Establecimientos")
    fig = px.bar(zi, x="zona", y="Establecimientos", color="industria", barmode="group", color_discrete_map=COLORES,
                 category_orders={"zona": ZONAS_ORDEN}, labels={"zona": "Zona", "industria": "Industria"}, title="Establecimientos por zona")
    grafica(fig, 380)
    nota("DENUE, INEGI.")
    top_col = m4["nomb_asent"].value_counts().head(8).index
    ci = m4[m4["nomb_asent"].isin(top_col)].groupby(["nomb_asent", "industria"]).size().reset_index(name="Establecimientos")
    ci["nomb_asent"] = ci["nomb_asent"].str.title()
    fig = px.bar(ci, x="Establecimientos", y="nomb_asent", color="industria", orientation="h", color_discrete_map=COLORES,
                 labels={"nomb_asent": "Colonia", "industria": "Industria"}, title="Las ocho colonias con más establecimientos")
    fig.update_yaxes(categoryorder="total ascending")
    grafica(fig, 380)
    nota("DENUE, INEGI.")
    enc = (m4["zona"] == "Centro").groupby(m4["industria"]).mean() * 100
    st.markdown(
        f"El Centro reúne {enc['Antros y bares']:.1f} por ciento de los antros y bares, {enc['Cafeterías']:.1f} por ciento de las cafeterías y solo {enc['Mascotas']:.1f} por ciento de los negocios de mascotas, "
        "que están más repartidos por la ciudad porque dependen de los hogares de cada zona."
    )

    st.subheader("Cuándo abrieron")
    anios_c = [2014, 2019, 2024]
    ai = m4[m4["anio_alta"].isin(anios_c)].groupby(["anio_alta", "industria"]).size().reset_index(name="Altas")
    ai["anio_alta"] = ai["anio_alta"].astype(int)
    fig = px.bar(ai, x="anio_alta", y="Altas", color="industria", barmode="group", color_discrete_map=COLORES,
                 labels={"anio_alta": "Año del Censo Económico", "industria": "Industria"}, title="Altas en el DENUE en los años de Censo Económico")
    fig.update_xaxes(type="category")
    grafica(fig, 380)
    nota("DENUE, INEGI.")


def competencia(denue):
    st.markdown(
        "Una oportunidad no depende solo de la demanda, porque muchos establecimientos pueden indicar un mercado atractivo "
        "pero también una señal de saturación. Aquí se mide dónde se concentra la oferta de cada industria y qué tan cerca están los competidores."
    )
    m4 = denue[denue["industria"].isin(TRES)]

    st.subheader("Concentración")
    por_ageb = pd.crosstab(m4["ageb"], m4["industria"])
    por_col = pd.crosstab(m4["nomb_asent"], m4["industria"])
    filas = []
    for ind in TRES:
        top = por_col[ind].sort_values(ascending=False)
        filas.append(
            {
                "Industria": ind,
                "Establecimientos": int(por_col[ind].sum()),
                "HHI por AGEB": round(hhi(por_ageb[ind])),
                "HHI por colonia": round(hhi(por_col[ind])),
                "Colonia con más": top.index[0].title(),
                "Establecimientos ahí": int(top.iloc[0]),
                "% del total": round(top.iloc[0] / por_col[ind].sum() * 100, 1),
            }
        )
    tab = pd.DataFrame(filas)
    tabla_clara(tab)
    fig = px.bar(tab.melt(id_vars="Industria", value_vars=["HHI por AGEB", "HHI por colonia"], var_name="Nivel", value_name="HHI"),
                 x="Industria", y="HHI", color="Nivel", barmode="group", color_discrete_sequence=[AZUL, AMARILLO],
                 title="Concentración de cada industria")
    grafica(fig, 380)
    st.markdown(
        "Por AGEB el HHI sale bajo en las tres industrias, lo que quiere decir que los negocios están repartidos en muchos AGEB. "
        "Por colonia cambia, sobre todo en antros y bares, donde el Centro concentra casi la mitad de los establecimientos."
    )

    st.subheader("Competencia cercana")
    filas = []
    for ind in TRES:
        mediana, cerca = competidor_cercano(m4[m4["industria"] == ind])
        filas.append({"Industria": ind, "Mediana al competidor más cercano (m)": round(mediana),
                      "% con competidor a menos de 100 m": round(cerca, 1)})
    comp = pd.DataFrame(filas)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(comp, x="Industria", y="Mediana al competidor más cercano (m)", color="Industria", color_discrete_map=COLORES)
        fig.update_layout(showlegend=False)
        grafica(fig, 350)
    with c2:
        fig = px.bar(comp, x="Industria", y="% con competidor a menos de 100 m", color="Industria", color_discrete_map=COLORES)
        fig.update_layout(showlegend=False)
        grafica(fig, 350)
    st.markdown("Las cafeterías son las que tienen a su competidor más cerca, por eso es donde la competencia directa es más intensa.")

    st.subheader("Saturación por AGEB")
    st.markdown(
        "El índice de saturación de mercado compara los establecimientos por cada mil habitantes de un AGEB contra el promedio de la ciudad. "
        "Menos de 50 se considera subatendido, de 50 a 150 equilibrado y más de 150 saturado. Entraron los AGEB con al menos 500 habitantes."
    )
    ism = pd.DataFrame(
        {"Mascotas": [155, 159, 121], "Cafeterías": [187, 138, 110], "Antros y bares": [325, 7, 103]},
        index=["Subatendido", "Equilibrado", "Saturado"],
    )
    ism_sc = pd.DataFrame(
        {"Mascotas": [147, 149, 116], "Cafeterías": [167, 138, 107], "Antros y bares": [323, 2, 87]},
        index=["Subatendido", "Equilibrado", "Saturado"],
    )
    opcion = st.radio("Versión del índice", ["Con todos los AGEB (435)", "Sin los AGEB del Centro (412)"], horizontal=True)
    base = ism if opcion.startswith("Con") else ism_sc
    pct = base.div(base.sum()) * 100
    graf = pct.reset_index().melt("index", var_name="Industria", value_name="% de AGEB").rename(columns={"index": "Nivel"})
    fig = px.bar(graf, y="Industria", x="% de AGEB", color="Nivel", orientation="h", barmode="stack",
                 color_discrete_map={"Subatendido": AZUL, "Equilibrado": AMARILLO, "Saturado": CORAL})
    grafica(fig, 320)
    st.caption("Cifras calculadas en el análisis de Colab con Censo 2020 por AGEB. En el Centro viven pocas personas y consumen muchas que vienen de fuera, por eso se muestra también la versión sin esos AGEB.")

    st.subheader("Zonas con espacio para nuevos participantes")
    st.markdown(
        "Un AGEB candidato tiene población arriba de la mediana y una densidad de establecimientos igual o menor a la mediana de la industria. "
        "Después se pidió que las viviendas con automóvil e internet estuvieran arriba de la mediana de la ciudad."
    )
    cand = pd.DataFrame(
        {
            "Industria": ["Mascotas", "Cafeterías", "Antros y bares"],
            "AGEB candidatos": [121, 125, 155],
            "Con automóvil e internet arriba de la mediana": [43, 43, 59],
            "Colonias de los más grandes": [
                "Unidad Habitacional CTM, Paseos de Opichén, Chuburná de Hidalgo, José María Iturralde, Francisco de Montejo II",
                "Caucel II, Unidad Habitacional CTM, Paseos de Opichén, Chuburná de Hidalgo, Serapio Rendón III",
                "Unidad Habitacional CTM, Paseos de Opichén, Polígono 108, Chuburná de Hidalgo, Los Héroes",
            ],
        }
    )
    tabla_clara(cand)

    st.subheader("Mezcla de industrias por AGEB")
    grupos = pd.DataFrame(
        {"Grupo": ["Predominio cafeterías", "Predominio mascotas", "Mixto cafeterías y mascotas", "Mixto antros y bares y mascotas"],
         "AGEB": [125, 74, 165, 42]}
    )
    fig = px.bar(grupos, x="AGEB", y="Grupo", orientation="h", color_discrete_sequence=[AZUL])
    fig.update_layout(yaxis={"autorange": "reversed"})
    grafica(fig, 300)
    st.markdown(
        "Con KMeans se agruparon los 406 AGEB con al menos un establecimiento según su mezcla de industrias. "
        "La mezcla casi no separa colonias por su perfil demográfico, y lo más claro es que donde se juntan antros y bares viven menos jóvenes."
    )

    comparacion_industrias(denue)



def pagina_industrias(denue):
    st.title("Industrias")
    st.markdown(
        "Esta página reúne las tres industrias del proyecto, mascotas, cafeterías y antros y bares, con su competencia, las reseñas de sus clientes y las personas que las consumen."
    )
    t1, t2, t3, t4 = st.tabs(["Las tres industrias", "Competencia", "Reseñas", "Buyer personas"])
    with t1:
        industrias_tres(denue)
    with t2:
        competencia(denue)
    with t3:
        resenas()
    with t4:
        buyer_personas(denue)


# Página 4
ASPECTOS = pd.DataFrame(
    {
        "Aspecto": ["Espera y rapidez", "Limpieza y baños", "Instalaciones", "Precio y costo", "Café y bebidas",
                    "Comida y postres", "Servicio y personal", "Ambiente y lugar"],
        "Menciones": [58, 34, 172, 176, 647, 604, 448, 301],
        "Reseñas": [49, 32, 140, 144, 416, 425, 380, 258],
        "Score medio": [0.46, 0.59, 0.64, 0.66, 0.73, 0.75, 0.79, 0.87],
        "% positivo": [62.07, 70.59, 69.19, 72.16, 77.74, 79.47, 84.38, 90.70],
        "% negativo": [15.52, 11.76, 5.23, 5.68, 4.48, 4.47, 5.13, 1.99],
    }
)
CAFETERIAS = pd.DataFrame(
    {
        "Cafetería": ["Benélica", "El Sereno", "Vandola", "Roles y Café Latte del Parque", "Mercy's Coffee & Co.", "El Hoyo Casa de Té",
                      "Santagloria", "Cafe Loco Coffee LAB", "The Coffee Mayab", "Arista Barista Norte", "El Amor de Dios Café",
                      "Aliberada", "Buen Confidente", "The Kai Coffee", "Katz Café", "Cultura Fika", "Kadus Café",
                      "Cafecito Necesito", "CAFELIA", "Condesa Barra de Café"],
        "Zona": ["Norte", "Noroeste", "Norte", "Norte", "Centro", "Centro", "Norte", "Norte", "Noroeste", "Norte", "Norte",
                 "Norte", "Noroeste", "Centro", "Norte", "Centro", "Centro", "Norte", "Norte", "Noroeste"],
        "Score medio": [1.00, 1.00, 1.00, 0.96, 0.96, 0.96, 0.95, 0.94, 0.93, 0.93, 0.92, 0.88, 0.86, 0.85, 0.84, 0.82, 0.79, 0.79, 0.74, 0.57],
        "% positivo": [100, 100, 100, 96.67, 96.67, 96.67, 96.67, 93.33, 96.67, 96.67, 93.33, 90, 93.33, 90, 86.67, 90, 90, 83.33, 83.33, 70],
        "% negativo": [0, 0, 0, 0, 0, 0, 0, 0, 3.33, 3.33, 0, 3.33, 6.67, 3.33, 3.33, 6.67, 10, 3.33, 10, 13.33],
    }
)
ZONAS = pd.DataFrame(
    {
        "Zona": ["Centro", "Norte", "Noroeste"],
        "Cafeterías": [5, 11, 4],
        "Reseñas": [150, 330, 120],
        "Score medio": [0.88, 0.90, 0.84],
        "% positivo": [92.67, 92.73, 90.00],
        "% negativo": [4.00, 2.12, 5.83],
    }
)
ZONA_ASPECTO = pd.DataFrame(
    [[0.90, 0.73, 0.73, 0.40, 0.76, 0.57, 0.68, 0.78],
     [0.86, 0.74, 0.73, 0.43, 0.64, 0.68, 0.67, 0.79],
     [0.86, 0.71, 0.84, 0.55, 0.52, 0.39, 0.62, 0.81]],
    index=["Centro", "Norte", "Noroeste"],
    columns=["Ambiente", "Café y bebidas", "Comida y postres", "Espera", "Instalaciones", "Limpieza", "Precio", "Servicio"],
)


def resenas():
    st.markdown(
        "Los datos estructurados dicen cuántas cafeterías hay y dónde, pero no qué piensa la gente. Aquí se analizaron 600 reseñas de Google Maps "
        "de 20 cafeterías de Mérida, con 30 reseñas por cafetería. El sentimiento se clasificó con el modelo BETO en español y cada reseña se "
        "separó en oraciones para medir la opinión por aspecto."
    )
    tarjetas([
        ("Reseñas analizadas", "600", "20 cafeterías"),
        ("Positivas", "92.2%", "553 reseñas"),
        ("Neutrales", "4.5%", "27 reseñas"),
        ("Negativas", "3.3%", "20 reseñas"),
    ])

    st.subheader("Qué se valora y qué molesta")
    a = ASPECTOS.copy()
    a["% de reseñas que lo mencionan"] = (a["Reseñas"] / 600 * 100).round(1)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(a.sort_values("% de reseñas que lo mencionan"), x="% de reseñas que lo mencionan", y="Aspecto",
                     orientation="h", color_discrete_sequence=[AZUL], title="Reseñas que mencionan cada aspecto (%)")
        grafica(fig, 400)
    with c2:
        graf = a.melt(id_vars="Aspecto", value_vars=["% positivo", "% negativo"], var_name="Sentimiento", value_name="%")
        fig = px.bar(graf, x="%", y="Aspecto", color="Sentimiento", orientation="h", barmode="group",
                     color_discrete_map={"% positivo": AZUL, "% negativo": AMARILLO},
                     title="Sentimiento de las menciones por aspecto (%)")
        grafica(fig, 400)
    st.markdown(
        "Lo que más se menciona es la comida y los postres, el café y las bebidas y el servicio, y el ambiente es lo mejor evaluado. "
        "La espera, la limpieza y el precio son los aspectos con más menciones negativas, aunque se mencionan poco. "
        "En las reseñas positivas dominan palabras como rico, comida, excelente, amable, servicio y atención."
    )
    tabla_clara(a)

    st.subheader("Diferencias entre zonas")
    st.markdown(
        "El archivo de reseñas no trae la zona, así que las cafeterías se agruparon a mano según la colonia de su dirección. "
        "La percepción es muy parecida entre zonas y la prueba de Kruskal Wallis no sale significativa, con H de 3.19 y p de 0.20."
    )
    tabla_clara(ZONAS)
    fig = px.imshow(ZONA_ASPECTO, text_auto=".2f", color_continuous_scale=[[0, CORAL], [0.5, "#FFF3C4"], [1, AZUL]],
                    zmin=-1, zmax=1, aspect="auto", title="Score medio por aspecto y zona")
    grafica(fig, 300)

    st.subheader("Diferencias entre cafeterías")
    zonas_sel = st.multiselect("Zonas", ["Centro", "Norte", "Noroeste"], default=["Centro", "Norte", "Noroeste"])
    c = CAFETERIAS[CAFETERIAS["Zona"].isin(zonas_sel)].sort_values("Score medio")
    fig = px.bar(c, x="Score medio", y="Cafetería", color="Zona", orientation="h",
                 color_discrete_map={"Centro": AZUL, "Norte": AMARILLO, "Noroeste": CORAL})
    grafica(fig, max(350, 28 * len(c)))
    st.markdown(
        "Entre cafeterías sí hay diferencias significativas, con H de 56.23 y p menor a 0.001. Nueve cafeterías no tienen ninguna reseña negativa "
        "y la más baja es Condesa Barra de Café con un score de 0.57 y 13.3 por ciento de reseñas negativas."
    )

    st.subheader("De qué se quejan")
    st.markdown(
        "Al leer las 20 reseñas negativas, unas ocho hablan del trato del personal, cuatro de la lentitud y dos del precio, y también hay quejas "
        "por la calidad de lo que se sirve. Además 49 reseñas tienen alguna queja puntual y 29 de ellas son positivas en general."
    )
    quejas = pd.DataFrame(
        {"Aspecto": ["Café y bebidas", "Comida y postres", "Servicio y personal", "Instalaciones", "Precio y costo",
                     "Espera y rapidez", "Ambiente y lugar", "Limpieza y baños"],
         "Reseñas con queja": [21, 21, 20, 8, 7, 6, 6, 4]}
    )
    fig = px.bar(quejas.sort_values("Reseñas con queja"), x="Reseñas con queja", y="Aspecto", orientation="h",
                 color_discrete_sequence=[CORAL])
    grafica(fig, 380)

    st.subheader("¿Coincide con los datos estructurados?")
    st.markdown(
        "Los indicadores mostraban que las cafeterías crecieron muy rápido, con un índice de 723.3 en 2024 contra 310.7 de la ciudad, y que 43.0 por ciento "
        "de los AGEB estaba subatendido. Las reseñas muestran que 92.2 por ciento de la gente está contenta, así que la demanda existe y en lo básico está bien atendida. "
        "La oportunidad de una cafetería nueva no está en ofrecer algo totalmente distinto sino en hacerlo mejor en rapidez, trato del personal y limpieza."
    )
    st.caption("Cifras calculadas en el notebook de reseñas. Son pocas cafeterías por zona y casi todas las reseñas salen positivas, por eso las comparaciones son orientativas.")



# Señales externas. Cada evento es un diccionario, se pueden agregar más siguiendo el mismo formato.
EVENTOS = [
    {
        "fecha": "2023-11", "tipo": "Cambio", "industria": "Mascotas", "proyeccion": False,
        "titulo": "Inauguran la primera clínica veterinaria pública del sureste en Mérida",
        "detalle": "Está en Yucalpetén, al poniente, y desde enero de 2024 ofrece consulta, desparasitación y esterilización a precios públicos, con vacuna antirrábica gratuita. Comparte edificio con el Centro Municipal de Atención Animal.",
        "fuentes": [("La Jornada Maya", "https://www.lajornadamaya.mx/yucatan/222682/inauguran-en-merida-la-primera-clinica-veterinaria-publica-del-sureste-yucatan-mauricio-vila-renan-barrera-cemma-centro-municipal-de-atencion-animal-perritos-gatos")],
    },
    {
        "fecha": "2025-07", "tipo": "Cambio", "industria": "Cafeterías", "proyeccion": False,
        "titulo": "Un instructor de baristas describe un auge del café en Yucatán",
        "detalle": "Diario de Yucatán recogió el testimonio de un instructor que ha capacitado a más de 300 baristas desde 2010 y que describe como notable la apertura de cafeterías después de la pandemia. Es un testimonio y no una cifra de mercado.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2025/07/04/el-cafe-en-auge-entre-yucatecos.html")],
    },
    {
        "fecha": "2026-01", "tipo": "Cambio", "industria": "General", "proyeccion": False,
        "titulo": "El salario mínimo sube 13 por ciento en 2026",
        "detalle": "El salario mínimo general de 2026 es de 315.04 pesos diarios desde el 1 de enero, un aumento total de 13 por ciento.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2025/12/09/de-cuanto-sera-el-salario-minimo-para-2026-en-yucatan-lista-de-profesiones.html")],
    },
    {
        "fecha": "2026-01", "tipo": "Inversión", "industria": "General", "proyeccion": True,
        "titulo": "Proyecto de plaza comercial en el noroeste sobre el Periférico",
        "detalle": "Propone 20 locales comerciales en un terreno de 9,813 metros cuadrados. En enero de 2026 la manifestación de impacto ambiental estuvo abierta a consulta pública y la Secretaría de Desarrollo Sustentable lo describe como propuesto y no aprobado.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2026/01/09/otra-plaza-comercial-en-merida-ahora-el-noroeste.html")],
    },
    {
        "fecha": "2026-03", "tipo": "Inversión", "industria": "Cafeterías", "proyeccion": False,
        "titulo": "Tim Hortons abre su primera sucursal en Mérida",
        "detalle": "Abrió el 26 de marzo de 2026 en la zona de Xcanatún, al norte de la ciudad sobre la carretera Mérida Progreso. Una nota menciona que podrían abrir más sucursales, sobre todo en el norte, sin dar sitios ni fechas.",
        "fuentes": [("Visitamex", "https://visitamex.com/%E2%98%95-tim-hortons-llega-a-merida-abre-su-primera-sucursal-en-la-ciudad/"),
                    ("Posta México", "https://www.posta.com.mx/yucatan/llega-tim-hortons-a-merida-esto-es-lo-que-se-sabe-sobre-la-nueva-sucursal/vl2167190")],
    },
    {
        "fecha": "2026-03", "tipo": "Inversión", "industria": "General", "proyeccion": False,
        "titulo": "Grupo La Comer abre Fresko, su primera tienda en Yucatán",
        "detalle": "Invirtió 885 millones de pesos, genera 370 empleos formales y tiene 6,315 metros cuadrados de piso de venta sobre la carretera Mérida Progreso, en la zona norte, igual que Tim Hortons.",
        "fuentes": [("Retailers.mx", "https://retailers.mx/la-comer-abre-fresko-en-merida-con-inversion-de-885-mdp/"),
                    ("Just Retail Latam", "https://www.justretail.news/latam/noticias/grupo-la-comer-abre-en-merida-la-primera-tienda-fresko-de-yucatan/")],
    },
    {
        "fecha": "2026-04", "tipo": "Riesgo", "industria": "General", "proyeccion": False,
        "titulo": "Los restaurantes absorben el alza de insumos y salarios",
        "detalle": "En abril de 2026 la Canirac Yucatán dijo que hay cerca de 17,000 negocios de comida en el estado, que la canasta alimentaria subió 8 por ciento en un año y que los empresarios absorben los costos sin subir precios. Según su presidente, la mayor afectación llegó a inicios de año por el aumento salarial.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2026/04/28/restaurantes-en-yucatan-enfrentan-costos-al-alza.html")],
    },
    {
        "fecha": "2026-05", "tipo": "Riesgo", "industria": "General", "proyeccion": False,
        "titulo": "Más visitantes en el primer trimestre pero menor ocupación hotelera",
        "detalle": "De enero a marzo de 2026 llegaron 194,477 visitantes a Mérida contra 191,602 un año antes, pero la ocupación hotelera de marzo fue de 59.7 por ciento contra 60.8. El IMEF lo atribuye a las rentas vacacionales y dice que el crecimiento beneficia a restaurantes, transporte y comercios.",
        "fuentes": [("El Momento Yucatán", "https://elmomentoyucatan.mx/turismo-impulsa-crecimiento-de-visitantes-en-merida-durante-2026/")],
    },
    {
        "fecha": "2026-05", "tipo": "Cambio", "industria": "Cafeterías", "proyeccion": False,
        "titulo": "Expo Specialty Coffee Sureste 2026 en Mérida",
        "detalle": "Se realizó del 29 al 31 de mayo en el Centro de Convenciones Yucatán Siglo XXI y reunió a productores, baristas, emprendedores y cafeterías independientes.",
        "fuentes": [("Posta México", "https://www.posta.com.mx/yucatan/inauguran-la-expo-specialty-coffee-sureste-2026-en-merida/vl2209904")],
    },
    {
        "fecha": "2026-07", "tipo": "Inversión", "industria": "Mascotas", "proyeccion": True,
        "titulo": "SimiPet Care prepara su primera clínica veterinaria en Mérida",
        "detalle": "Estará en la colonia Lázaro Cárdenas, junto a una Farmacias Similares, y está en construcción. No hay fecha de apertura confirmada y la empresa no ha confirmado oficialmente la sucursal ni su horario.",
        "fuentes": [("Posta México", "https://www.posta.com.mx/yucatan/simipet-care-en-merida-donde-estara-la-nueva-veterinaria-de-farmacias-similares/vl2230986"),
                    ("Yucatán Magazine", "https://yucatanmagazine.com/dr-simis-doc-in-a-box-for-pets-comes-to-merida/")],
    },
    {
        "fecha": "2026-07", "tipo": "Cambio", "industria": "Mascotas", "proyeccion": True,
        "titulo": "Presentan el nuevo reglamento de protección y bienestar de la fauna de Mérida",
        "detalle": "La alcaldesa lo presentó a finales de julio de 2026 y falta que lo discuta el Cabildo. Incluye esterilización, registro de animales de compañía y reglas para establecimientos relacionados con mascotas, que entrarían a un padrón municipal. Las notas no precisan obligaciones concretas.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2026/07/29/presentan-el-nuevo-reglamento-para-la-proteccion-y-bienestar-de-la-fauna-en-merida.html"),
                    ("Heraldo de México Yucatán", "https://yucatan.heraldodemexico.com.mx/local/2026/7/30/historico-merida-presenta-un-nuevo-reglamento-de-bienestar-animal-con-mas-proteccion-para-perros-gat-6015.html")],
    },
    {
        "fecha": "2026-07", "tipo": "Riesgo", "industria": "General", "proyeccion": False,
        "titulo": "La inflación de Mérida está arriba del promedio nacional",
        "detalle": "La inflación de Mérida cerró junio de 2026 en 4.15 por ciento contra 3.37 nacional. Yucatán tenía 435,524 trabajadores asegurados al IMSS, 3,485 menos que en junio de 2025, aunque las ventas al menudeo crecieron 5.2 por ciento anual.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2026/07/30/el-imef-reporta-inflacion-y-franco-estancamiento-en-la-economia-de-yucatan.html")],
    },
    {
        "fecha": "2026-08", "tipo": "Inversión", "industria": "General", "proyeccion": True,
        "titulo": "Consulta pública para la Plaza Xcanatún",
        "detalle": "Plaza de 14 locales con zona de drive thru y terrazas en un lote de 2,486 metros cuadrados. La consulta de la manifestación de impacto ambiental fue del 13 al 19 de agosto de 2026 y la nota no dice si el permiso ya se otorgó.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2026/08/13/proxima-plaza-comercial-en-xcanatun-merida-consulta-publica-para-autorizar-la-obra.html")],
    },
    {
        "fecha": "2026-09", "tipo": "Riesgo", "industria": "General", "proyeccion": False,
        "titulo": "Baja la pernocta y la ocupación hotelera en Mérida",
        "detalle": "En julio de 2026 la pernocta bajó 1.3 por ciento y la ocupación hotelera de Mérida cayó 10.7 puntos porcentuales contra 2025. Hubo contrapesos, el aeropuerto de Mérida movió 362,828 pasajeros en julio, 3.3 por ciento más, y los cruceros en Progreso subieron 7.7 por ciento en pasajeros.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2026/09/08/disminucion-turistica-en-yucatan-menos-pernocta-y-baja-ocupacion.html")],
    },
    {
        "fecha": "2026-10", "tipo": "Cambio", "industria": "General", "proyeccion": False,
        "titulo": "Yucatán es cuarto lugar nacional en afiliados a la Canirac",
        "detalle": "Es el segundo estado con mayor crecimiento de afiliados en 2026, pero su presidente estatal dice que el sector sigue presionado por la inflación y por un cierre de 2025 por debajo de lo esperado.",
        "fuentes": [("Diario de Yucatán", "https://www.yucatan.com.mx/merida/2026/10/17/restaurantes-en-yucatan-entre-costos-y-crecimiento.html")],
    },
]
TIPOS_EVENTO = ["Inversión", "Cambio", "Riesgo"]
COLOR_TIPO = {"Inversión": AZUL, "Cambio": AMARILLO, "Riesgo": CORAL}


def senales():
    st.markdown(
        "Línea de tiempo de inversiones, cambios y riesgos de Mérida que pueden influir en mascotas, cafeterías y antros y bares, actualizada a octubre de 2026. "
        "Las tarjetas con borde punteado son proyecciones o trámites sin resolver y no hechos consumados, y las marcadas como fuente única todavía necesitan una segunda fuente independiente. "
        "No se encontraron notas con fuente confiable que hablen específicamente de antros y bares, así que para esa industria aplican los eventos generales."
    )
    c1, c2, c3 = st.columns([2, 2, 1])
    tipos = c1.multiselect("Tipo", TIPOS_EVENTO, placeholder="Todos")
    inds = c2.multiselect("Industria", TRES + ["General"], placeholder="Todas")
    proy = c3.checkbox("Incluir proyecciones", value=True)

    ev = sorted(EVENTOS, key=lambda e: e["fecha"])
    if tipos:
        ev = [e for e in ev if e["tipo"] in tipos]
    if inds:
        ev = [e for e in ev if e["industria"] in inds]
    if not proy:
        ev = [e for e in ev if not e.get("proyeccion")]
    if not ev:
        st.info("No hay eventos con esos filtros.")
        return

    cuenta = pd.Series([e["tipo"] for e in ev]).value_counts()
    tarjetas([(t, str(int(cuenta.get(t, 0))), "eventos") for t in TIPOS_EVENTO])
    st.markdown("<br>", unsafe_allow_html=True)

    lt = pd.DataFrame(ev)
    lt["fecha_dt"] = pd.to_datetime(lt["fecha"] + "-01")
    lt["Estado"] = np.where(lt["proyeccion"], "Proyección o trámite", "Hecho reportado")
    lt["Evento"] = lt["titulo"]
    fig = px.scatter(lt, x="fecha_dt", y="industria", color="tipo", symbol="Estado", hover_name="Evento",
                     color_discrete_map=COLOR_TIPO, symbol_map={"Hecho reportado": "circle", "Proyección o trámite": "circle-open"},
                     category_orders={"industria": TRES + ["General"]}, labels={"fecha_dt": "Fecha", "industria": "Industria", "tipo": "Tipo"})
    fig.update_traces(marker=dict(size=15, line=dict(width=2)))
    fig.update_xaxes(range=["2023-09-01", "2026-12-01"])
    grafica(fig, 330)

    for e in ev:
        fuentes = e.get("fuentes", [])
        enlaces = " · ".join(f"<a href='{escape(u)}' target='_blank'>{escape(n)}</a>" for n, u in fuentes)
        if len(fuentes) == 1:
            enlaces += " · Fuente única"
        clase = "evento proyeccion" if e.get("proyeccion") else "evento"
        color = COLOR_TIPO.get(e["tipo"], AZUL)
        st.markdown(
            f"<div class='{clase}' style='border-left-color:{color}'>"
            f"<div class='f'>{escape(e['fecha'])} · {escape(e['tipo'])} · {escape(e['industria'])}</div>"
            f"<div class='h'>{escape(e['titulo'])}</div>"
            f"<div class='d'>{escape(e['detalle'])}</div>"
            f"<div class='s'>{enlaces}</div></div>",
            unsafe_allow_html=True,
        )
    st.caption("Verifica cada cifra con la fuente primaria antes de citarla.")


# Explorador urbano
def k_actual(z):
    sil = silhouettes(z, "zonas")
    mejor = int(sil.loc[sil["Silhouette"].idxmax(), "k"])
    return int(st.session_state.get("k_perfil", min(max(mejor, 3), 7)))


METRICAS_ZONA = ["Actividad comercial total", "Mascotas", "Cafeterías", "Antros y bares", "Grupo (K-means)", "Oportunidad"]


def zonas_con_estado(z, d, metrica, industria_op, pesos):
    """Devuelve las zonas con color y altura según la métrica elegida."""
    z = z.copy()
    if metrica == "Actividad comercial total" or metrica in TRES:
        if metrica in TRES:
            cuenta = d[d["industria"] == metrica].groupby("zone_id").size()
            z["valor"] = z["zone_id"].map(cuenta).fillna(0)
        else:
            z["valor"] = z["total"]
        z = z[z["valor"] > 0].copy()
        norm = np.sqrt(z["valor"] / z["valor"].max())
        z["color"] = color_continuo(norm.values)
        z["altura"] = z["valor"] * 8
        z["texto"] = z["valor"].astype(int).astype(str) + (" negocios de todas las industrias" if metrica not in TRES else " establecimientos")
    elif metrica == "Grupo (K-means)":
        sil = silhouettes(z, "zonas")
        k = k_actual(z)
        grupos, _ = agrupar(z, "zonas", k)
        z = z[z["zone_id"].isin(grupos.index)].copy()
        z["grupo"] = z["zone_id"].map(grupos)
        letras = sorted(z["grupo"].unique())
        hexa = {g: PALETA_GRUPOS[i % len(PALETA_GRUPOS)] for i, g in enumerate(letras)}
        z["color"] = z["grupo"].map(lambda g: [int(hexa[g][i:i + 2], 16) for i in (1, 3, 5)] + [215])
        z["altura"] = z["total"] * 4
        z["texto"] = z["grupo"]
    else:
        p = puntuar(z, "zonas", industria_op, pesos)
        z = z[z["zone_id"].isin(p["zone_id"])].merge(p[["zone_id", "score", "rank"]], on="zone_id")
        z["color"] = color_continuo((z["score"] / 100).values)
        z["altura"] = z["score"] * 12
        z["texto"] = "Oportunidad " + z["score"].round(1).astype(str) + " · lugar " + z["rank"].astype(str)
    z["titulo"] = z["colonia"].str.title()
    z["l2"] = z["texto"]
    z["l3"] = z["zona"]
    return z


def _mapa_cuerpo(denue):
    base = denue[denue["industria"].isin(TRES)].dropna(subset=["latitud", "longitud"]).copy()
    z = construir_zonas(denue.dropna(subset=["latitud", "longitud"]).reset_index(drop=True), "zonas")
    base["zone_id"] = [a_celda(a, b) for a, b in zip(base["latitud"], base["longitud"])]
    actividades = (base.drop_duplicates("codigo_act").set_index("codigo_act")["nombre_act"]
                   .pipe(lambda s: {c: f"{c} · {n}" for c, n in s.sort_index().items()}))
    colonias_lista = sorted(base["nomb_asent"].unique())

    defecto = {
        "f_ind": list(TRES), "f_zona": [], "f_tam": [], "f_act": [], "f_col": [], "f_nombre": "",
        "f_base": "Mapa", "f_capa": "Puntos", "f_metrica": "Actividad comercial total", "f_op": "Cafeterías",
        "f_3d": False, "f_puntos": False, "f_radio": 300,
    }
    for k, v in defecto.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("zona_sel", None)
    st.session_state.setdefault("ult_clic", None)

    def limpiar():
        for k, v in defecto.items():
            st.session_state[k] = v
        st.session_state["zona_sel"] = None

    cont_kpi = st.container()
    izq, der = st.columns([1, 3], gap="medium")
    with izq:
        with st.container(border=True):
            st.markdown("#### Filtros")
            st.multiselect("Industria", TRES, key="f_ind")
            st.multiselect("Zona", [x for x in ZONAS_ORDEN if x in set(base["zona"])], key="f_zona", placeholder="Todas")
            st.multiselect("Colonia", colonias_lista, key="f_col", placeholder="Todas")
            st.multiselect("Tamaño del establecimiento", ORDEN_TAM, key="f_tam", placeholder="Todos")
            st.multiselect("Actividad SCIAN", list(actividades.values()), key="f_act", placeholder="Todas")
            st.text_input("Buscar por nombre", key="f_nombre", placeholder="Ej. Starbucks")
            st.button("Limpiar filtros", on_click=limpiar, width="stretch")
        with st.container(border=True):
            st.markdown("#### Vista")
            st.radio("Capa", ["Puntos", "Zonas H3", "Hexágonos 3D", "Mapa de calor"], key="f_capa")
            st.radio("Mapa base", list(BASES), key="f_base", horizontal=True)
            es_zonas = st.session_state["f_capa"] == "Zonas H3"
            st.markdown("<div class='sub-filtro'>Opciones de la capa</div>", unsafe_allow_html=True)
            st.selectbox("Qué colorear en las zonas", METRICAS_ZONA, key="f_metrica", disabled=not es_zonas)
            st.selectbox("Industria para oportunidad", TRES, key="f_op",
                         disabled=not (es_zonas and st.session_state["f_metrica"] == "Oportunidad"))
            st.checkbox("Inclinar en 3D", key="f_3d", disabled=not es_zonas)
            st.checkbox("Mostrar establecimientos encima", key="f_puntos", disabled=not es_zonas)
            st.slider("Radio del hexágono (m)", 150, 800, step=50, key="f_radio", disabled=st.session_state["f_capa"] != "Hexágonos 3D")

    ss = st.session_state
    d = base[base["industria"].isin(ss["f_ind"] or TRES)]
    if ss["f_zona"]:
        d = d[d["zona"].isin(ss["f_zona"])]
    if ss["f_tam"]:
        d = d[d["per_ocu"].isin(ss["f_tam"])]
    if ss["f_act"]:
        d = d[d["codigo_act"].isin([c for c, e in actividades.items() if e in ss["f_act"]])]
    if ss["f_col"]:
        d = d[d["nomb_asent"].isin(ss["f_col"])]
    if ss["f_nombre"].strip():
        d = d[d["nom_estab"].astype(str).str.contains(ss["f_nombre"].strip(), case=False, na=False, regex=False)]

    if d.empty:
        with der:
            st.info("No hay establecimientos con esos filtros. Prueba con quitar alguno.")
        return
    zona_top = d["zona"].value_counts()
    with cont_kpi:
        tarjetas([
            ("Establecimientos", f"{len(d):,}", "en la selección"),
            ("Colonias", f"{d['nomb_asent'].nunique():,}", "con al menos uno"),
            ("En el Centro", f"{(d['zona'] == 'Centro').mean() * 100:.1f}%", "de la selección"),
            ("Zona con más", zona_top.index[0], f"{zona_top.iloc[0]:,} establecimientos"),
        ], igual=True)

    with der:
        capa = ss["f_capa"]
        d = d.assign(color=d["industria"].map(lambda i: RGB[i] + [235]))
        d = d.assign(tam_n=d["per_ocu"].map({t: i + 1 for i, t in enumerate(ORDEN_TAM)}).fillna(1))
        d = d.assign(titulo=d["nom_estab"].astype(str), l2=d["industria"], l3=d["nomb_asent"] + " · " + d["zona"])
        datos = d[["titulo", "l2", "l3", "latitud", "longitud", "color", "tam_n"]].copy()
        tooltip = {"html": "<b>{titulo}</b><br/>{l2}<br/>{l3}",
                   "style": {"backgroundColor": NAVY, "color": "white", "fontSize": "12px", "borderRadius": "6px"}}
        capas, pitch, seleccionable = [], 0, False
        pesos_op = PESOS_BASE[ss["f_op"]]
        leyenda = ""

        def puntos(id_capa="estab"):
            return pdk.Layer("ScatterplotLayer", id=id_capa, data=datos, get_position=["longitud", "latitud"], get_fill_color="color",
                             get_radius=40, radius_min_pixels=3, radius_max_pixels=14, pickable=True, auto_highlight=True,
                             stroked=True, get_line_color=[255, 255, 255, 255], line_width_min_pixels=1)

        if capa == "Zonas H3":
            zz = zonas_con_estado(z, d, ss["f_metrica"], ss["f_op"], pesos_op)
            if zz.empty:
                st.info("No hay zonas con esa selección.")
                return
            datos_z = zz[["zone_id", "color", "altura", "titulo", "l2", "l3"]]
            capas.append(capa_zonas(datos_z, ss["f_3d"]))
            pitch = 50 if ss["f_3d"] else 0
            seleccionable = True
            if ss["f_puntos"]:
                capas.append(puntos())
            if ss["zona_sel"]:
                capas.append(capa_seleccion(ss["zona_sel"]))
            if ss["f_metrica"] == "Grupo (K-means)":
                nombres = sorted(zz["grupo"].unique())
                leyenda = "".join(f"<span class='etiqueta' style='background:{PALETA_GRUPOS[i % len(PALETA_GRUPOS)]}'>{g}</span>"
                                  for i, g in enumerate(nombres))
            else:
                leyenda = ("<span style='font-size:0.85rem'>Menos </span>"
                           + "".join(f"<span style='display:inline-block;width:28px;height:12px;background:rgb({c[0]},{c[1]},{c[2]})'></span>"
                                     for c in color_continuo(np.linspace(0, 1, 6), 255))
                           + "<span style='font-size:0.85rem'> Más</span>")
        elif capa == "Puntos":
            capas.append(puntos())
            leyenda = "".join(f"<span class='etiqueta' style='background:{COLORES[i]}'>{i}</span>" for i in TRES if i in set(d["industria"]))
        elif capa == "Hexágonos 3D":
            capas.append(pdk.Layer("HexagonLayer", data=datos, get_position=["longitud", "latitud"], radius=ss["f_radio"],
                                   elevation_scale=6, elevation_range=[0, 300], extruded=True, pickable=True, coverage=0.9,
                                   color_range=[[219, 234, 254], [147, 197, 253], [96, 165, 250], [59, 130, 246], [30, 64, 175], [11, 42, 74]]))
            pitch = 50
            tooltip = {"text": "Establecimientos {elevationValue}"}
        else:
            capas.append(pdk.Layer("HeatmapLayer", data=datos, get_position=["longitud", "latitud"], radius_pixels=40, opacity=0.85,
                                   color_range=[[31, 95, 191, 60], [96, 165, 250], [253, 230, 138], [245, 180, 0], [228, 87, 46], [160, 30, 30]]))
            tooltip = None

        lat_lo, lat_hi = d["latitud"].quantile([0.02, 0.98])
        lon_lo, lon_hi = d["longitud"].quantile([0.02, 0.98])
        clat, clon = (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2
        ext = max(lat_hi - lat_lo, (lon_hi - lon_lo) * np.cos(np.radians(clat)), 0.01)
        zoom = float(np.clip(np.log2(360 / ext) - 1.3, 9, 15))
        vista = pdk.ViewState(latitude=float(clat), longitude=float(clon), zoom=zoom, pitch=pitch)
        deck = pdk.Deck(layers=capas, initial_view_state=vista, map_style=BASES[ss["f_base"]], tooltip=tooltip)

        if seleccionable:
            try:
                evento = st.pydeck_chart(deck, width="stretch", height=620, on_select="rerun", selection_mode="single-object",
                                         key=f"mapa_{ss['f_base']}_{ss['f_metrica']}")
            except TypeError:
                evento = None
                mapa(deck, 620)
            clic = zona_clic(evento) if evento is not None else None
            if clic and clic != ss["ult_clic"]:
                ss["ult_clic"] = clic
                ss["zona_sel"] = clic
                st.rerun()
        else:
            mapa(deck, 620)
        if leyenda:
            st.markdown(leyenda, unsafe_allow_html=True)
        st.caption("Zonas H3 de resolución 8 calculadas solo con el DENUE. Mapas base de CARTO, Esri y colaboradores de OpenStreetMap.")

        if capa == "Zonas H3":
            ids = z["zone_id"].tolist()
            etiqueta = {r.zone_id: f"{r.colonia.title()} · {int(r.total)} negocios" for r in z.itertuples()}
            orden = sorted(ids, key=lambda i: -z.set_index("zone_id").loc[i, "total"])
            actual = ss["zona_sel"] if ss["zona_sel"] in ids else None
            idx = ([None] + orden).index(actual) if actual else 0
            elegida = st.selectbox("O elige una zona de la lista", [None] + orden, index=idx,
                                   format_func=lambda i: "Ninguna" if i is None else etiqueta[i])
            if elegida != ss["zona_sel"]:
                ss["zona_sel"] = elegida
                st.rerun()
            if ss["zona_sel"]:
                detalle_zona(z, d, ss["zona_sel"])
            else:
                st.info("Haz clic en una zona del mapa o elige una de la lista para ver su detalle.")


def pagina_explorador(denue):
    exigir_h3()
    st.title("Mapa")
    st.markdown(
        "Mapa interactivo de Mérida con los establecimientos del DENUE. Se pueden filtrar los establecimientos, cambiar la capa y la vista, "
        "inclinar el mapa en 3D y hacer clic en una zona para ver su detalle."
    )
    _mapa_cuerpo(denue)


def detalle_zona(z, d, zid):
    fila = z[z["zone_id"] == zid]
    if fila.empty:
        return
    r = fila.iloc[0]
    with st.container(border=True):
        st.markdown(f"#### Zona de {r['colonia'].title()}")
        st.caption(f"Celda {zid} · sector {r['zona']} · a {r['dist']:.1f} km de la Plaza Grande")
        tarjetas([
            ("Negocios en la zona", f"{int(r['total'])}", "de todas las industrias"),
            ("Mascotas", f"{int(r['Mascotas'])}", f"{int(r['comp_Mascotas'])} contando vecinas"),
            ("Cafeterías", f"{int(r['Cafeterías'])}", f"{int(r['comp_Cafeterías'])} contando vecinas"),
            ("Antros y bares", f"{int(r['Antros y bares'])}", f"{int(r['comp_Antros y bares'])} contando vecinas"),
        ], igual=True)
        if not r["elegible"]:
            st.markdown(f"Esta zona tiene menos de {MIN_EST} establecimientos, así que no entra a los grupos ni al ranking de oportunidad.")
        else:
            grupos, _ = agrupar(z, "zonas", k_actual(z))
            filas = []
            for ind in TRES:
                p = puntuar(z, "zonas", ind, PESOS_BASE[ind])
                q = p[p["zone_id"] == zid].iloc[0]
                filas.append({"Industria": ind, "Oportunidad": round(q["score"], 1), "Lugar": f"{int(q['rank'])} de {len(p)}",
                              "Estable en el top 10": f"{q['robustez']:.0f}%"})
            tabla(pd.DataFrame(filas))
            if zid in grupos.index:
                st.markdown(f"Pertenece al {grupos[zid].lower()} del análisis de perfiles.")
        st.caption("No se muestra población, edad ni ingreso de la zona porque el análisis por zona usa solo el DENUE. Esos datos están por AGEB en el análisis de Colab.")
        top = d[d["zone_id"] == zid]["nombre_act"].value_counts().head(5).reset_index()
        top.columns = ["Actividad más frecuente (tres industrias)", "Establecimientos"]
        if not top.empty:
            tabla(top)


# Perfiles territoriales
def pagina_perfiles(denue):
    exigir_h3()
    st.title("Perfiles territoriales")
    z = construir_zonas(denue.dropna(subset=["latitud", "longitud"]).reset_index(drop=True), "zonas")
    e = z[z["elegible"]].reset_index(drop=True)
    st.markdown(
        f"Las zonas H3 con al menos {MIN_EST} establecimientos de cualquier giro ({len(e)} de {len(z)}) se agruparon con K-means según su perfil comercial. "
        "Las variables se estandarizaron antes de agrupar para que ninguna pese más solo por su escala."
    )
    st.subheader("Variables usadas")
    tabla(pd.DataFrame({"Variable": list(VARS_PERFIL), "Cómo se calcula": list(VARS_PERFIL.values())}))

    st.subheader("Cuántos grupos")
    sil = silhouettes(z, "zonas")
    mejor = int(sil.loc[sil["Silhouette"].idxmax(), "k"])
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(sil, x="k", y="Silhouette", color_discrete_sequence=[AZUL], title="Silhouette por número de grupos")
        fig.update_xaxes(dtick=1)
        grafica(fig, 320)
    with c2:
        k = st.slider("Número de grupos", 2, 8, min(max(mejor, 3), 7), key="k_perfil")
        grupos, ari = agrupar(z, "zonas", k)
        st.markdown(
            f"El mejor silhouette es {sil['Silhouette'].max():.2f} con {mejor} grupos. Un silhouette alrededor de 0.2 indica separación débil, "
            f"así que los grupos son una forma útil de resumir las zonas y no categorías nítidas. Con {k} grupos y distintas semillas el acuerdo promedio entre corridas (ARI) es {ari:.2f}, que es moderado y confirma que los límites entre grupos cambian un poco según la semilla."
        )

    e = e.assign(grupo=e["zone_id"].map(grupos))
    X = matriz_perfil(e)
    Z = (X - X.mean()) / X.std(ddof=0)
    Z["grupo"] = e["grupo"].values
    medias_z = Z.groupby("grupo").mean()
    medias = X.assign(grupo=e["grupo"].values).groupby("grupo").mean()

    st.subheader("Comparación de perfiles")
    cuenta = e["grupo"].value_counts()
    perfil = medias.round(1).copy()
    perfil.insert(0, "Zonas", cuenta.reindex(perfil.index))
    perfil.insert(1, "Se distingue por", [describir_grupo(medias_z.loc[g]) for g in perfil.index])
    perfil["Actividad comercial"] = np.expm1(medias["Actividad comercial"]).round(0).astype(int)
    perfil = perfil.rename(columns={"Actividad comercial": "Establecimientos típicos"}).reset_index().rename(columns={"grupo": "Grupo"})
    tabla(perfil)
    fig = px.imshow(medias_z.round(2), text_auto=".2f", aspect="auto", zmin=-2, zmax=2,
                    color_continuous_scale=[[0, AZUL], [0.5, "#FFFFFF"], [1, AMARILLO]], title="Perfil de cada grupo (desviaciones estándar)")
    grafica(fig, 120 + 50 * len(medias_z))
    st.caption("Azul quiere decir por debajo del promedio de las zonas y amarillo por encima. Los nombres A, B, C ordenan los grupos por número de zonas y no por qué tan buenos son.")

    st.subheader("Dónde están")
    ee = e.assign(titulo=e["colonia"].str.title(), l2=e["grupo"], l3=e["zona"])
    letras = sorted(ee["grupo"].unique())
    hexa = {g: PALETA_GRUPOS[i % len(PALETA_GRUPOS)] for i, g in enumerate(letras)}
    ee["color"] = ee["grupo"].map(lambda g: [int(hexa[g][i:i + 2], 16) for i in (1, 3, 5)] + [215])
    ee["altura"] = 0
    vista = pdk.ViewState(latitude=20.99, longitude=-89.62, zoom=10.6, pitch=0)
    tip = {"html": "<b>{titulo}</b><br/>{l2}<br/>{l3}", "style": {"backgroundColor": NAVY, "color": "white", "fontSize": "12px"}}
    mapa(pdk.Deck(layers=[capa_zonas(ee[["zone_id", "color", "altura", "titulo", "l2", "l3"]], False, "perfiles")],
                  initial_view_state=vista, map_style=BASES["Mapa"], tooltip=tip), 520)
    st.markdown("".join(f"<span class='etiqueta' style='background:{hexa[g]}'>{g}</span>" for g in letras), unsafe_allow_html=True)
    st.caption("Las celdas grises claras que no aparecen tienen menos de 30 establecimientos y no entran al análisis.")


# Modelos y oportunidades
def pagina_modelos(denue):
    exigir_h3()
    st.title("Modelos y oportunidades")
    z = construir_zonas(denue.dropna(subset=["latitud", "longitud"]).reset_index(drop=True), "zonas")
    st.markdown(
        "El ranking de oportunidad es un modelo multicriterio y no un modelo predictivo. Cada zona recibe tres puntajes entre 0 y 100, que salen del lugar que ocupa entre todas las zonas "
        "(percentil), y se combinan con pesos que se pueden cambiar. Sirve para ordenar zonas con una regla transparente y no para pronosticar ventas."
    )
    industria = st.radio("Industria", TRES, horizontal=True, key="mod_ind")
    base_w = PESOS_BASE[industria]
    st.markdown("#### Pesos de los criterios")
    cols = st.columns(3)
    w = []
    for i, (cl, texto) in enumerate(CRITERIOS.items()):
        w.append(cols[i].slider(texto, 0.0, 1.0, float(base_w[i]), 0.05, key=f"w_{industria}_{cl}"))
    if sum(w) == 0:
        st.warning("Pon al menos un peso mayor que cero.")
        return
    st.caption("Los pesos se normalizan para sumar uno. Los valores iniciales son supuestos del equipo y no están estimados con datos.")
    p = puntuar(z, "zonas", industria, tuple(w))

    st.subheader("Ranking")
    top = p.sort_values("score", ascending=False).head(15)
    tabla(top.assign(
        Colonia=top["colonia"].str.title(), Sector=top["zona"], Oportunidad=top["score"].round(1), Lugar=top["rank"],
        Negocios=top["total"], **{"Competidores cerca": top["comp_" + industria],
                                          "Estable en el top 10 (%)": top["robustez"].round(0).astype(int),
                                          "Lugar mediano": top["rango_mediano"].round(0).astype(int)},
    )[["Lugar", "Colonia", "Sector", "Oportunidad", "Negocios", "Competidores cerca", "Estable en el top 10 (%)", "Lugar mediano"]])
    st.caption("Estable en el top 10 es el porcentaje de 1,000 combinaciones de pesos al azar en las que la zona queda entre las 10 mejores. Competidores cerca incluye la zona y sus 6 vecinas.")

    c1, c2 = st.columns(2)
    t10 = top.head(10).assign(Zona=lambda x: x["colonia"].str.title() + " " + x["zone_id"].str[-4:])
    with c1:
        fig = px.bar(t10.sort_values("score"), x="score", y="Zona", orientation="h", color_discrete_sequence=[AZUL],
                     labels={"score": "Oportunidad"}, title="Las 10 zonas con más oportunidad")
        grafica(fig, 420)
    with c2:
        fig = px.bar(t10.sort_values("robustez"), x="robustez", y="Zona", orientation="h", color_discrete_sequence=[AMARILLO],
                     labels={"robustez": "% de combinaciones de pesos en el top 10"}, title="Qué tan estable es cada zona")
        grafica(fig, 420)

    st.subheader("Por qué sale una zona")
    ids = top["zone_id"].tolist()
    nombre = {r.zone_id: f"{r.colonia.title()} · lugar {r.rank}" for r in top.itertuples()}
    zid = st.selectbox("Zona", ids, format_func=lambda i: nombre[i])
    r = p[p["zone_id"] == zid].iloc[0]
    wn = np.array(w) / sum(w)
    expl = pd.DataFrame({
        "Criterio": list(CRITERIOS.values()),
        "Puntaje (0 a 100)": [round(r["actividad"] * 100, 1), round(r["competencia"] * 100, 1), round(r["centralidad"] * 100, 1)],
        "Peso": [round(x, 2) for x in wn],
        "Aporta": [round(r[c] * 100 * wi, 1) for c, wi in zip(["actividad", "competencia", "centralidad"], wn)],
    })
    tabla(expl)
    fig = px.bar(expl, x="Aporta", y="Criterio", orientation="h", color_discrete_sequence=[AZUL], title=f"Puntos que aporta cada criterio, total {r['score']:.1f}")
    grafica(fig, 260)

    st.subheader("Mapa del puntaje")
    pp = p.assign(titulo=p["colonia"].str.title(), l2="Oportunidad " + p["score"].round(1).astype(str), l3="Lugar " + p["rank"].astype(str))
    pp["color"] = color_continuo((pp["score"] / 100).values)
    pp["altura"] = 0
    vista = pdk.ViewState(latitude=20.99, longitude=-89.62, zoom=10.6)
    tip = {"html": "<b>{titulo}</b><br/>{l2}<br/>{l3}", "style": {"backgroundColor": NAVY, "color": "white", "fontSize": "12px"}}
    mapa(pdk.Deck(layers=[capa_zonas(pp[["zone_id", "color", "altura", "titulo", "l2", "l3"]], False, "puntaje")],
                  initial_view_state=vista, map_style=BASES["Mapa"], tooltip=tip), 480)

    st.subheader("Cómo se construye y qué no hace")
    tabla(pd.DataFrame({
        "Criterio": list(CRITERIOS.values()),
        "Variable": ["Total de establecimientos de todos los giros en la zona", "Establecimientos de la industria en la zona y sus vecinas",
                     "Distancia en línea recta a la Plaza Grande"],
        "Dirección": ["Más es mejor", "Menos es mejor", "Más cerca es mejor"],
    }))
    st.markdown(
        f"Solo entran las zonas con al menos {MIN_EST} establecimientos de cualquier giro para no premiar celdas con muy pocos datos. Los puntajes son percentiles, por eso no se ven afectados por valores extremos. "
        "No se entrenó un modelo supervisado ni se calcularon valores SHAP porque no hay una variable real que predecir, como ventas o éxito de aperturas por zona. "
        "Tampoco se hizo un pronóstico, porque la serie de los Censos Económicos tiene solo tres puntos (2013, 2018 y 2023) y cualquier proyección sería inventada. "
        "El criterio de actividad premia zonas con mucho comercio, que a veces son las más saturadas, y por eso se compensa con el de competencia. "
        "En mascotas el modelo tiende a favorecer el Centro, que es una aproximación débil de la demanda de los hogares."
    )


# Decisión final
def pagina_decision(denue):
    exigir_h3()
    st.title("Decisión final")
    z = construir_zonas(denue.dropna(subset=["latitud", "longitud"]).reset_index(drop=True), "zonas")
    m4 = denue[denue["industria"].isin(TRES)]
    st.subheader("Industria prioritaria: cafeterías")
    st.markdown(
        "Se recomienda empezar por cafeterías. Es la industria que más creció en Mérida, tiene demanda sin cubrir en una parte importante de la ciudad y la gente que ya consume está contenta, "
        "así que la oportunidad está en operar mejor y no en inventar un producto distinto."
    )
    por_col = pd.crosstab(m4["nomb_asent"], m4["industria"])
    filas = []
    for ind, sub in zip(TRES, [35.6, 43.0, 74.7]):
        mediana, cerca = competidor_cercano(m4[m4["industria"] == ind])
        filas.append({"Industria": ind, "Establecimientos": int((m4["industria"] == ind).sum()),
                      "% de AGEB subatendidos": sub, "Mediana al competidor más cercano (m)": round(mediana),
                      "% con competidor a menos de 100 m": round(cerca, 1), "HHI por colonia": round(hhi(por_col[ind]))})
    tabla(pd.DataFrame(filas))
    st.caption("Los AGEB subatendidos salen del índice de saturación del análisis de Colab con los 435 AGEB. Los demás números se calculan en vivo con el DENUE.")
    st.markdown(
        "Antros y bares tiene el mayor porcentaje de AGEB subatendidos, pero es una industria muy concentrada en el Centro, donde vive poca gente y la demanda depende de visitantes, "
        "así que ese porcentaje no se lee como una oportunidad en barrios residenciales."
    )

    st.subheader("Tres zonas para cafeterías")
    p = puntuar(z, "zonas", "Cafeterías", PESOS_BASE["Cafeterías"]).sort_values("score", ascending=False)
    tres = p.drop_duplicates("colonia").head(3)
    tabla(tres.assign(
        Colonia=tres["colonia"].str.title(), Sector=tres["zona"], Oportunidad=tres["score"].round(1), Lugar=tres["rank"],
        Negocios=tres["total"], Cafeterías=tres["Cafeterías"], **{"Cafeterías cerca": tres["comp_Cafeterías"],
                                                                          "Estable en el top 10 (%)": tres["robustez"].round(0).astype(int)},
    )[["Colonia", "Sector", "Lugar", "Oportunidad", "Negocios", "Cafeterías", "Cafeterías cerca", "Estable en el top 10 (%)"]])
    st.caption("Se muestra la mejor zona de cada una de las tres colonias distintas mejor calificadas, con los pesos 0.4, 0.4 y 0.2 del equipo.")
    st.markdown(
        "Estas zonas salen de un modelo hecho solo con el DENUE y no incluyen población ni ingreso. El análisis de Colab con AGEB señala otras áreas candidatas, sobre todo hacia Caucel, Paseos de Opichén y Chuburná de Hidalgo, "
        "donde hay mucha población y menos cafeterías. Ambas lecturas no coinciden por completo, y lo sensato es validar primero las zonas donde las dos apunten en la misma dirección."
    )

    st.subheader("Alternativas")
    st.markdown(
        "Mascotas queda como segunda opción. Hay hacia dónde crecer, el entorno ya muestra movimiento, con una clínica de cadena en construcción y un reglamento de bienestar animal en discusión, "
        "pero el reglamento puede cambiar los requisitos para operar y todavía no está aprobado. Antros y bares se descarta como primera apuesta por su concentración en el Centro y porque no hay señales verificables de inversión en esa industria."
    )

    st.subheader("Incertidumbres")
    st.markdown(
        "Las zonas usan solo el DENUE, que es un directorio y no mide ventas, y puede tener negocios cerrados o mal ubicados. Los pesos del ranking son supuestos del equipo, aunque se probó su sensibilidad. "
        "La percepción del cliente se basa en 600 reseñas de 20 cafeterías, que son pocas y casi todas positivas. Las zonas del análisis de población se asignaron por posición respecto a la Plaza Grande. "
        "Además, el entorno económico trae señales mixtas, con inflación arriba del promedio nacional, menos asegurados al IMSS y menor ocupación hotelera, aunque el sector restaurantero sigue creciendo en afiliados."
    )

    st.subheader("Próximos pasos")
    st.markdown(
        "Traer la población por AGEB y su geometría para cruzar demanda y zonas H3. Visitar en campo las tres zonas para verificar que los negocios existan y medir tráfico. "
        "Entrenar un modelo supervisado cuando el cliente comparta datos reales de desempeño de sus sucursales. Conseguir reseñas autorizadas de mascotas y bares para comparar percepción entre industrias. "
        "Replicar el método en Torreón y Querétaro."
    )



# Navegación
def vista(funcion, necesita_denue=True):
    def _pagina():
        if necesita_denue:
            funcion(obtener_denue())
        else:
            funcion()
    _pagina.__name__ = funcion.__name__
    return _pagina


LOGO_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 392 136"><defs><clipPath id="c"><polygon points="111.0,68.0 92.5,100.0 55.5,100.0 37.0,68.0 55.5,36.0 92.5,36.0"/></clipPath></defs><polygon points="140.0,68.0 107.0,125.2 41.0,125.2 8.0,68.0 41.0,10.8 107.0,10.8" fill="none" stroke="#325596" stroke-width="8" stroke-linejoin="miter"/><polygon points="125.0,68.0 99.5,112.2 48.5,112.2 23.0,68.0 48.5,23.8 99.5,23.8" fill="none" stroke="#325596" stroke-width="6" stroke-linejoin="miter"/><g clip-path="url(#c)" stroke="#9FB3D9" stroke-width="1.6"><line x1="46" y1="23" x2="46" y2="113"/><line x1="65" y1="23" x2="65" y2="113"/><line x1="83" y1="23" x2="83" y2="113"/><line x1="102" y1="23" x2="102" y2="113"/><line x1="29" y1="46" x2="119" y2="46"/><line x1="29" y1="64" x2="119" y2="64"/><line x1="29" y1="82" x2="119" y2="82"/><line x1="29" y1="98" x2="119" y2="98"/></g><path d="M74 98 L59 66 A17 17 0 1 1 89 66 Z" fill="#FFA800"/><circle cx="74" cy="56" r="6.5" fill="#fff"/><path d="M173.9 99.2C187.0 99.2 195.1 87.4 195.1 68.0C195.1 48.6 187.0 36.8 173.9 36.8L158.0 36.8L158.0 99.2L173.9 99.2ZM168.2 86.5L168.2 49.5L173.3 49.5C180.8 49.5 184.9 54.5 184.9 68.0C184.9 81.5 180.8 86.5 173.4 86.5L168.2 86.5ZM196.1 99.2L207.5 99.2L210.3 85.9L224.6 85.9L227.6 99.2L239.0 99.2L224.2 36.8L210.5 36.8L196.1 99.2ZM212.7 74.6L213.3 71.9C214.7 64.9 216.0 57.0 217.2 48.6C218.6 57.0 220.1 64.8 221.6 71.9L222.1 74.6L212.7 74.6ZM236.9 48.9L250.0 48.9L250.0 99.2L260.2 99.2L260.2 48.9L273.2 48.9L273.2 36.8L236.9 36.8L236.9 48.9ZM278.5 99.2L306.7 99.2L306.7 87.1L288.7 87.1L288.7 36.8L278.5 36.8L278.5 99.2ZM310.0 99.2L321.3 99.2L324.1 85.9L338.5 85.9L341.4 99.2L352.9 99.2L338.1 36.8L324.4 36.8L310.0 99.2ZM326.5 74.6L327.1 71.9C328.5 64.9 329.8 57.0 331.1 48.6C332.5 57.0 333.9 64.8 335.4 71.9L336.0 74.6L326.5 74.6ZM373.0 100.0C383.6 100.0 390.0 92.9 390.0 81.1C390.0 71.1 385.2 65.1 376.8 62.5L373.1 61.2C369.2 60.0 366.7 58.1 366.7 54.3C366.7 50.8 368.9 48.3 372.9 48.3C377.0 48.3 379.3 50.9 379.6 55.4L389.4 55.4C389.3 43.9 382.9 36.0 373.0 36.0C363.2 36.0 356.2 43.7 356.2 55.3C356.2 64.7 360.7 70.0 368.1 72.5L372.7 74.1C377.4 75.6 379.7 77.5 379.7 81.0C379.7 85.0 377.1 87.7 372.8 87.7C368.2 87.7 365.1 84.6 364.9 78.6L355.0 78.6C355.2 93.0 362.2 100.0 373.0 100.0Z" fill="#325596"/></svg>'''
st.markdown(
    "<style>[data-testid='stSidebarNav']::before {content:''; display:block; height:64px; margin:-1.3rem 0 1.7rem 0; "
    "background:url(\"data:image/svg+xml;utf8," + quote(LOGO_SVG) + "\") no-repeat left center / contain;}</style>",
    unsafe_allow_html=True,
)

paginas = [
    st.Page(vista(pagina_inicio, False), title="Inicio", icon=":material/home:", url_path="inicio", default=True),
    st.Page(vista(pagina_2), title="Ciudad", icon=":material/groups:", url_path="ciudad"),
    st.Page(vista(pagina_industrias), title="Industrias", icon=":material/storefront:", url_path="industrias"),
    st.Page(vista(pagina_explorador), title="Mapa", icon=":material/map:", url_path="mapa"),
    st.Page(vista(pagina_perfiles), title="Perfiles", icon=":material/hub:", url_path="perfiles"),
    st.Page(vista(pagina_modelos), title="Oportunidades", icon=":material/insights:", url_path="oportunidades"),
    st.Page(vista(pagina_decision), title="Decisión", icon=":material/task_alt:", url_path="decision"),
]
navegacion = st.navigation(paginas, position="sidebar")
st.sidebar.markdown(
    "<div style='margin-top:1.2rem;font-size:0.68rem;line-height:1.35;color:#6B7280'><b>Fuentes</b><br>"
    "DENUE · Censos Económicos (SAIC) · Censo de Población 2020 · Encuesta Intercensal 2025 · ANUIES · "
    "Perfiles de turista · Reseñas de Google Maps · Notas de prensa</div>",
    unsafe_allow_html=True,
)
navegacion.run()