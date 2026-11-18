"""¿Qué sismos marca la USGS con alerta de tsunami? — Proyecto Integrador, Grupo 9 (UTN FRM, 2026).

Aplicación de Streamlit con forma de copa de martini:
  portada → tallo (los datos, lo que encontramos, el modelo) → copa (explorá, probá el modelo).

Lee lo que genera build_app.py en app_data/. Para correrla:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

import modelo  # define EsOceanico: joblib la necesita para cargar el pipeline

BASE = Path(__file__).parent
APP_DATA = BASE / "app_data"
MUNDO = alt.topo_feature("https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/world-110m.json", "countries")
alt.data_transformers.disable_max_rows()   # los mapas tienen ~13.600 sismos

# --- Colores: significan lo mismo en toda la aplicación -----------------------------------------
AZUL, AZUL_CLARO = "#2a78d6", "#9cc3ef"          # con marca (claro: con marca y magnitud 6 o más, en el mapa)
GRIS_CLARO, GRIS = "#cfcac3", "#7a7974"          # sin marca · sin marca cargada todavía
NARANJA, VIOLETA = "#d95926", "#8a5cc2"          # falso negativo · falsa alarma
MAR, TIERRA = "#1b9e8f", "#b8862b"               # dónde está el epicentro
NEUTRO = "#4a5560"                               # barras que no son una clase (distribuciones, importancias)
PROFUNDIDAD = ["#cfe8c4", "#7fbf6a", "#2e7d32"]  # superficial · intermedio · profundo

RESULTADOS = ["acierto: con marca", "falso negativo", "falsa alarma", "acierto: sin marca", "sin marca cargada"]
ESCALA_RESULTADO = alt.Scale(domain=RESULTADOS, range=[AZUL, NARANJA, VIOLETA, GRIS_CLARO, GRIS])
CON, SIN = "con marca de la USGS", "sin marca de la USGS"   # las dos clases, con el recordatorio de qué es la marca
ESCALA_MARCA = alt.Scale(domain=[CON, SIN], range=[AZUL, GRIS_CLARO])
ESCALA_ZONA = alt.Scale(domain=["mar", "tierra"], range=[MAR, TIERRA])

# Regiones donde la USGS usa un umbral regional (sección 2.1 de la Entrega 3)
REGION_ALASKA_OESTE = ["Alaska", "Oregon", "off the coast of Oregon", "Canada", "CA", "California", "OR",
                       "south of Alaska"]

# Rangos de magnitud y de distancia que usan los gráficos de «Lo que encontramos»
CORTES_MAG = [4.99, 5.5, 6.0, 6.5, 7.0, 7.5, 9.5]
RANGOS_MAG = ["5,0–5,5", "5,5–6,0", "6,0–6,5", "6,5–7,0", "7,0–7,5", "más de 7,5"]
CORTES_PROF = [-1, 70, 300, 1000]
RANGOS_PROF = ["superficial (menos de 70 km)", "intermedio (70 a 300 km)", "profundo (más de 300 km)"]

# Tipos de magnitud de la USGS: el nombre que muestra el selector de «Probá el modelo» y la explicación de su cuadro
NOMBRES_MAGTYPE = {
    "mww": "Magnitud de momento, fase W (la estándar de la USGS)",
    "mb": "Magnitud de ondas de cuerpo (sismos moderados y lejanos)",
    "mw": "Magnitud de momento, informada por otra red",
    "mwr": "Magnitud de momento regional",
    "ml": "Magnitud local, tipo Richter (redes regionales)",
    "mwb": "Magnitud de momento con ondas de cuerpo",
}
EXPLICACION_MAGTYPE = {  # sigla: (cómo se calcula, cuándo la usa la USGS)
    "mww": ("Calcula el momento sísmico (qué tan grande es la falla que se rompió y cuánto se deslizó) con la fase W, "
            "una onda de período muy largo que registran estaciones de todo el mundo.",
            "Es la que la USGS prefiere para la mayoría de los sismos medianos y grandes, y la primera que publica en "
            "los grandes. Por eso es la de casi todos los sismos con marca."),
    "mb": ("Mide la amplitud de las ondas P (las primeras en llegar) en sismógrafos lejanos al sismo.",
           "Sismos moderados y lejanos que todavía no tienen magnitud de momento. Se satura cerca de 6: en sismos "
           "más grandes deja de crecer, así que no sirve para medirlos."),
    "mw": ("Calcula el momento sísmico, pero el dato lo aporta otra agencia o red sismológica.",
           "Cuando la magnitud preferida del catálogo viene de un contribuyente externo, por ejemplo una red de "
           "California."),
    "mwr": ("Calcula el momento sísmico con ondas registradas por estaciones de la región del sismo.",
            "Sismos moderados en zonas con buena red regional (en los datos: Chile, Alaska y Taiwán, entre otras)."),
    "ml": ("Es la escala original de Richter: mide la amplitud de las ondas en estaciones cercanas.",
           "La informan redes regionales (en los datos: Alaska, Puerto Rico, Hawái y Nevada). También se satura en "
           "sismos grandes."),
    "mwb": ("Calcula el momento sísmico con ondas de cuerpo de período largo.",
            "Una alternativa a mww para sismos moderados a grandes."),
}


# --- Formato en castellano ----------------------------------------------------------------------
def pct(x, dec=0):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def num(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


def grafico(chart):
    """Todos los gráficos van sin el tema de Streamlit, para que los colores sean los de arriba."""
    st.altair_chart(chart, theme=None)


# --- Las funciones de la cátedra (Entrega 2) ----------------------------------------------------
CORTES_SEMAFORO = {"separacion": (0.2, 0.8), "correlacion": (0.2, 0.6)}


def separacion(x, y):
    """Distancia entre dos muestras, en desviaciones estándar."""
    x, y = x.dropna(), y.dropna()
    return abs(x.mean() - y.mean()) / np.sqrt((x.std() ** 2 + y.std() ** 2) / 2)


def semaforo(valor, medida="separacion"):
    bajo, alto = CORTES_SEMAFORO[medida]
    v = abs(valor)
    return "roja" if v < bajo else "verde" if v > alto else "amarilla"


def asociacion(x, y):
    """Pearson, Spearman y la brecha entre ambos."""
    p, s = x.corr(y), x.corr(y, method="spearman")
    return p, s, s - p


# --- Carga ---------------------------------------------------------------------------------------
@st.cache_data
def cargar_sismos():
    return pd.read_csv(APP_DATA / "sismos_app.csv")


@st.cache_data
def cargar_resumen():
    with open(APP_DATA / "resumen.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo():
    return joblib.load(APP_DATA / "modelo.joblib")


st.set_page_config(page_title="Sismos y marca de tsunami · Grupo 9", layout="wide")

if not (APP_DATA / "sismos_app.csv").exists():
    st.error("Faltan los datos de la aplicación: correr primero `python build_app.py`.")
    st.stop()

sismos = cargar_sismos().round({"latitude": 2, "longitude": 2, "depth": 1})   # menos decimales = página más liviana
resumen = cargar_resumen()
guardado = cargar_modelo()
exp = resumen["experimentos"]
UMBRAL = guardado["umbral"]
modelados = sismos[sismos["conjunto"] != "sin marca cargada"].copy()   # los que usa el modelo (hasta marzo de 2026)
modelados["marca"] = modelados["tsunami"].map({1: CON, 0: SIN})
con_marca, sin_marca = modelados[modelados["tsunami"] == 1], modelados[modelados["tsunami"] == 0]
test = resumen["test"]
vc = resumen["validacion_cruzada"]
vc05 = resumen["validacion_cruzada_umbral_05"]
fondo = alt.Chart(MUNDO).mark_geoshape(fill="#eeeeee", stroke="white", strokeWidth=0.5)

if guardado.get("version_sklearn") != sklearn.__version__:
    st.warning(f"El modelo se guardó con scikit-learn {guardado.get('version_sklearn')} y acá corre "
               f"{sklearn.__version__}. Instalar la versión de requirements.txt.")


# =================================================================================================
# PORTADA
# =================================================================================================
st.caption("Ciencia de Datos · UTN FRM · 2026 · Proyecto Integrador · Grupo 9 · Comisión 5K09")
st.title("¿Qué características de un sismo permiten estimar si está asociado a un tsunami?")

with st.container(border=True):
    st.markdown("#### Antes de empezar: qué significa `tsunami = 1` en estos datos")
    es, no_es = st.columns(2)
    es.info("**Es la marca de alerta que pone la USGS** en su catálogo cuando un sismo cumple su criterio de alerta "
            "de tsunami. Ese criterio depende de la magnitud **y de la región**: cerca de Alaska y de la costa oeste "
            "de Norteamérica, donde la USGS opera un sistema de alerta propio, marca sismos más chicos que en el resto "
            "del mundo.")
    no_es.warning("**No es la confirmación de que hubo una ola.** La marca no mide el mar: un sismo puede tenerla sin "
                  "que se haya registrado un tsunami. Por eso todo lo que sigue habla de **sismos con "
                  "marca de la USGS**, y el modelo predice esa marca, no la ola.")

st.markdown(f"#### Con la magnitud y el lugar del sismo se anticipa el {pct(test['recall_tsunami'])} de los sismos "
            "que la USGS marca con alerta de tsunami, y el lugar pesa tanto como la magnitud.")

chicos = modelados[(modelados["tsunami"] == 1) & (modelados["mag"] < 6)]
c1, c2, c3 = st.columns(3)
c1.metric("sismos con marca de la USGS detectados en el test",
          f"{test['aciertos_con_marca']} de {test['aciertos_con_marca'] + test['falsos_negativos']}",
          help=f"Recall de la clase con marca, con umbral {num(UMBRAL, 2)}")
c2.metric("f1 macro en el test", num(test["f1_macro"]),
          f"{num(test['f1_macro'] - resumen['piso_test']['f1_macro'])} sobre el piso", delta_color="off")
c3.metric("marcas con magnitud menor a 6 que están en Alaska y la costa oeste",
          f"{chicos['region'].isin(REGION_ALASKA_OESTE).sum()} de {len(chicos)}")
st.caption(f"Medido sobre {miles(len(modelados))} sismos de magnitud 5 o más registrados por la USGS entre "
           f"{pd.Timestamp(resumen['datos']['desde']).strftime('%d/%m/%Y')} y el 31/03/2026.")

st.divider()


# =================================================================================================
# TALLO 1 · LOS DATOS
# =================================================================================================
st.header("Los datos")
n_total, n_marca = len(sismos), int(sismos["tsunami"].sum())
st.subheader(f"De {miles(n_total)} sismos fuertes registrados desde 2019, sólo {miles(n_marca)} llevan la marca de "
             f"tsunami: 1 de cada {round(n_total / n_marca)}")

clases = pd.DataFrame({"marca": [SIN, CON], "sismos": [n_total - n_marca, n_marca]})
clases["texto"] = [f"{miles(int(n))} ({pct(n / n_total, 1)})" for n in clases["sismos"]]
barras_clases = alt.Chart(clases).mark_bar(size=46).encode(
    y=alt.Y("marca:N", title=None, sort=[SIN, CON], axis=alt.Axis(labelFontSize=13)),
    x=alt.X("sismos:Q", title="sismos", scale=alt.Scale(domain=[0, n_total * 1.12])),
    color=alt.Color("marca:N", scale=ESCALA_MARCA, legend=None),
    tooltip=[alt.Tooltip("marca:N", title=" "), alt.Tooltip("sismos:Q")])
texto_clases = barras_clases.mark_text(align="left", dx=6, fontSize=14, fontWeight="bold", color="#222").encode(
    text="texto:N")
grafico((barras_clases + texto_clases).properties(width=900, height=150))
st.markdown(f"Cada fila es un sismo de magnitud 5 o más del catálogo de la USGS, y la columna `tsunami` dice si tiene "
            f"la marca. **La diferencia es enorme**: un modelo que diga siempre «sin marca» acierta el "
            f"{pct(1 - n_marca / n_total)} sin aprender nada. Por eso la accuracy no sirve para medir el modelo, y se usa "
            "el f1 macro, que le da el mismo peso a la clase chica.")

d1, d2, d3, d4 = st.columns(4)
d1.metric("columnas", resumen["datos"]["columnas"])
d2.metric("desde", pd.Timestamp(resumen["datos"]["desde"]).strftime("%d/%m/%Y"))
d3.metric("hasta", pd.Timestamp(resumen["datos"]["hasta"]).strftime("%d/%m/%Y"))
d4.metric("sismos duplicados", resumen["datos"]["ids_duplicados"])

st.markdown("##### Los sismos en el tiempo")
mensual = sismos.groupby("mes", as_index=False).agg(sismos=("id", "count"), con_marca=("tsunami", "sum"))
mensual["fecha"] = pd.to_datetime(mensual["mes"] + "-01")
picos = mensual.nlargest(2, "sismos").copy()
picos["etiqueta"] = [f"{r.sismos} sismos: réplicas en {sismos.loc[sismos['mes'] == r.mes, 'region'].mode()[0]}"
                     for r in picos.itertuples()]
n_recientes = int((sismos["conjunto"] == "sin marca cargada").sum())
sombra = pd.DataFrame({"desde": [pd.Timestamp(modelo.CORTE)],
                       "hasta": [mensual["fecha"].max() + pd.offsets.MonthBegin(1)],
                       "t": [f"sin marca cargada ({n_recientes} sismos)"]})
eje_t = alt.X("fecha:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
zona_gris = alt.Chart(sombra).mark_rect(color=GRIS, opacity=0.18).encode(x="desde:T", x2="hasta:T")
linea_t = alt.Chart(mensual).mark_line(color=NEUTRO, point=alt.OverlayMarkDef(color=NEUTRO, size=16)).encode(
    x=eje_t, y=alt.Y("sismos:Q", title="sismos por mes"),
    tooltip=[alt.Tooltip("mes:N"), alt.Tooltip("sismos:Q"), alt.Tooltip("con_marca:Q", title="con marca")])
etiquetas_t = alt.Chart(picos).mark_text(align="right", dx=-8, dy=-2, fontSize=11, color="#222").encode(
    x="fecha:T", y="sismos:Q", text="etiqueta:N")
texto_sombra = alt.Chart(sombra).mark_text(align="right", dx=-4, y=12, fontSize=11, color=GRIS).encode(
    x="hasta:T", text="t:N")
arriba = (zona_gris + linea_t + etiquetas_t + texto_sombra).properties(width=900, height=210)
barras_t = alt.Chart(mensual).mark_bar(color=AZUL).encode(
    x=eje_t, y=alt.Y("con_marca:Q", title="con marca por mes"),
    tooltip=[alt.Tooltip("mes:N"), alt.Tooltip("con_marca:Q", title="con marca")])
abajo = (zona_gris + barras_t).properties(width=900, height=110)
grafico(alt.vconcat(arriba, abajo, spacing=6))
st.markdown(f"El catálogo trae unos {round(mensual['sismos'].median())} sismos por mes, con dos cosas para tener en "
            "cuenta:  \n"
            "- **Los picos son secuencias de réplicas**: decenas de sismos en el mismo lugar después de uno grande. Las "
            "filas no son independientes, y por eso la validación del modelo se arma por grupos de lugar y mes.  \n"
            "- **Desde abril de 2026 la marca todavía no está cargada** (barras azules en cero): la USGS la carga cuando "
            f"revisa el evento, y la descarga tomó esos sismos antes. Por eso el modelo usa sólo los "
            f"{miles(len(modelados))} sismos hasta el 31/03/2026.")

st.markdown("##### Las columnas del dataset")
filas_columnas = [
    ("mag", "numérica", "Magnitud del sismo", "Entra al modelo"),
    ("latitude, longitude", "numérica", "Ubicación del epicentro", "Entran al modelo"),
    ("depth", "numérica", "Profundidad en km (46 % vale 10 km: el valor que fija la USGS cuando no puede medirla)",
     "Entra al modelo"),
    ("magType", "categórica", "Método con que se calculó la magnitud (mww, mb, ml…)", "Entra al modelo (one-hot)"),
    ("dist_costa_km", "numérica", "Distancia a la costa, con signo (columna propia del pipeline)",
     "Entra al modelo como mar o tierra"),
    ("tsunami", "categórica (0/1)", "La marca de alerta de la USGS", "Es lo que se predice"),
    ("sig", "numérica", "«Significancia» del evento según la USGS", "Sale: fuga (contiene la alerta)"),
    ("nst, dmin, gap, rms", "numéricas", "Cómo registró el sismo la red de estaciones",
     "Salen: no mejoran el modelo"),
    ("time", "temporal", "Fecha y hora (UTC)", "Sólo arma los grupos de validación"),
    ("place", "texto libre", "Descripción del lugar (de ahí sale la región)", "Sale: identificador"),
    ("id, type, status, net", "identificador / categóricas",
     "Código del sismo, tipo de evento, estado de revisión y red", "Salen: identificador o constantes"),
]
tabla_columnas = pd.DataFrame(filas_columnas, columns=["columna", "tipo", "qué es", "uso"])
nulos_por = {c: sismos[c].isna().mean() for c in ["nst", "dmin", "gap", "rms"]}
tabla_columnas["nulos"] = np.where(tabla_columnas["columna"] == "nst, dmin, gap, rms",
                                   f"nst {pct(nulos_por['nst'])}, el resto menos de 1 %", "—")
st.dataframe(tabla_columnas, hide_index=True)
st.caption("Los únicos nulos están en las cuatro columnas de la red de estaciones: son de instrumentación (muchas "
           "redes no informan `nst`), no errores de carga. Las otras 14 columnas están completas.")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**De dónde salen.** De la API pública de la USGS (`earthquake.usgs.gov/fdsnws/event/1/query`), en GeoJSON y sin clave.
Los baja un pipeline de Airflow con arquitectura Medallón:

1. `wait_for_source`: un sensor que pregunta cuántos sismos hay en la ventana; si hay cero, saltea el resto.
2. `extract` (Bronce): guarda el GeoJSON crudo, sin tocarlo, con las fechas de la ventana en el nombre.
3. `transform` (Plata): lo aplana en una tabla de 17 columnas, un sismo por fila.
4. `enrich` (Plata): agrega nuestra columna, `dist_costa_km`.
5. `validate`: corta el pipeline si el `id` se repite, si hay menos de 1.000 filas, menos de 5 columnas o alguna
   columna vacía del todo.
6. `save` (Oro): agrega los sismos a `sismos_master.csv` y reemplaza los que ya estaban por `id` (*upsert*), así
   las correcciones de la USGS pisan la versión vieja.
7. `notify`: avisa por Discord que terminó.

Corre **todos los días a las 02:00** y trae los sismos del día anterior. Esta aplicación usa una copia congelada de esa
salida (`{resumen['datos']['archivo']}`, del {resumen['datos']['desde']} al {resumen['datos']['hasta']}), para que todos
los números se puedan reproducir.

**Qué es una fila.** Un sismo de magnitud 5 o más. Por debajo de 5 la posibilidad de tsunami es prácticamente nula.

**Nuestra columna, `dist_costa_km`.** La distancia del epicentro a la costa más cercana, con signo:
- la distancia sale de la fórmula de Haversine, que tiene en cuenta la curvatura de la Tierra, contra la línea de costa
  de Natural Earth (escala 50 m);
- tierra o mar sale de *ray casting*: se traza un rayo desde el epicentro y se cuenta cuántas veces cruza el borde de
  los polígonos de tierra (impar = adentro de un continente);
- negativa en tierra, positiva en el mar.

**Qué se filtró y qué quedó.**
""")
    st.dataframe(pd.DataFrame(resumen["filtros"]).rename(
        columns={"paso": "Paso", "filas": "Sismos", "con_marca": "Con marca"}), hide_index=True)
    sep_sig = separacion(con_marca["sig"], sin_marca["sig"])
    _, s_ms, _ = asociacion(modelados["mag"], modelados["sig"])
    con_sig = {a["variante"]: a for a in exp["ablacion"]}["+ sig (fuga)"]
    base_ = {a["variante"]: a for a in exp["ablacion"]}["base"]
    st.markdown(f"""
**La columna descartada por fuga: `sig`.** Es un índice de la USGS que separa muy bien las dos clases (separación
{num(sep_sig, 2)}, zona {semaforo(sep_sig)}), pero porque la USGS lo calcula **usando la alerta de tsunami**: contiene la
respuesta. El número solo no lo delataba: su correlación con la marca es apenas
{num(modelados['tsunami'].corr(modelados['sig']), 2)}, y es casi una copia de la magnitud (Spearman {num(s_ms, 2)}). La
fuga se detecta leyendo cómo se construye la columna. En el modelo la medimos igual: con `sig` el f1 da
{num(con_sig['f1'])} y sin ella {num(base_['f1'])}; sale por cómo se construye, no por el número.

**Las demás columnas que salen:** `id`, `place` y `time` son identificadores (`time` sólo arma los grupos de
validación); `status` tiene un solo valor, `type` es «earthquake» en casi todas las filas y `net` es «us» en el
{pct((sismos['net'] == 'us').mean(), 1)}; `nst`, `dmin`, `gap` y `rms` no mejoran el modelo y la USGS las recalcula al
revisar el evento.
""")

st.divider()


# =================================================================================================
# TALLO 2 · LO QUE ENCONTRAMOS
# =================================================================================================
st.header("Lo que encontramos")
st.subheader("La magnitud es la que más separa, pero no es una regla: ni los sismos más grandes llevan siempre la marca")

modelados["rango_mag"] = pd.cut(modelados["mag"], CORTES_MAG, labels=RANGOS_MAG)
por_mag = (modelados.groupby(["rango_mag", "en_el_mar"], observed=True)["tsunami"]
           .agg(sismos="size", con_marca="sum").reset_index())
por_mag["proporcion"] = por_mag["con_marca"] / por_mag["sismos"]
por_mag["rango_mag"] = por_mag["rango_mag"].astype(str)
lineas_mag = alt.Chart(por_mag).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=3).encode(
    x=alt.X("rango_mag:O", sort=RANGOS_MAG, title="magnitud", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("proporcion:Q", title="sismos con marca de la USGS", axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("en_el_mar:N", title="epicentro", scale=ESCALA_ZONA, legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("rango_mag:O", title="magnitud"), alt.Tooltip("en_el_mar:N", title="epicentro"),
             alt.Tooltip("sismos:Q"), alt.Tooltip("con_marca:Q", title="con marca"),
             alt.Tooltip("proporcion:Q", title="con marca", format=".1%")])
etiquetas_mag = lineas_mag.mark_text(dy=-12, fontSize=11).encode(text=alt.Text("proporcion:Q", format=".0%"))
regla = alt.Chart(pd.DataFrame({"y": [1.0], "texto": ["si la magnitud fuera una regla, todo llegaría acá (100 %)"]}))
regla_linea = regla.mark_rule(color=GRIS, strokeDash=[6, 4]).encode(y="y:Q")
regla_texto = regla.mark_text(align="left", x=5, dy=-8, color=GRIS, fontSize=11).encode(y="y:Q", text="texto:N")
grafico((lineas_mag + etiquetas_mag + regla_linea + regla_texto).properties(width=900, height=330))

ult = por_mag[por_mag["rango_mag"] == RANGOS_MAG[-1]].set_index("en_el_mar")
st.markdown(f"La marca sube con la magnitud: la tienen menos de 2 de cada 100 sismos de 5,0 a 5,5 y "
            f"{pct(ult.loc['mar', 'proporcion'])} de los de más de 7,5 en el mar. Pero **nunca llega al 100 %**, y en "
            f"tierra queda más abajo ({pct(ult.loc['tierra', 'proporcion'])}, sobre sólo "
            f"{int(ult.loc['tierra', 'sismos'])} sismos).")

st.markdown("##### ¿Qué separa a los sismos con marca de los sin marca?")
mar_ = modelados[modelados["dist_costa_km"] > 0]
seps = pd.DataFrame([
    ("Magnitud", separacion(con_marca["mag"], sin_marca["mag"]), "se usa"),
    ("sig (índice de la USGS)", separacion(con_marca["sig"], sin_marca["sig"]), "fuga: se descarta"),
    ("Distancia a la costa, sólo en el mar",
     separacion(mar_.loc[mar_["tsunami"] == 1, "dist_costa_km"], mar_.loc[mar_["tsunami"] == 0, "dist_costa_km"]),
     "se usa"),
    ("Distancia a la costa", separacion(con_marca["dist_costa_km"], sin_marca["dist_costa_km"]), "se usa"),
    ("Profundidad", separacion(con_marca["depth"], sin_marca["depth"]), "se usa"),
], columns=["variable", "separacion", "estado"])
seps["texto"] = [f"{num(v, 2)}" + ("  · fuga: se descarta" if e != "se usa" else "")
                 for v, e in zip(seps["separacion"], seps["estado"])]
orden_vars = seps["variable"].tolist()
x_max = 1.6
franjas = pd.DataFrame({"desde": [0, 0.2, 0.8], "hasta": [0.2, 0.8, x_max],
                        "zona": ["zona roja", "zona amarilla", "zona verde"]})
fondo_sem = alt.Chart(franjas).mark_rect(opacity=0.35).encode(
    x=alt.X("desde:Q", title="separación (desvíos entre el promedio con marca y sin marca)",
            scale=alt.Scale(domain=[0, x_max])), x2="hasta:Q",
    color=alt.Color("zona:N", title="semáforo de la cátedra", legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=["zona roja", "zona amarilla", "zona verde"],
                                    range=["#f4b6b0", "#f7e3a1", "#bfe3c0"])))
barras_sep = alt.Chart(seps).mark_bar(size=22).encode(
    y=alt.Y("variable:N", sort=orden_vars, title=None, axis=alt.Axis(labelFontSize=12)),
    x="separacion:Q",
    color=alt.Color("estado:N", scale=alt.Scale(domain=["se usa", "fuga: se descarta"], range=[NEUTRO, "#b9b6b0"]),
                    legend=None),
    tooltip=["variable:N", alt.Tooltip("separacion:Q", format=".3f"), "estado:N"])
texto_sep = barras_sep.mark_text(align="left", dx=5, color="#222", fontSize=12).encode(text="texto:N")
grafico((fondo_sem + barras_sep + texto_sep).properties(width=900, height=230))

tipos_m = modelados.groupby("magType")["tsunami"].agg(sismos="size", con_marca="sum")
st.markdown(
    f"**La magnitud es la única variable física que separa bien.** Las otras dos que dicta la intuición física casi no "
    f"separan: los sismos con marca no son más superficiales que los sin marca, y la distancia a la costa distingue algo "
    f"sólo si se mira dentro del mar. `sig` separa muy bien, pero porque la USGS la calcula con la alerta: es una fuga y "
    f"se descartó (sección «Los datos»). Además, el **tipo de magnitud** cambia mucho las cosas: de los "
    f"{miles(int(tipos_m.loc['mb', 'sismos']))} sismos medidos en `mb` sólo {int(tipos_m.loc['mb', 'con_marca'])} "
    f"tienen marca, contra {pct(tipos_m.loc['mww', 'con_marca'] / tipos_m.loc['mww', 'sismos'], 1)} en `mww`.  \n"
    "Lo que el análisis no podía explicar es por qué hay sismos grandes sin marca y sismos chicos con marca. "
    "Eso lo resolvió el modelo: **el lugar**.")

with st.expander("Cómo lo sabemos"):
    st.markdown("**La separación** es la herramienta de la cátedra para medir una hipótesis con un número:")
    st.latex(r"\text{separación} = \frac{|\bar{x}_{\text{con marca}} - \bar{x}_{\text{sin marca}}|}"
             r"{\sqrt{(s^2_{\text{con marca}} + s^2_{\text{sin marca}})/2}}")
    grandes = modelados[modelados["mag"] > 7.5]
    excepciones = grandes[grandes["tsunami"] == 0]
    sep_mag = separacion(con_marca["mag"], sin_marca["mag"])
    sep_prof = separacion(con_marca["depth"], sin_marca["depth"])
    prof = modelados.assign(rango=pd.cut(modelados["depth"], CORTES_PROF, labels=RANGOS_PROF))
    reparto = prof.groupby(["marca", "rango"], observed=True).size().reset_index(name="sismos")
    reparto["parte"] = reparto["sismos"] / reparto.groupby("marca")["sismos"].transform("sum")
    reparto["rango"] = reparto["rango"].astype(str)
    reparto["orden"] = reparto["rango"].map({r_: i for i, r_ in enumerate(RANGOS_PROF)})
    sup = reparto[reparto["orden"] == 0].set_index("marca")["parte"]
    st.markdown(f"""
Es la distancia entre los dos promedios medida en desvíos. Menos de 0,2 es zona roja, de 0,2 a 0,8 amarilla y más de 0,8
verde. Un verde no alcanza para usar una columna (`sig`), y un rojo no alcanza para descartarla (la profundidad entró
igual, por motivo físico).

**La hipótesis que se sostuvo, como tendencia: la magnitud** (separación {num(sep_mag, 2)}, zona {semaforo(sep_mag)}).
La pregunta era si todo sismo de más de 7,5 tiene marca: {len(excepciones)} de los {len(grandes)}
({pct(len(excepciones) / len(grandes), 1)}) no la tienen. Buena variable, mal umbral fijo.

**La que no se sostuvo: la profundidad** (separación {num(sep_prof, 3)}, zona {semaforo(sep_prof)}). La pregunta era si
los sismos con marca son más superficiales. Es cierto que {pct(sup[CON], 1)} de ellos tiene menos de 70 km, pero también
{pct(sup[SIN], 1)} de los sin marca: las dos barras de abajo son casi iguales, porque los sismos profundos son raros en
general. Además, {int((con_marca['depth'] > 300).sum())} sismos de más de 300 km tienen la marca, aunque un sismo tan
profundo casi no mueve el fondo del mar.
""")
    barras_p = alt.Chart(reparto).mark_bar().encode(
        y=alt.Y("marca:N", title=None, sort=[CON, SIN]),
        x=alt.X("parte:Q", title=None, axis=alt.Axis(format="%"), stack="normalize"),
        color=alt.Color("rango:N", title="profundidad", sort=RANGOS_PROF,
                        scale=alt.Scale(domain=RANGOS_PROF, range=PROFUNDIDAD), legend=alt.Legend(orient="top")),
        order=alt.Order("orden:Q"),
        tooltip=["marca:N", "rango:N", "sismos:Q", alt.Tooltip("parte:Q", format=".1%")])
    grafico(barras_p.properties(width=700, height=110))
    st.markdown(f"""
**Sobre los números.** En la Entrega 2 las hipótesis se midieron sobre los {miles(n_total)} sismos. Acá se miden sobre
los {miles(len(modelados))} que usa el modelo, porque desde la Entrega 3 sabemos que los de abril a julio de 2026 no
tienen la marca cargada. Las separaciones cambian en el tercer decimal; lo único que se mueve es la proporción de sismos
de más de 7,5 sin marca: 21,7 % en la Entrega 2 y {pct(len(excepciones) / len(grandes), 1)} acá.

**Redundancias.** En la matriz de correlación ningún par de columnas candidatas pasa 0,6, así que no se descartó nada por
redundancia. Los pares más altos son `mag`–`sig` (otra señal de que `sig` copia a la magnitud) y
`dist_costa_km`–`dmin` (las estaciones están en tierra: cuanto más mar adentro, más lejos queda la estación).
""")

st.divider()


# =================================================================================================
# TALLO 3 · EL MODELO
# =================================================================================================
st.header("El modelo")
st.subheader(f"El modelo detecta {test['aciertos_con_marca']} de cada "
             f"{test['aciertos_con_marca'] + test['falsos_negativos']} sismos con marca, "
             f"a cambio de una falsa alarma cada "
             f"{round((test['verdaderos_negativos'] + test['falsas_alarmas']) / test['falsas_alarmas'])} sismos sin marca")

conf = (sismos[sismos["conjunto"] == "test"]
        .assign(real=lambda d: d["tsunami"].map({1: CON, 0: SIN}))
        .groupby(["real", "resultado"], as_index=False).size())
conf["parte"] = conf["size"] / conf.groupby("real")["size"].transform("sum")
barra_conf = alt.Chart(conf).mark_bar().encode(
    y=alt.Y("real:N", title="en el test, sismos…", sort=[CON, SIN]),
    x=alt.X("parte:Q", title=None, axis=alt.Axis(format="%"), stack="normalize"),
    color=alt.Color("resultado:N", title="qué dijo el modelo", scale=ESCALA_RESULTADO, legend=alt.Legend(orient="top")),
    order=alt.Order("resultado:N", sort="descending"),
    tooltip=[alt.Tooltip("resultado:N"), alt.Tooltip("size:Q", title="sismos"),
             alt.Tooltip("parte:Q", title="parte", format=".1%")])
grafico(barra_conf.properties(width=900, height=160))
st.markdown(f"Es un gradient boosting que usa la magnitud, la ubicación, la profundidad, el tipo de magnitud y si el "
            f"epicentro está en el mar. Con umbral {num(UMBRAL, 2)}, de los "
            f"{test['aciertos_con_marca'] + test['falsos_negativos']} sismos con marca del test detecta "
            f"{test['aciertos_con_marca']} ({pct(test['recall_tsunami'])}) y pierde {test['falsos_negativos']}: ese es "
            f"el error caro. A cambio da {test['falsas_alarmas']} falsas alarmas entre "
            f"{miles(test['verdaderos_negativos'] + test['falsas_alarmas'])} sismos sin marca: de cada 100 alertas, "
            f"{round(test['precision_tsunami'] * 100)} son correctas.")

st.markdown("##### En qué se apoya: la ubicación pesa tanto como la magnitud")
nombres_col = {"longitude": "longitud", "mag": "magnitud", "latitude": "latitud", "magType": "tipo de magnitud",
               "dist_costa_km": "mar o tierra", "depth": "profundidad"}
imp = pd.DataFrame(exp["importancia"])
imp["columna"] = imp["columna"].map(nombres_col)
imp["bajo"], imp["alto"] = imp["caida"] - imp["desvio"], imp["caida"] + imp["desvio"]
orden_imp = imp.sort_values("caida", ascending=False)["columna"].tolist()
barras_i = alt.Chart(imp).mark_bar(color=NEUTRO).encode(
    y=alt.Y("columna:N", sort=orden_imp, title=None),
    x=alt.X("caida:Q", title="cuánto cae el f1 macro al desordenarla"),
    tooltip=["columna:N", alt.Tooltip("caida:Q", format=".3f"), alt.Tooltip("desvio:Q", format=".3f")])
bigotes_i = alt.Chart(imp).mark_rule(color="#222").encode(y=alt.Y("columna:N", sort=orden_imp), x="bajo:Q", x2="alto:Q")

cortes = exp["arbol"]["cortes"]
lat_c, lon_c, mag_c = cortes.get("latitude", 40.2), cortes.get("longitude", -116.6), cortes.get("mag", 6.05)
mapa = modelados.assign(grupo=lambda d: d["tsunami"].map({0: "sin marca", 1: "con marca, magnitud 6 o más"}).where(
    ~((d["tsunami"] == 1) & (d["mag"] < 6)), "con marca, magnitud menor a 6"))
dominio_g = ["sin marca", "con marca, magnitud 6 o más", "con marca, magnitud menor a 6"]
puntos_g = alt.Chart(mapa[["longitude", "latitude", "grupo", "tsunami", "place", "mag"]]).mark_circle().encode(
    longitude="longitude:Q", latitude="latitude:Q",
    color=alt.Color("grupo:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=dominio_g, range=[GRIS_CLARO, AZUL_CLARO, AZUL])),
    size=alt.Size("grupo:N", legend=None, scale=alt.Scale(domain=dominio_g, range=[3, 14, 26])),
    order=alt.Order("tsunami:Q"),
    tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("mag:Q", title="magnitud")])
lados = pd.DataFrame({"lon": [-179.9, lon_c, -179.9, -179.9], "lat": [lat_c, lat_c, 75, lat_c],
                      "lon2": [lon_c, lon_c, lon_c, -179.9], "lat2": [lat_c, 75, 75, 75]})
caja = alt.Chart(lados).mark_rule(color="#e3242b", strokeWidth=2, strokeDash=[6, 3]).encode(
    longitude="lon:Q", latitude="lat:Q", longitude2="lon2:Q", latitude2="lat2:Q")
texto_caja = alt.Chart(pd.DataFrame({"lon": [lon_c + 3], "lat": [72.0], "t": ["la «caja» del árbol"]})).mark_text(
    align="left", color="#e3242b", fontSize=12, fontWeight="bold").encode(longitude="lon:Q", latitude="lat:Q", text="t:N")
izq, der = st.columns([1, 1.9])
with izq:
    grafico((barras_i + bigotes_i).properties(width=330, height=230))
with der:
    grafico(alt.layer(fondo, puntos_g, caja, texto_caja).project(type="equirectangular")
            .properties(width=620, height=310))
f1_elegido = {c["modelo"]: c for c in exp["comparacion"]}["Gradient boosting ajustado (elegido)"]["f1"]
st.markdown(f"**A la izquierda:** si se desordena una columna y el modelo empeora mucho, es que se apoyaba en ella. "
            "Longitud, magnitud y latitud pesan parecido; el tipo de magnitud, algo; la profundidad y mar o tierra, "
            "nada. La ubicación no es fuga: las coordenadas se conocen apenas ocurre el sismo y no se calculan con la "
            "marca.  \n"
            f"**A la derecha, por qué pesa el lugar:** los puntos azul oscuro son los {len(chicos)} sismos con marca y "
            "magnitud menor a 6, y casi todos están en Alaska y la costa oeste, donde la USGS tiene un sistema de alerta "
            f"propio con un umbral más bajo. El rectángulo rojo es la regla que encuentra un árbol de sólo tres preguntas "
            f"(f1 {num(exp['arbol']['f1'])}): si la magnitud es mayor a {num(mag_c, 2)}, casi siempre hay marca; si no, "
            f"la hay al norte de {num(lat_c, 1)}° y al oeste de {num(lon_c, 1)}°. El boosting llega a "
            f"{num(f1_elegido)} combinando muchos cortes de este tipo.")

recientes = sismos[sismos["conjunto"] == "sin marca cargada"]
grandes_recientes = recientes[recientes["mag"] >= 7.2]
with st.container(border=True):
    st.markdown("#### Las conclusiones")
    st.markdown(f"""
- **La respuesta a la pregunta:** lo que permite estimar si un sismo lleva la marca de tsunami de la USGS es **su
  magnitud y su ubicación**, casi por partes iguales. El tipo de magnitud aporta algo; la profundidad, la distancia a la
  costa y las métricas de la red de estaciones, nada.
- **Lo que el modelo explicó y el análisis no:** por qué hay sismos grandes sin marca y chicos con marca. La marca sigue
  la regla regional de alerta de la USGS, además de la física del sismo.
- **Lo que predice es la marca, no la ola.** El modelo aprendió el criterio de la USGS, incluido su sesgo regional: donde
  la USGS casi no marca sismos (por ejemplo, el Caribe), el modelo tampoco.
- **Para comprobarlo con datos nuevos:** de los {miles(len(recientes))} sismos de abril a julio de 2026, que la USGS
  todavía no marcó, el modelo da alerta en {int(recientes['alerta'].sum())}, entre ellos los {len(grandes_recientes)} de
  magnitud 7,2 o más. Cuando la USGS cargue esas marcas se va a poder ver cuántos acertó.
""")

with st.expander("Cómo lo sabemos"):
    g = resumen["grupos"]
    comparacion = pd.DataFrame(exp["comparacion"])
    piso = comparacion.loc[comparacion["modelo"] == "Piso: siempre «sin marca»", "f1"].iloc[0]
    ablacion = {a["variante"]: a for a in exp["ablacion"]}
    base_f1, base_desvio = ablacion["base"]["f1"], ablacion["base"]["desvio"]
    st.markdown(f"""
**La métrica: f1 macro**, declarada antes de comparar modelos. Promedia el f1 de las dos clases, así que la marca (4 %)
pesa lo mismo que el resto. El error caro es el falso negativo (un sismo con potencial de tsunami que no se marca); una
falsa alarma cuesta menos.

**El piso:** {num(piso)}, un `DummyClassifier` que dice siempre «sin marca».

**La validación.** Un sismo grande trae réplicas en el mismo lugar durante días: si una réplica queda en entrenamiento y
otra en validación, el modelo «reconoce» la secuencia y el número sale inflado. Por eso los pedazos se arman con
`StratifiedGroupKFold` (5 pedazos), con grupos de **celda de 2° × 2° de latitud y longitud y mes**: {miles(g['cantidad'])}
grupos; el más grande son los {g['mayor_sismos']} sismos de la secuencia de Kermadec de marzo de 2021. El test (20 %) se
separó con el mismo criterio y se miró una sola vez. ¿Importa cómo se arman los pedazos?
""")
    part = pd.DataFrame(exp["particiones"])
    part["valor"] = [f"{num(f)} ± {num(d)}" for f, d in zip(part["f1"], part["desvio"])]
    st.dataframe(part.pivot(index="modelo", columns="forma", values="valor")[["Al azar", "Por grupo", "Por fecha"]])
    st.markdown("Dan lo mismo, siempre dentro del desvío: **las réplicas no inflan el número**, porque son casi todas "
                "chicas y sin marca. Igual nos quedamos con la validación por grupo, porque imita el uso real.")

    st.markdown("**La comparación de modelos** (f1 macro en validación cruzada, media ± desvío):")
    modelos = comparacion[~comparacion["modelo"].str.startswith("Piso")].copy()
    modelos["bajo"], modelos["alto"] = modelos["f1"] - modelos["desvio"], modelos["f1"] + modelos["desvio"]
    modelos["elegido"] = np.where(modelos["modelo"].str.contains("elegido"), "elegido", "probado")
    modelos["texto"] = [f"{num(f)} ± {num(d)}" for f, d in zip(modelos["f1"], modelos["desvio"])]
    orden_modelos = modelos.sort_values("f1")["modelo"].tolist()
    color_elegido = alt.Color("elegido:N", scale=alt.Scale(domain=["elegido", "probado"], range=[AZUL, NEUTRO]),
                              legend=None)
    eje_m = alt.Y("modelo:N", sort=orden_modelos, title=None)
    bigotes = alt.Chart(modelos).mark_rule(strokeWidth=2).encode(
        x=alt.X("bajo:Q", title="f1 macro", scale=alt.Scale(domain=[0.45, 0.95])), x2="alto:Q", color=color_elegido,
        y=eje_m)
    puntos_m = alt.Chart(modelos).mark_circle(size=110).encode(x="f1:Q", color=color_elegido, y=eje_m,
                                                               tooltip=["modelo:N", "texto:N"])
    textos_m = alt.Chart(modelos).mark_text(align="left", dx=10, color="#222").encode(x="alto:Q", text="texto:N", y=eje_m)
    linea_piso = alt.Chart(pd.DataFrame({"x": [piso], "t": [f"piso: {num(piso)}"]}))
    grafico((bigotes + puntos_m + textos_m + linea_piso.mark_rule(color=GRIS, strokeDash=[4, 4]).encode(x="x:Q")
             + linea_piso.mark_text(align="left", dx=4, y=8, color=GRIS).encode(x="x:Q", text="t:N"))
            .properties(width=760, height=210))
    rf = comparacion.set_index("modelo").loc["Random forest"]
    gb = comparacion.set_index("modelo").loc["Gradient boosting"]
    logi = comparacion.set_index("modelo").loc["Regresión logística"]
    st.markdown(f"""
Random forest y gradient boosting **empatan**: la diferencia ({num(rf['f1'] - gb['f1'])}) es menor que el desvío
({num(gb['desvio'])}). Se eligió el boosting porque entrena unas tres veces más rápido y da un modelo más liviano para la
app. La logística pierde porque sólo puede sumar columnas con pesos («más al norte, más marca») y no puede dibujar la caja
de Alaska: sin latitud y longitud baja de {num(logi['f1'])} a {num(exp['logistica_sin_ubicacion']['f1'])}, y el boosting
de {num(base_f1)} a {num(ablacion['− latitud y longitud']['f1'])}.

**El ajuste:** `GridSearchCV` probó 36 combinaciones (tasa de aprendizaje, cantidad de árboles, hojas por árbol y
balanceo de clases). Ganó tasa {num(resumen['hiperparametros']['learning_rate'], 2)},
{resumen['hiperparametros']['max_iter']} árboles, {resumen['hiperparametros']['max_leaf_nodes']} hojas y sin balancear
({num(f1_elegido)}). Balancear las clases empató con el mejor y fue menos estable.

**El umbral.** El modelo da una probabilidad, y el umbral decide a partir de cuánto eso es «alerta». En la Entrega 3
usamos 0,5; la corrección fue bajarlo. Como el f1 es casi plano entre 0,2 y 0,5, se eligió el umbral más bajo con el f1
máximo, {num(UMBRAL, 2)}, para no perder sismos marcados. Se eligió con validación cruzada, sin mirar el test:
""")
    curva = pd.DataFrame(exp["curva_umbral"]).melt(id_vars="umbral", var_name="medida", value_name="valor")
    medidas = ["recall", "precisión", "f1 macro"]
    lineas_u = alt.Chart(curva).mark_line(strokeWidth=2.5).encode(
        x=alt.X("umbral:Q", title="umbral"),
        y=alt.Y("valor:Q", title="validación cruzada", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("medida:N", title=None, sort=medidas,
                        scale=alt.Scale(domain=medidas, range=["#222222", "#9a9a9a", "#2e7d32"]),
                        legend=alt.Legend(orient="top")),
        tooltip=["umbral:Q", "medida:N", alt.Tooltip("valor:Q", format=".3f")])
    marcas_u = alt.Chart(pd.DataFrame({"umbral": [0.5, UMBRAL], "t": ["antes: 0,5", f"elegido: {num(UMBRAL, 2)}"],
                                       "c": [GRIS, AZUL]}))
    grafico((lineas_u + marcas_u.mark_rule(strokeDash=[4, 4]).encode(x="umbral:Q", color=alt.Color("c:N", scale=None))
             + marcas_u.mark_text(align="left", dx=4, y=10).encode(x="umbral:Q", text="t:N",
                                                                    color=alt.Color("c:N", scale=None)))
            .properties(width=760, height=240))
    st.markdown(f"""
En validación cruzada el recall sube de {pct(vc05['recall_tsunami'], 1)} a {pct(vc['recall_tsunami'], 1)} y la precisión
baja de {pct(vc05['precision_tsunami'], 1)} a {pct(vc['precision_tsunami'], 1)}.

**El test, una sola vez** (umbral {num(UMBRAL, 2)}): f1 macro {num(test['f1_macro'])} (validación cruzada:
{num(vc['f1_macro'])}), recall {pct(test['recall_tsunami'])}, precisión {pct(test['precision_tsunami'])}, PR-AUC
{num(test['pr_auc'], 2)} contra {num(resumen['piso_test']['pr_auc'], 2)} del piso.

**Lo que probamos y no anduvo** (la ablación: el modelo base contra el mismo modelo agregando o sacando una columna por
vez; la franja gris es ± un desvío del base, y lo que cae adentro es un empate):
""")
    abl = pd.DataFrame([a for a in exp["ablacion"] if a["variante"] != "base"])
    abl["diferencia"] = abl["f1"] - base_f1
    abl["lectura"] = np.where(abl["diferencia"].abs() > base_desvio, "supera el desvío", "empate")
    franja = alt.Chart(pd.DataFrame({"a": [-base_desvio], "b": [base_desvio]})).mark_rect(
        color="#e8e6e1").encode(x="a:Q", x2="b:Q")
    barras_a = alt.Chart(abl).mark_bar().encode(
        y=alt.Y("variante:N", sort=abl.sort_values("diferencia")["variante"].tolist(), title=None),
        x=alt.X("diferencia:Q", title="diferencia de f1 macro con el modelo base"),
        color=alt.Color("lectura:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["supera el desvío", "empate"], range=[NEUTRO, "#b9b6b0"])),
        tooltip=["variante:N", alt.Tooltip("f1:Q", format=".3f"), alt.Tooltip("diferencia:Q", format="+.3f")])
    grafico((franja + barras_a).properties(width=700, height=300))
    st.markdown("Sólo sacar la latitud y la longitud supera el desvío. Las métricas de la red (`nst`, `dmin`, `gap`, "
                "`rms`) no mejoran nada, ni solas ni juntas; la distancia continua empata con mar o tierra; la "
                "profundidad empata; `sig` tampoco cambia el número.")

    casos = sismos[sismos["conjunto"] == "test"]
    columnas_e = {"time": "fecha (UTC)", "place": "lugar", "mag": "magnitud", "depth": "profundidad (km)",
                  "en_el_mar": "epicentro", "probabilidad": "probabilidad"}
    fn_ = casos[casos["resultado"] == "falso negativo"].sort_values("probabilidad")
    st.markdown(f"**Dónde se equivoca.** Los {len(fn_)} falsos negativos del test:")
    st.dataframe(fn_[list(columnas_e)].rename(columns=columnas_e), hide_index=True)
    st.markdown(f"""
Son sismos de 6,3 a 6,5 fuera de Alaska (en tierra o profundos), sismos chicos en el mar frente a California y Alaska que
la USGS marcó con su umbral regional, y uno en el Caribe, una región casi sin marcas. Las falsas alarmas más seguras son
sismos fuertes en el mar que, por magnitud y lugar, son el caso típico que la USGS marca, pero esta vez no lo hizo.

**Limitaciones.** Los grupos de validación son una aproximación: una secuencia de réplicas que cruza el borde de una
celda o el cambio de mes queda partida (igual, partir al azar dio lo mismo). Los sismos desde abril de 2026 quedaron
afuera hasta que la USGS cargue su marca. Y la importancia por permutación usa la predicción por defecto del modelo, que
corta en 0,5 y no en {num(UMBRAL, 2)}; el orden de las columnas no depende del umbral.
""")

st.divider()


# =================================================================================================
# COPA · EXPLORÁ
# =================================================================================================
st.header("Tu turno · Explorá")
recientes = sismos[sismos["conjunto"] == "sin marca cargada"]
grandes_recientes = recientes[recientes["mag"] >= 7.2]
st.markdown("Todos los sismos, con lo que dijo el modelo de cada uno. Cada sismo lo predice un modelo que no lo vio: "
            "en el entrenamiento, la predicción sale de la validación cruzada. **Arrastrá el mouse sobre el histograma** "
            "para elegir un rango de magnitud, hacé clic en la leyenda para resaltar un resultado y pasá el mouse por "
            "un sismo para ver el detalle.")
st.markdown(f"En gris oscuro están los {miles(len(recientes))} sismos que la USGS todavía no marcó. El modelo da alerta en "
            f"{int(recientes['alerta'].sum())}, entre ellos {'los' if len(grandes_recientes) > 1 else 'el'} "
            f"{len(grandes_recientes)} de magnitud 7,2 o más "
            f"({', '.join(f'{r.region} {num(r.mag, 1)}' for r in grandes_recientes.itertuples())}). Cuando la USGS "
            "cargue esas marcas, van a servir para comprobar al modelo con datos que nunca vio.")

f_1, f_2 = st.columns(2)
que_mostrar = f_1.multiselect("Qué dijo el modelo", RESULTADOS, default=RESULTADOS)
region = f_2.selectbox("Región", ["todas"] + sorted(sismos["region"].dropna().unique()))
vista = sismos[sismos["resultado"].isin(que_mostrar)]
if region != "todas":
    vista = vista[vista["region"] == region]
vista = vista.assign(marca_real=vista["tsunami"].map({1: "sí", 0: "no"}).where(vista["conjunto"] != "sin marca cargada",
                                                                               "todavía no cargada"))
vista = vista[["longitude", "latitude", "mag", "resultado", "probabilidad", "place", "time", "magType", "depth",
               "en_el_mar", "marca_real", "conjunto"]]

rango = alt.selection_interval(encodings=["x"])
resaltar = alt.selection_point(fields=["resultado"], bind="legend")
color_res = alt.Color("resultado:N", title="qué dijo el modelo", scale=ESCALA_RESULTADO,
                      legend=alt.Legend(orient="top"))
puntos_copa = alt.Chart(vista).mark_circle().encode(
    longitude="longitude:Q", latitude="latitude:Q", color=color_res,
    size=alt.Size("mag:Q", title="magnitud", scale=alt.Scale(domain=[5, 9], range=[6, 260]), legend=None),
    opacity=alt.condition(resaltar, alt.value(0.85), alt.value(0.05)),
    order=alt.Order("probabilidad:Q"),
    tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("time:N", title="fecha (UTC)"),
             alt.Tooltip("mag:Q", title="magnitud"), alt.Tooltip("magType:N", title="tipo de magnitud"),
             alt.Tooltip("depth:Q", title="profundidad (km)"), alt.Tooltip("en_el_mar:N", title="mar o tierra"),
             alt.Tooltip("marca_real:N", title="marca de la USGS"),
             alt.Tooltip("probabilidad:Q", title="probabilidad según el modelo", format=".0%"),
             alt.Tooltip("resultado:N", title="resultado"), alt.Tooltip("conjunto:N", title="conjunto")],
).add_params(resaltar).transform_filter(rango)
mapa_copa = alt.layer(fondo, puntos_copa).project(type="equirectangular").properties(width=640, height=330)
histograma = alt.Chart(vista).mark_bar().encode(
    x=alt.X("mag:Q", bin=alt.Bin(step=0.1), title="magnitud (arrastrá para elegir un rango)"),
    y=alt.Y("count():Q", title="sismos", scale=alt.Scale(type="symlog")),
    color=color_res, tooltip=[alt.Tooltip("count():Q", title="sismos"), "resultado:N"],
).add_params(rango).properties(width=300, height=330)
grafico(alt.hconcat(mapa_copa, histograma))
st.caption(f"{miles(len(vista))} sismos con los filtros de arriba (el rango del histograma filtra sólo el mapa). "
           f"Alerta = probabilidad de {num(UMBRAL, 2)} o más. El eje del histograma está en escala logarítmica, para "
           "que se vean los sismos grandes, que son pocos.")

st.divider()


# =================================================================================================
# COPA · PROBÁ EL MODELO
# =================================================================================================
st.header("Tu turno · Probá el modelo")
st.markdown("Armá un sismo y mirá qué probabilidad le da el modelo de que la USGS lo marque. Los controles van del "
            "mínimo al máximo que vio el entrenamiento, así que no se puede cargar un valor que el modelo no conoce.")

r = resumen["rangos"]
tipos = resumen["magtypes"]


def ejemplo(filtro, por, descripcion):
    """El sismo real con el mayor valor de `por` entre los que cumplen el filtro."""
    candidatos = modelados[filtro(modelados)]
    return (descripcion, candidatos.loc[candidatos[por].idxmax()]) if len(candidatos) else None


ejemplos = [e for e in [
    ejemplo(lambda d: d["mag"] > 0, "mag", "El más grande del catálogo"),
    ejemplo(lambda d: (d["region"] == "Alaska") & (d["tsunami"] == 1) & (d["mag"] < 5.5), "probabilidad",
            "Un sismo chico de Alaska, con marca"),
    ejemplo(lambda d: (d["en_el_mar"] == "tierra") & (d["mag"] >= 6.8), "mag", "Un sismo fuerte en tierra"),
    ejemplo(lambda d: d["depth"] > 300, "mag", "Un sismo grande y profundo"),
    ejemplo(lambda d: (d["magType"] == "mb") & (d["tsunami"] == 0), "mag", "El mayor sismo informado en mb"),
] if e]
opciones = ["— armarlo desde cero —"] + [f"{d}: {s['place']} ({num(s['mag'], 1)}, {s['time'][:10]})" for d, s in ejemplos]
elegido = st.selectbox("Partir de un sismo real", opciones)
real = ejemplos[opciones.index(elegido) - 1][1] if elegido != opciones[0] else None


def dentro(valor, rango_):
    return float(min(max(valor, rango_[0]), rango_[1]))


inicial = {"mag": 7.0, "depth": 25.0, "latitude": -33.5, "longitude": -72.0, "magType": "mww", "mar": True}
if real is not None:
    inicial = {"mag": round(float(real["mag"]), 1), "depth": round(float(real["depth"])),
               "latitude": round(float(real["latitude"]), 1), "longitude": round(float(real["longitude"]), 1),
               "magType": real["magType"], "mar": real["en_el_mar"] == "mar"}
    st.caption(f"En la realidad, la USGS {'**lo marcó**' if real['tsunami'] == 1 else '**no lo marcó**'}. Los controles "
               "arrancan con sus valores (redondeados): movelos para ver qué cambia.")

p1, p2 = st.columns(2)
mag = p1.slider("Magnitud", r["mag"][0], r["mag"][1], dentro(inicial["mag"], r["mag"]), step=0.1, format="%.1f")
depth = p1.slider("Profundidad (km)", r["depth"][0], r["depth"][1], dentro(inicial["depth"], r["depth"]), step=1.0)
lat = p2.slider("Latitud", r["latitude"][0], r["latitude"][1], dentro(inicial["latitude"], r["latitude"]), step=0.1,
                format="%.1f")
lon = p2.slider("Longitud", r["longitude"][0], r["longitude"][1], dentro(inicial["longitude"], r["longitude"]),
                step=0.1, format="%.1f")
en_el_mar = p2.radio("¿El epicentro está en el mar o en tierra?", ["mar", "tierra"], horizontal=True,
                     index=0 if inicial["mar"] else 1) == "mar"

with st.expander("Opciones avanzadas: tipo de magnitud"):
    tipo_inicial = inicial["magType"] if inicial["magType"] in tipos else "mww"
    mag_type = st.selectbox("Tipo de magnitud", tipos, index=tipos.index(tipo_inicial) if tipo_inicial in tipos else 0,
                            format_func=lambda m: f"{NOMBRES_MAGTYPE.get(m, m)} ({m})",
                            help="Sólo se ofrecen los tipos con al menos 20 sismos en el entrenamiento.")
    cuota_mww = (modelados["magType"] == "mww").mean()
    st.caption(f"Es el método con que la USGS calculó la magnitud del sismo. Si no sabés cuál elegir, dejá *mww*: es "
               f"la de {round(cuota_mww * 10)} de cada 10 sismos y la que la USGS usa para los sismos grandes. "
               "Qué significa cada una está en el cuadro de abajo.")

en_mb = modelados[modelados["magType"] == "mb"]
with st.expander("¿Qué significa cada tipo de magnitud?"):
    conteo_tipos = modelados.groupby("magType")["tsunami"].agg(["size", "sum"])
    filas = []
    for m in tipos:
        como, cuando = EXPLICACION_MAGTYPE.get(m, ("—", "—"))
        n, k = (int(conteo_tipos.loc[m, "size"]), int(conteo_tipos.loc[m, "sum"])) if m in conteo_tipos.index else (0, 0)
        en_datos = f"{miles(n)} sismos · {pct(k / n, 1)} con marca" if n else "—"
        filas.append(f"| **{m}** | {NOMBRES_MAGTYPE.get(m, m)} | {como} | {cuando} | {en_datos} |")
    st.markdown(f"""
La **magnitud** mide el tamaño de un sismo, pero hay varias formas de calcularla. La sigla indica qué método usó la
USGS (o la red que informó el sismo). Hay dos familias:

- **Magnitudes de momento** (las que empiezan con *mw*): se calculan a partir del momento sísmico, es decir, de qué tan
  grande es la falla que se rompió y cuánto se deslizó. Es la medida física más completa y **no se satura**: sirve
  igual para un sismo de 5 que para uno de 9.
- **Magnitudes de amplitud** (*mb* y *ml*): miden qué tan fuerte se mueve un tipo de onda en el sismógrafo. Son más
  rápidas de calcular, pero **se saturan**: en los sismos grandes dejan de crecer aunque el sismo sea mayor.

| Sigla | Nombre | Cómo se calcula | Cuándo la usa la USGS | En los datos (hasta marzo de 2026) |
|---|---|---|---|---|
""" + "\n".join(filas) + f"""

**Por qué cambia la predicción.** Las magnitudes de momento dan resultados casi iguales entre sí. La que cambia todo es
*mb*: casi ningún sismo informado en mb tiene marca ({int(en_mb['tsunami'].sum())} de {miles(len(en_mb))}), porque la
USGS la usa en sismos moderados. Por eso el modelo aprendió a asociar *mb* con «sin marca». Las siglas que no aparecen
en el selector (como *mwp* o *mwc*) tienen menos de 20 sismos en el entrenamiento y no se ofrecen.
""")

if mag_type == "mb" and mag >= 6.5:
    todos_mb = sismos[sismos["magType"] == "mb"]
    st.warning(f"Combinación casi inexistente: la escala *mb* se satura cerca de magnitud 6, así que la USGS casi no "
               f"informa en mb sismos tan grandes. En los datos hay {(todos_mb['mag'] >= 6.5).sum()} de "
               f"{miles(len(todos_mb))} (el mayor, de {num(todos_mb['mag'].max(), 1)}). El modelo casi no vio casos "
               "así: tomá esta probabilidad con cuidado.")

probabilidad = guardado["pipeline"].predict_proba(modelo.caso(mag, depth, lat, lon, mag_type, en_el_mar))[0, 1]
m1, m2 = st.columns(2)
m1.metric("probabilidad de que la USGS lo marque", pct(probabilidad))
m2.metric(f"alerta (umbral {num(UMBRAL, 2)})", "sí" if probabilidad >= UMBRAL else "no")

rango_elegido = RANGOS_MAG[int(np.digitize(mag, CORTES_MAG[1:-1], right=True))]
mismo_rango = modelados[modelados["rango_mag"].astype(str) == rango_elegido]
misma_zona = mismo_rango[mismo_rango["en_el_mar"] == ("mar" if en_el_mar else "tierra")]
st.caption(f"Para comparar: en los datos, los sismos de magnitud {rango_elegido} tienen la marca el "
           f"{pct(mismo_rango['tsunami'].mean())} de las veces ({pct(misma_zona['tsunami'].mean())} los que ocurren en "
           f"{'el mar' if en_el_mar else 'tierra'}). La diferencia con la probabilidad del modelo la pone, sobre todo, "
           "el lugar.")

tu_sismo = alt.Chart(pd.DataFrame({"latitude": [lat], "longitude": [lon]})).mark_point(
    shape="cross", size=400, strokeWidth=3, color="#e3242b").encode(longitude="longitude:Q", latitude="latitude:Q")
grafico(alt.layer(fondo, tu_sismo).project(type="equirectangular").properties(width=900, height=360))

st.divider()
st.caption(f"Datos: USGS (earthquake.usgs.gov), sismos del {resumen['datos']['desde']} al {resumen['datos']['hasta']}. "
           f"Modelo: gradient boosting entrenado con datos hasta el 31/03/2026, semilla {resumen['semilla']}.")
