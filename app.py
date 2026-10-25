import io
import hashlib
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Operations Intelligence Center",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = [
    "Fecha",
    "Planta",
    "Linea",
    "Maquina",
    "Turno",
    "Horas_programadas",
    "Tiempo_operativo_h",
    "Tiempo_parada_h",
    "Produccion_total",
    "Produccion_buena",
    "Produccion_defectuosa",
    "Capacidad_esperada",
    "Tiempo_ciclo_real_s",
    "Tiempo_ciclo_ideal_s",
    "Causa_parada",
]

NUMERIC_COLUMNS = [
    "Horas_programadas",
    "Tiempo_operativo_h",
    "Tiempo_parada_h",
    "Produccion_total",
    "Produccion_buena",
    "Produccion_defectuosa",
    "Capacidad_esperada",
    "Tiempo_ciclo_real_s",
    "Tiempo_ciclo_ideal_s",
]

GREEN = "#178A55"
AMBER = "#C98A00"
RED = "#C73B3B"
BLUE = "#2F6B9A"
DARK = "#18212B"
MID = "#5E6B78"
LIGHT = "#F4F6F8"
GRID = "#E5E9ED"


# ============================================================
# ESTILO
# ============================================================
st.markdown(
    f"""
    <style>
        .stApp {{
            background: {LIGHT};
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1600px;
        }}

        [data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #E4E8EC;
        }}

        [data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid #E2E7EB;
            border-radius: 10px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 1px 2px rgba(20, 30, 40, 0.04);
        }}

        [data-testid="stMetricLabel"] {{
            color: {MID};
            font-weight: 600;
        }}

        [data-testid="stMetricValue"] {{
            color: {DARK};
        }}

        h1, h2, h3 {{
            color: {DARK};
        }}

        .app-header {{
            background: #FFFFFF;
            border: 1px solid #E2E7EB;
            border-left: 5px solid {BLUE};
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 18px;
        }}

        .app-title {{
            font-size: 1.72rem;
            font-weight: 750;
            color: {DARK};
            margin: 0;
        }}

        .app-subtitle {{
            color: {MID};
            margin: 4px 0 0 0;
            font-size: 0.95rem;
        }}

        .section-note {{
            background: #FFFFFF;
            border: 1px solid #E2E7EB;
            border-radius: 9px;
            padding: 12px 15px;
            color: {MID};
            margin-bottom: 12px;
        }}

        .alert-critical {{
            background: #FFF3F3;
            border: 1px solid #F2CACA;
            border-left: 5px solid {RED};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 6px 0;
            color: #6D2424;
        }}

        .alert-warning {{
            background: #FFF8E8;
            border: 1px solid #F0DCA4;
            border-left: 5px solid {AMBER};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 6px 0;
            color: #6A4C00;
        }}

        .alert-ok {{
            background: #EDF8F2;
            border: 1px solid #C8E6D5;
            border-left: 5px solid {GREEN};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 6px 0;
            color: #155B3A;
        }}

        .small-label {{
            color: {MID};
            font-size: 0.82rem;
        }}

        .status-chip {{
            display: inline-block;
            padding: 4px 9px;
            border-radius: 16px;
            font-size: 0.78rem;
            font-weight: 650;
            background: #EEF2F5;
            color: {DARK};
            margin-right: 6px;
        }}

        .footer {{
            color: #7A8793;
            font-size: 0.78rem;
            margin-top: 28px;
            padding-top: 12px;
            border-top: 1px solid #E1E6EA;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNCIONES DE DATOS
# ============================================================
def strip_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(c) != "Mn"
    )


def normalize_headers(df):
    aliases = {
        "fecha": "Fecha",
        "planta": "Planta",
        "linea": "Linea",
        "maquina": "Maquina",
        "turno": "Turno",
        "horas_programadas": "Horas_programadas",
        "tiempo_operativo": "Tiempo_operativo_h",
        "tiempo_operativo_h": "Tiempo_operativo_h",
        "tiempo_parada": "Tiempo_parada_h",
        "tiempo_parada_h": "Tiempo_parada_h",
        "produccion_total": "Produccion_total",
        "produccion_buena": "Produccion_buena",
        "produccion_defectuosa": "Produccion_defectuosa",
        "capacidad_esperada": "Capacidad_esperada",
        "produccion_esperada": "Capacidad_esperada",
        "tiempo_ciclo_real": "Tiempo_ciclo_real_s",
        "tiempo_ciclo_real_s": "Tiempo_ciclo_real_s",
        "tiempo_ciclo_ideal": "Tiempo_ciclo_ideal_s",
        "tiempo_ciclo_ideal_s": "Tiempo_ciclo_ideal_s",
        "causa_parada": "Causa_parada",
    }

    rename_map = {}
    for col in df.columns:
        key = strip_accents(col).strip().lower().replace(" ", "_")
        key = key.replace("/", "_").replace("-", "_")
        while "__" in key:
            key = key.replace("__", "_")
        if key in aliases:
            rename_map[col] = aliases[key]

    return df.rename(columns=rename_map)


def clean_base_data(df):
    df = normalize_headers(df.copy())

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["Planta", "Linea", "Maquina", "Turno", "Causa_parada"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=["Fecha", "Planta", "Linea", "Maquina", "Turno"])
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(0)

    # Reglas mínimas de consistencia
    df["Horas_programadas"] = df["Horas_programadas"].clip(lower=0.01)
    df["Tiempo_operativo_h"] = df["Tiempo_operativo_h"].clip(
        lower=0, upper=df["Horas_programadas"]
    )
    df["Tiempo_parada_h"] = (
        df["Horas_programadas"] - df["Tiempo_operativo_h"]
    ).clip(lower=0)

    df["Produccion_total"] = df["Produccion_total"].clip(lower=0)
    df["Produccion_buena"] = df["Produccion_buena"].clip(
        lower=0, upper=df["Produccion_total"]
    )
    df["Produccion_defectuosa"] = (
        df["Produccion_total"] - df["Produccion_buena"]
    ).clip(lower=0)

    df["Capacidad_esperada"] = df["Capacidad_esperada"].clip(lower=0.01)
    df["Tiempo_ciclo_real_s"] = df["Tiempo_ciclo_real_s"].clip(lower=0.01)
    df["Tiempo_ciclo_ideal_s"] = df["Tiempo_ciclo_ideal_s"].clip(lower=0.01)

    return df.sort_values("Fecha").reset_index(drop=True)


def add_metrics(df):
    out = df.copy()

    out["Disponibilidad"] = np.divide(
        out["Tiempo_operativo_h"],
        out["Horas_programadas"],
        out=np.zeros(len(out), dtype=float),
        where=out["Horas_programadas"].to_numpy() != 0,
    )

    out["Rendimiento"] = np.divide(
        out["Produccion_total"],
        out["Capacidad_esperada"],
        out=np.zeros(len(out), dtype=float),
        where=out["Capacidad_esperada"].to_numpy() != 0,
    )

    out["Calidad"] = np.divide(
        out["Produccion_buena"],
        out["Produccion_total"],
        out=np.zeros(len(out), dtype=float),
        where=out["Produccion_total"].to_numpy() != 0,
    )

    # Evitar porcentajes físicamente imposibles por errores de captura.
    out["Disponibilidad"] = out["Disponibilidad"].clip(0, 1)
    out["Rendimiento"] = out["Rendimiento"].clip(0, 1)
    out["Calidad"] = out["Calidad"].clip(0, 1)

    out["OEE"] = (
        out["Disponibilidad"] *
        out["Rendimiento"] *
        out["Calidad"]
    )

    out["Productividad"] = np.divide(
        out["Produccion_buena"],
        out["Tiempo_operativo_h"],
        out=np.zeros(len(out), dtype=float),
        where=out["Tiempo_operativo_h"].to_numpy() != 0,
    )

    out["Perdida_velocidad_unid"] = (
        out["Capacidad_esperada"] - out["Produccion_total"]
    ).clip(lower=0)

    out["Perdida_calidad_unid"] = out["Produccion_defectuosa"]

    out["Desviacion_ciclo_pct"] = (
        (out["Tiempo_ciclo_real_s"] - out["Tiempo_ciclo_ideal_s"])
        / out["Tiempo_ciclo_ideal_s"]
        * 100
    )

    return out


@st.cache_data
def generate_sample_data(days=45):
    np.random.seed(23)

    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end_date, periods=days, freq="D")

    plants = ["Planta Lima", "Planta Norte", "Planta Sur"]
    lines = ["Linea A", "Linea B", "Linea C"]
    machines = ["M01", "M02", "M03", "M04", "M05"]
    shifts = ["Mañana", "Tarde", "Noche"]

    profiles = {
        "M01": {"a": 0.955, "p": 0.965, "q": 0.992, "cycle": 34},
        "M02": {"a": 0.900, "p": 0.915, "q": 0.975, "cycle": 40},
        "M03": {"a": 0.790, "p": 0.885, "q": 0.965, "cycle": 43},
        "M04": {"a": 0.945, "p": 0.955, "q": 0.987, "cycle": 36},
        "M05": {"a": 0.905, "p": 0.905, "q": 0.915, "cycle": 39},
    }

    machine_line = {
        "M01": "Linea A",
        "M02": "Linea A",
        "M03": "Linea B",
        "M04": "Linea B",
        "M05": "Linea C",
    }

    rows = []

    for date in dates:
        for plant in plants:
            plant_factor = {
                "Planta Lima": 1.00,
                "Planta Norte": 0.98,
                "Planta Sur": 0.965,
            }[plant]

            for machine in machines:
                profile = profiles[machine]
                line = machine_line[machine]

                for shift in shifts:
                    shift_factor = {
                        "Mañana": 1.00,
                        "Tarde": 0.985,
                        "Noche": 0.955,
                    }[shift]

                    scheduled = 8.0

                    availability = np.clip(
                        np.random.normal(profile["a"] * shift_factor, 0.018),
                        0.55,
                        0.995,
                    )

                    performance = np.clip(
                        np.random.normal(profile["p"] * plant_factor, 0.022),
                        0.60,
                        0.995,
                    )

                    quality = np.clip(
                        np.random.normal(profile["q"], 0.010 if machine != "M05" else 0.025),
                        0.75,
                        0.999,
                    )

                    operating = scheduled * availability
                    downtime = scheduled - operating

                    ideal_cycle = profile["cycle"]
                    capacity_expected = max(1, operating * 3600 / ideal_cycle)
                    production_total = int(max(0, capacity_expected * performance))
                    good = int(max(0, production_total * quality))
                    defective = production_total - good
                    real_cycle = ideal_cycle / max(performance, 0.01)

                    if downtime < 0.2:
                        cause = "Sin parada relevante"
                    elif machine == "M03":
                        cause = np.random.choice(
                            ["Falla mecánica", "Ajuste de equipo", "Falta de material", "Cambio de formato"],
                            p=[0.58, 0.18, 0.12, 0.12],
                        )
                    elif machine == "M05":
                        cause = np.random.choice(
                            ["Ajuste de calidad", "Cambio de formato", "Falta de material", "Falla mecánica"],
                            p=[0.45, 0.25, 0.18, 0.12],
                        )
                    else:
                        cause = np.random.choice(
                            ["Cambio de formato", "Falta de material", "Ajuste de equipo", "Falla mecánica"],
                            p=[0.30, 0.25, 0.25, 0.20],
                        )

                    rows.append(
                        {
                            "Fecha": date,
                            "Planta": plant,
                            "Linea": line,
                            "Maquina": machine,
                            "Turno": shift,
                            "Horas_programadas": round(scheduled, 2),
                            "Tiempo_operativo_h": round(operating, 3),
                            "Tiempo_parada_h": round(downtime, 3),
                            "Produccion_total": production_total,
                            "Produccion_buena": good,
                            "Produccion_defectuosa": defective,
                            "Capacidad_esperada": round(capacity_expected, 2),
                            "Tiempo_ciclo_real_s": round(real_cycle, 2),
                            "Tiempo_ciclo_ideal_s": round(ideal_cycle, 2),
                            "Causa_parada": cause,
                        }
                    )

    return clean_base_data(pd.DataFrame(rows))


def template_dataframe():
    today = pd.Timestamp.today().normalize()
    return pd.DataFrame(
        [
            {
                "Fecha": today,
                "Planta": "Planta Lima",
                "Linea": "Linea A",
                "Maquina": "M01",
                "Turno": "Mañana",
                "Horas_programadas": 8,
                "Tiempo_operativo_h": 7.4,
                "Tiempo_parada_h": 0.6,
                "Produccion_total": 720,
                "Produccion_buena": 705,
                "Produccion_defectuosa": 15,
                "Capacidad_esperada": 760,
                "Tiempo_ciclo_real_s": 37,
                "Tiempo_ciclo_ideal_s": 35,
                "Causa_parada": "Cambio de formato",
            }
        ]
    )


def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
    output.seek(0)
    return output.getvalue()


def weighted_ratio(numerator, denominator):
    den = float(denominator.sum())
    return float(numerator.sum() / den) if den else 0.0


def kpi_summary(df):
    availability = weighted_ratio(df["Tiempo_operativo_h"], df["Horas_programadas"])
    performance = weighted_ratio(df["Produccion_total"], df["Capacidad_esperada"])
    quality = weighted_ratio(df["Produccion_buena"], df["Produccion_total"])

    availability = np.clip(availability, 0, 1)
    performance = np.clip(performance, 0, 1)
    quality = np.clip(quality, 0, 1)
    oee = availability * performance * quality

    productivity = (
        df["Produccion_buena"].sum() / df["Tiempo_operativo_h"].sum()
        if df["Tiempo_operativo_h"].sum() else 0
    )

    avg_cycle = np.average(
        df["Tiempo_ciclo_real_s"],
        weights=np.maximum(df["Produccion_total"], 1),
    )

    return {
        "Disponibilidad": availability * 100,
        "Rendimiento": performance * 100,
        "Calidad": quality * 100,
        "OEE": oee * 100,
        "Productividad": productivity,
        "Tiempo_ciclo": avg_cycle,
        "Produccion_total": df["Produccion_total"].sum(),
        "Produccion_buena": df["Produccion_buena"].sum(),
        "Defectos": df["Produccion_defectuosa"].sum(),
        "Paradas_h": df["Tiempo_parada_h"].sum(),
    }


def aggregate_machine_metrics(df):
    records = []
    for machine, g in df.groupby("Maquina"):
        s = kpi_summary(g)
        s["Maquina"] = machine
        s["Tiempo_ciclo_ideal"] = np.average(
            g["Tiempo_ciclo_ideal_s"],
            weights=np.maximum(g["Produccion_total"], 1),
        )
        records.append(s)
    return pd.DataFrame(records)


def aggregate_daily_metrics(df):
    records = []
    for date, g in df.groupby("Fecha"):
        s = kpi_summary(g)
        s["Fecha"] = date
        records.append(s)
    return pd.DataFrame(records).sort_values("Fecha")


def metric_status(value, target, higher_is_better=True):
    if higher_is_better:
        if value >= target:
            return "Cumple"
        if value >= target * 0.95:
            return "Advertencia"
        return "Crítico"
    else:
        if value <= target:
            return "Cumple"
        if value <= target * 1.08:
            return "Advertencia"
        return "Crítico"


def status_color(status):
    return {
        "Cumple": GREEN,
        "Advertencia": AMBER,
        "Crítico": RED,
    }.get(status, MID)


def show_header(section, subtitle):
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">{section}</div>
            <div class="app-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_alert(message, level="warning"):
    css = {
        "critical": "alert-critical",
        "warning": "alert-warning",
        "ok": "alert-ok",
    }[level]
    st.markdown(f'<div class="{css}">{message}</div>', unsafe_allow_html=True)


def format_plot(fig, height=390, legend=True):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=25, r=25, t=45, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=DARK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ) if legend else None,
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def recommendation_engine(df, goals):
    s = kpi_summary(df)
    machine = aggregate_machine_metrics(df)

    recs = []

    if s["OEE"] < goals["OEE"]:
        recs.append(
            {
                "Prioridad": "Alta",
                "Hallazgo": f"OEE global de {s['OEE']:.1f}% frente a meta de {goals['OEE']:.1f}%.",
                "Acción": "Separar la pérdida dominante entre disponibilidad, rendimiento y calidad y asignar responsable por máquina.",
            }
        )

    if s["Disponibilidad"] < goals["Disponibilidad"]:
        worst = machine.sort_values("Disponibilidad").iloc[0]
        recs.append(
            {
                "Prioridad": "Alta",
                "Hallazgo": f"Disponibilidad baja. {worst['Maquina']} registra {worst['Disponibilidad']:.1f}%.",
                "Acción": "Aplicar Pareto de paradas y priorizar mantenimiento preventivo sobre las causas con mayor tiempo perdido.",
            }
        )

    if s["Rendimiento"] < goals["Rendimiento"]:
        worst = machine.sort_values("Rendimiento").iloc[0]
        recs.append(
            {
                "Prioridad": "Media",
                "Hallazgo": f"Rendimiento por debajo de objetivo. {worst['Maquina']} registra {worst['Rendimiento']:.1f}%.",
                "Acción": "Revisar microparadas, velocidad estándar, abastecimiento y condiciones del método de trabajo.",
            }
        )

    if s["Calidad"] < goals["Calidad"]:
        worst = machine.sort_values("Calidad").iloc[0]
        recs.append(
            {
                "Prioridad": "Alta",
                "Hallazgo": f"Calidad inferior a meta. {worst['Maquina']} registra {worst['Calidad']:.1f}%.",
                "Acción": "Ejecutar análisis de defectos, verificación de parámetros críticos y plan de reducción de scrap.",
            }
        )

    if s["Tiempo_ciclo"] > goals["Ciclo"]:
        worst = machine.sort_values("Tiempo_ciclo", ascending=False).iloc[0]
        recs.append(
            {
                "Prioridad": "Media",
                "Hallazgo": f"Tiempo de ciclo elevado. {worst['Maquina']} promedia {worst['Tiempo_ciclo']:.1f} s.",
                "Acción": "Comparar ciclo real contra estándar, observar esperas internas y estandarizar el método del cuello de botella.",
            }
        )

    downtime = (
        df.groupby("Causa_parada", as_index=False)["Tiempo_parada_h"]
        .sum()
        .sort_values("Tiempo_parada_h", ascending=False)
    )
    downtime = downtime[downtime["Causa_parada"] != "Sin parada relevante"]

    if not downtime.empty:
        top_cause = downtime.iloc[0]
        recs.append(
            {
                "Prioridad": "Alta",
                "Hallazgo": f"La principal causa de parada es '{top_cause['Causa_parada']}' con {top_cause['Tiempo_parada_h']:.1f} h.",
                "Acción": "Abrir acción correctiva sobre esta causa antes de atacar pérdidas de menor impacto.",
            }
        )

    if not recs:
        recs.append(
            {
                "Prioridad": "Baja",
                "Hallazgo": "Los principales indicadores se encuentran dentro de las metas configuradas.",
                "Acción": "Mantener controles, estandarizar las buenas prácticas y revisar tendencias para prevenir deterioros.",
            }
        )

    return pd.DataFrame(recs)


# ============================================================
# ESTADO DE SESIÓN
# ============================================================
if "data" not in st.session_state:
    st.session_state["data"] = generate_sample_data()

if "source_name" not in st.session_state:
    st.session_state["source_name"] = "Base simulada industrial"

if "upload_hash" not in st.session_state:
    st.session_state["upload_hash"] = None


# ============================================================
# SIDEBAR: NAVEGACIÓN, DATOS Y FILTROS
# ============================================================
st.sidebar.markdown("## Operations Center")
st.sidebar.caption("Analítica de Operaciones y KPIs")

section = st.sidebar.radio(
    "Módulo",
    [
        "Dashboard Ejecutivo",
        "Análisis OEE",
        "Productividad",
        "Tiempos de Ciclo",
        "Pérdidas y Paradas",
        "Análisis por Máquina",
        "Datos de Producción",
        "Simulador de Escenarios",
        "Recomendaciones",
    ],
)

with st.sidebar.expander("Fuente de datos", expanded=False):
    st.caption(f"Activa: {st.session_state['source_name']}")

    uploaded = st.file_uploader(
        "Cargar Excel o CSV",
        type=["xlsx", "xls", "csv"],
        help="Use la plantilla de datos disponible en el módulo Datos de Producción.",
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if file_hash != st.session_state["upload_hash"]:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    loaded = pd.read_csv(io.BytesIO(file_bytes))
                else:
                    loaded = pd.read_excel(io.BytesIO(file_bytes))

                loaded = clean_base_data(loaded)

                st.session_state["data"] = loaded
                st.session_state["source_name"] = uploaded.name
                st.session_state["upload_hash"] = file_hash
                st.success("Archivo cargado.")
            except Exception as exc:
                st.error(str(exc))

    if st.button("Restablecer datos simulados", use_container_width=True):
        st.session_state["data"] = generate_sample_data()
        st.session_state["source_name"] = "Base simulada industrial"
        st.session_state["upload_hash"] = None
        st.rerun()

data_base = clean_base_data(st.session_state["data"])
data_all = add_metrics(data_base)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros generales")

date_min = data_all["Fecha"].min().date()
date_max = data_all["Fecha"].max().date()

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
else:
    start_date = pd.Timestamp(date_min)
    end_date = pd.Timestamp(date_max)

def multi_filter(label, column):
    options = sorted(data_all[column].dropna().astype(str).unique())
    return st.sidebar.multiselect(label, options, default=options)

selected_plants = multi_filter("Planta", "Planta")
selected_lines = multi_filter("Línea de producción", "Linea")
selected_machines = multi_filter("Máquina", "Maquina")
selected_shifts = multi_filter("Turno", "Turno")

st.sidebar.markdown("---")
with st.sidebar.expander("Metas de gestión", expanded=False):
    goal_oee = st.number_input("OEE mínimo (%)", 1.0, 100.0, 85.0, 0.5)
    goal_availability = st.number_input("Disponibilidad mínima (%)", 1.0, 100.0, 90.0, 0.5)
    goal_performance = st.number_input("Rendimiento mínimo (%)", 1.0, 100.0, 95.0, 0.5)
    goal_quality = st.number_input("Calidad mínima (%)", 1.0, 100.0, 99.0, 0.1)
    goal_productivity = st.number_input("Productividad mínima (unid/h)", 1.0, 10000.0, 90.0, 1.0)
    goal_cycle = st.number_input("Tiempo de ciclo máximo (s)", 1.0, 10000.0, 50.0, 1.0)

goals = {
    "OEE": goal_oee,
    "Disponibilidad": goal_availability,
    "Rendimiento": goal_performance,
    "Calidad": goal_quality,
    "Productividad": goal_productivity,
    "Ciclo": goal_cycle,
}

filtered = data_all[
    (data_all["Fecha"] >= start_date)
    & (data_all["Fecha"] <= end_date)
    & (data_all["Planta"].isin(selected_plants))
    & (data_all["Linea"].isin(selected_lines))
    & (data_all["Maquina"].isin(selected_machines))
    & (data_all["Turno"].isin(selected_shifts))
].copy()

if filtered.empty:
    show_header("Sin datos para mostrar", "Revise los filtros seleccionados.")
    st.warning("La combinación actual de filtros no contiene registros.")
    st.stop()

summary = kpi_summary(filtered)
machine_metrics = aggregate_machine_metrics(filtered)
daily_metrics = aggregate_daily_metrics(filtered)


# ============================================================
# DASHBOARD EJECUTIVO
# ============================================================
if section == "Dashboard Ejecutivo":
    show_header(
        "Dashboard Ejecutivo",
        "Centro de control para monitoreo integral de desempeño operativo y toma de decisiones.",
    )

    st.markdown(
        f"""
        <span class="status-chip">Fuente: {st.session_state['source_name']}</span>
        <span class="status-chip">Registros filtrados: {len(filtered):,}</span>
        <span class="status-chip">Última fecha: {filtered['Fecha'].max().strftime('%d/%m/%Y')}</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "OEE",
        f"{summary['OEE']:.1f}%",
        f"{summary['OEE'] - goal_oee:+.1f} pp vs meta",
    )
    c2.metric(
        "Productividad",
        f"{summary['Productividad']:.1f} unid/h",
        f"{summary['Productividad'] - goal_productivity:+.1f} vs meta",
    )
    c3.metric(
        "Tiempo de ciclo",
        f"{summary['Tiempo_ciclo']:.1f} s",
        f"{summary['Tiempo_ciclo'] - goal_cycle:+.1f} s vs máximo",
        delta_color="inverse",
    )
    c4.metric(
        "Producción buena",
        f"{summary['Produccion_buena']:,.0f}",
        f"{summary['Defectos']:,.0f} defectuosas",
        delta_color="inverse",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Disponibilidad",
        f"{summary['Disponibilidad']:.1f}%",
        f"{summary['Disponibilidad'] - goal_availability:+.1f} pp",
    )
    c6.metric(
        "Rendimiento",
        f"{summary['Rendimiento']:.1f}%",
        f"{summary['Rendimiento'] - goal_performance:+.1f} pp",
    )
    c7.metric(
        "Calidad",
        f"{summary['Calidad']:.1f}%",
        f"{summary['Calidad'] - goal_quality:+.1f} pp",
    )
    c8.metric(
        "Tiempo perdido",
        f"{summary['Paradas_h']:,.1f} h",
        "Paradas acumuladas",
        delta_color="off",
    )

    problems = []
    if summary["OEE"] < goal_oee:
        problems.append("OEE")
    if summary["Disponibilidad"] < goal_availability:
        problems.append("disponibilidad")
    if summary["Rendimiento"] < goal_performance:
        problems.append("rendimiento")
    if summary["Calidad"] < goal_quality:
        problems.append("calidad")
    if summary["Tiempo_ciclo"] > goal_cycle:
        problems.append("tiempo de ciclo")

    if problems:
        show_alert(
            "Atención gerencial: existen desviaciones en " + ", ".join(problems) + ".",
            "warning",
        )
    else:
        show_alert("La operación cumple las metas configuradas.", "ok")

    left, right = st.columns([1.65, 1])

    with left:
        fig = px.line(
            daily_metrics,
            x="Fecha",
            y="OEE",
            markers=True,
            title="Evolución diaria del OEE",
            labels={"OEE": "OEE (%)"},
        )
        fig.add_hline(
            y=goal_oee,
            line_dash="dash",
            line_color=RED,
            annotation_text=f"Meta {goal_oee:.1f}%",
        )
        format_plot(fig, 390, False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        gauge_color = status_color(
            metric_status(summary["OEE"], goal_oee)
        )
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=summary["OEE"],
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "OEE actual"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 60], "color": "#F8E0E0"},
                        {"range": [60, 85], "color": "#FFF1CC"},
                        {"range": [85, 100], "color": "#DDF1E7"},
                    ],
                    "threshold": {
                        "line": {"color": DARK, "width": 3},
                        "value": goal_oee,
                    },
                },
            )
        )
        format_plot(fig, 390, False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        comp = pd.DataFrame(
            {
                "Indicador": ["Disponibilidad", "Rendimiento", "Calidad"],
                "Valor": [
                    summary["Disponibilidad"],
                    summary["Rendimiento"],
                    summary["Calidad"],
                ],
                "Meta": [
                    goal_availability,
                    goal_performance,
                    goal_quality,
                ],
            }
        )
        fig = px.bar(
            comp,
            x="Indicador",
            y="Valor",
            text_auto=".1f",
            title="Componentes del OEE",
            labels={"Valor": "Porcentaje (%)"},
        )
        fig.update_traces(marker_color=BLUE)
        fig.update_yaxes(range=[0, 105])
        format_plot(fig, 370, False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        temp = machine_metrics.sort_values("OEE")
        fig = px.bar(
            temp,
            x="OEE",
            y="Maquina",
            orientation="h",
            text_auto=".1f",
            title="OEE por máquina",
            labels={"OEE": "OEE (%)", "Maquina": "Máquina"},
        )
        fig.add_vline(
            x=goal_oee,
            line_dash="dash",
            line_color=RED,
            annotation_text="Meta",
        )
        fig.update_traces(marker_color=BLUE)
        format_plot(fig, 370, False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Semáforo de desempeño")
    traffic = machine_metrics[
        ["Maquina", "OEE", "Disponibilidad", "Rendimiento", "Calidad", "Productividad", "Tiempo_ciclo"]
    ].copy()

    traffic["Estado OEE"] = traffic["OEE"].apply(
        lambda x: metric_status(x, goal_oee)
    )
    traffic["Estado ciclo"] = traffic["Tiempo_ciclo"].apply(
        lambda x: metric_status(x, goal_cycle, False)
    )

    st.dataframe(
        traffic.sort_values("OEE"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "OEE": st.column_config.NumberColumn("OEE (%)", format="%.1f"),
            "Disponibilidad": st.column_config.NumberColumn("Disponibilidad (%)", format="%.1f"),
            "Rendimiento": st.column_config.NumberColumn("Rendimiento (%)", format="%.1f"),
            "Calidad": st.column_config.NumberColumn("Calidad (%)", format="%.1f"),
            "Productividad": st.column_config.NumberColumn("Productividad", format="%.1f"),
            "Tiempo_ciclo": st.column_config.NumberColumn("Ciclo (s)", format="%.1f"),
        },
    )


# ============================================================
# ANÁLISIS OEE
# ============================================================
elif section == "Análisis OEE":
    show_header(
        "Análisis OEE",
        "Descomposición del indicador para identificar si la pérdida proviene de disponibilidad, velocidad o calidad.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OEE", f"{summary['OEE']:.1f}%", f"{summary['OEE']-goal_oee:+.1f} pp")
    c2.metric("Disponibilidad", f"{summary['Disponibilidad']:.1f}%", f"{summary['Disponibilidad']-goal_availability:+.1f} pp")
    c3.metric("Rendimiento", f"{summary['Rendimiento']:.1f}%", f"{summary['Rendimiento']-goal_performance:+.1f} pp")
    c4.metric("Calidad", f"{summary['Calidad']:.1f}%", f"{summary['Calidad']-goal_quality:+.1f} pp")

    long_machine = machine_metrics.melt(
        id_vars="Maquina",
        value_vars=["Disponibilidad", "Rendimiento", "Calidad"],
        var_name="Componente",
        value_name="Valor",
    )

    fig = px.bar(
        long_machine,
        x="Maquina",
        y="Valor",
        color="Componente",
        barmode="group",
        title="Descomposición OEE por máquina",
        labels={"Valor": "Porcentaje (%)", "Maquina": "Máquina"},
        color_discrete_sequence=[BLUE, AMBER, GREEN],
    )
    fig.update_yaxes(range=[0, 105])
    format_plot(fig, 430, True)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            machine_metrics,
            x="Disponibilidad",
            y="Calidad",
            size="Produccion_total",
            color="OEE",
            text="Maquina",
            title="Mapa de desempeño por máquina",
            labels={
                "Disponibilidad": "Disponibilidad (%)",
                "Calidad": "Calidad (%)",
                "OEE": "OEE (%)",
            },
            color_continuous_scale="RdYlGn",
        )
        fig.add_vline(x=goal_availability, line_dash="dash")
        fig.add_hline(y=goal_quality, line_dash="dash")
        fig.update_traces(textposition="top center")
        format_plot(fig, 390, True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        temp = machine_metrics.sort_values("OEE")
        fig = px.bar(
            temp,
            x="Maquina",
            y="OEE",
            text_auto=".1f",
            title="Ranking de OEE",
            labels={"OEE": "OEE (%)", "Maquina": "Máquina"},
        )
        fig.update_traces(
            marker_color=[
                GREEN if x >= goal_oee else RED for x in temp["OEE"]
            ]
        )
        fig.add_hline(y=goal_oee, line_dash="dash", annotation_text="Meta")
        format_plot(fig, 390, False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Metodología de cálculo"):
        st.markdown(
            """
            **Disponibilidad = Tiempo operativo / Horas programadas**

            **Rendimiento = Producción total / Capacidad esperada**

            **Calidad = Producción buena / Producción total**

            **OEE = Disponibilidad x Rendimiento x Calidad**

            El OEE se recalcula con los registros que permanecen después de aplicar
            los filtros de fecha, planta, línea, máquina y turno.
            """
        )


# ============================================================
# PRODUCTIVIDAD
# ============================================================
elif section == "Productividad":
    show_header(
        "Productividad",
        "Análisis de unidades buenas obtenidas por hora efectiva de operación.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Productividad global", f"{summary['Productividad']:.1f} unid/h", f"{summary['Productividad']-goal_productivity:+.1f} vs meta")
    best = machine_metrics.sort_values("Productividad", ascending=False).iloc[0]
    worst = machine_metrics.sort_values("Productividad").iloc[0]
    c2.metric("Mejor máquina", best["Maquina"], f"{best['Productividad']:.1f} unid/h")
    c3.metric("Máquina crítica", worst["Maquina"], f"{worst['Productividad']:.1f} unid/h", delta_color="inverse")

    left, right = st.columns(2)

    with left:
        temp = machine_metrics.sort_values("Productividad", ascending=False)
        fig = px.bar(
            temp,
            x="Maquina",
            y="Productividad",
            text_auto=".1f",
            title="Productividad por máquina",
            labels={"Productividad": "Unidades buenas / hora", "Maquina": "Máquina"},
        )
        fig.add_hline(y=goal_productivity, line_dash="dash", line_color=RED, annotation_text="Meta")
        fig.update_traces(marker_color=BLUE)
        format_plot(fig, 400, False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            daily_metrics,
            x="Fecha",
            y="Productividad",
            markers=True,
            title="Tendencia diaria de productividad",
            labels={"Productividad": "Unidades buenas / hora"},
        )
        fig.add_hline(y=goal_productivity, line_dash="dash", line_color=RED, annotation_text="Meta")
        format_plot(fig, 400, False)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        machine_metrics,
        x="OEE",
        y="Productividad",
        size="Produccion_buena",
        color="Tiempo_ciclo",
        text="Maquina",
        title="Relación entre OEE y productividad",
        labels={
            "OEE": "OEE (%)",
            "Productividad": "Unidades buenas / hora",
            "Tiempo_ciclo": "Ciclo (s)",
        },
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_traces(textposition="top center")
    format_plot(fig, 430, True)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TIEMPOS DE CICLO
# ============================================================
elif section == "Tiempos de Ciclo":
    show_header(
        "Tiempos de Ciclo",
        "Comparación del ciclo real contra el estándar para detectar pérdidas de velocidad y cuellos de botella.",
    )

    cycle = (
        filtered.groupby("Maquina", as_index=False)
        .agg(
            Ciclo_real=("Tiempo_ciclo_real_s", "mean"),
            Ciclo_ideal=("Tiempo_ciclo_ideal_s", "mean"),
        )
    )
    cycle["Desviacion_pct"] = (
        (cycle["Ciclo_real"] - cycle["Ciclo_ideal"])
        / cycle["Ciclo_ideal"] * 100
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Ciclo promedio", f"{summary['Tiempo_ciclo']:.1f} s", f"{summary['Tiempo_ciclo']-goal_cycle:+.1f} s vs máximo", delta_color="inverse")
    worst_cycle = cycle.sort_values("Desviacion_pct", ascending=False).iloc[0]
    c2.metric("Mayor desviación", worst_cycle["Maquina"], f"{worst_cycle['Desviacion_pct']:.1f}% sobre ideal", delta_color="inverse")
    c3.metric("Máquinas analizadas", f"{cycle['Maquina'].nunique()}")

    long_cycle = cycle.melt(
        id_vars="Maquina",
        value_vars=["Ciclo_real", "Ciclo_ideal"],
        var_name="Tipo",
        value_name="Segundos",
    )

    fig = px.bar(
        long_cycle,
        x="Maquina",
        y="Segundos",
        color="Tipo",
        barmode="group",
        text_auto=".1f",
        title="Tiempo de ciclo real vs ideal",
        labels={"Maquina": "Máquina"},
        color_discrete_sequence=[RED, GREEN],
    )
    format_plot(fig, 420, True)
    st.plotly_chart(fig, use_container_width=True)

    daily_cycle = (
        filtered.groupby("Fecha", as_index=False)["Tiempo_ciclo_real_s"]
        .mean()
    )
    fig = px.line(
        daily_cycle,
        x="Fecha",
        y="Tiempo_ciclo_real_s",
        markers=True,
        title="Evolución del tiempo de ciclo",
        labels={"Tiempo_ciclo_real_s": "Tiempo de ciclo real (s)"},
    )
    fig.add_hline(y=goal_cycle, line_dash="dash", line_color=RED, annotation_text="Máximo")
    format_plot(fig, 390, False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cycle.sort_values("Desviacion_pct", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ciclo_real": st.column_config.NumberColumn("Ciclo real (s)", format="%.2f"),
            "Ciclo_ideal": st.column_config.NumberColumn("Ciclo ideal (s)", format="%.2f"),
            "Desviacion_pct": st.column_config.NumberColumn("Desviación (%)", format="%.1f"),
        },
    )


# ============================================================
# PÉRDIDAS Y PARADAS
# ============================================================
elif section == "Pérdidas y Paradas":
    show_header(
        "Pérdidas y Paradas",
        "Priorización de causas mediante Pareto y cuantificación de pérdidas operativas.",
    )

    downtime = (
        filtered.groupby("Causa_parada", as_index=False)["Tiempo_parada_h"]
        .sum()
        .sort_values("Tiempo_parada_h", ascending=False)
    )
    downtime = downtime[downtime["Causa_parada"] != "Sin parada relevante"].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tiempo total de parada", f"{summary['Paradas_h']:,.1f} h")
    c2.metric("Defectos", f"{summary['Defectos']:,.0f} unid.")
    c3.metric("Pérdida por velocidad", f"{filtered['Perdida_velocidad_unid'].sum():,.0f} unid.")
    c4.metric("Tasa de defectos", f"{(summary['Defectos']/max(summary['Produccion_total'],1)*100):.2f}%")

    if not downtime.empty:
        downtime["Acumulado_pct"] = downtime["Tiempo_parada_h"].cumsum() / downtime["Tiempo_parada_h"].sum() * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=downtime["Causa_parada"],
                y=downtime["Tiempo_parada_h"],
                name="Horas de parada",
                marker_color=BLUE,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=downtime["Causa_parada"],
                y=downtime["Acumulado_pct"],
                name="% acumulado",
                mode="lines+markers",
                line=dict(color=RED, width=3),
            ),
            secondary_y=True,
        )
        fig.add_hline(y=80, line_dash="dash", line_color=AMBER, secondary_y=True)
        fig.update_yaxes(title_text="Horas de parada", secondary_y=False)
        fig.update_yaxes(title_text="Acumulado (%)", range=[0, 105], secondary_y=True)
        fig.update_layout(title="Pareto de causas de parada")
        format_plot(fig, 450, True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No existen paradas relevantes en la selección actual.")

    loss = pd.DataFrame(
        {
            "Tipo de pérdida": [
                "Pérdida por velocidad",
                "Pérdida por calidad",
            ],
            "Unidades": [
                filtered["Perdida_velocidad_unid"].sum(),
                filtered["Perdida_calidad_unid"].sum(),
            ],
        }
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            loss,
            x="Tipo de pérdida",
            y="Unidades",
            text_auto=",.0f",
            title="Pérdidas estimadas en unidades",
        )
        fig.update_traces(marker_color=[AMBER, RED])
        format_plot(fig, 370, False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        down_machine = (
            filtered.groupby("Maquina", as_index=False)["Tiempo_parada_h"]
            .sum()
            .sort_values("Tiempo_parada_h", ascending=False)
        )
        fig = px.bar(
            down_machine,
            x="Maquina",
            y="Tiempo_parada_h",
            text_auto=".1f",
            title="Horas de parada por máquina",
            labels={"Maquina": "Máquina", "Tiempo_parada_h": "Horas"},
        )
        fig.update_traces(marker_color=RED)
        format_plot(fig, 370, False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ANÁLISIS POR MÁQUINA
# ============================================================
elif section == "Análisis por Máquina":
    show_header(
        "Análisis por Máquina",
        "Diagnóstico individual para localizar equipos críticos y comparar desempeño contra objetivos.",
    )

    machine_options = sorted(filtered["Maquina"].unique())
    selected_machine = st.selectbox("Seleccione una máquina", machine_options)

    machine_data = filtered[filtered["Maquina"] == selected_machine].copy()
    ms = kpi_summary(machine_data)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OEE", f"{ms['OEE']:.1f}%")
    c2.metric("Disponibilidad", f"{ms['Disponibilidad']:.1f}%")
    c3.metric("Rendimiento", f"{ms['Rendimiento']:.1f}%")
    c4.metric("Calidad", f"{ms['Calidad']:.1f}%")
    c5.metric("Productividad", f"{ms['Productividad']:.1f} unid/h")

    left, right = st.columns([1.3, 1])

    with left:
        mdaily = aggregate_daily_metrics(machine_data)
        fig = px.line(
            mdaily,
            x="Fecha",
            y="OEE",
            markers=True,
            title=f"Evolución del OEE - {selected_machine}",
            labels={"OEE": "OEE (%)"},
        )
        fig.add_hline(y=goal_oee, line_dash="dash", line_color=RED, annotation_text="Meta")
        format_plot(fig, 410, False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        categories = ["Disponibilidad", "Rendimiento", "Calidad", "OEE"]
        actual = [
            ms["Disponibilidad"],
            ms["Rendimiento"],
            ms["Calidad"],
            ms["OEE"],
        ]
        target = [
            goal_availability,
            goal_performance,
            goal_quality,
            goal_oee,
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=actual + [actual[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Actual",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=target + [target[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Meta",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Perfil de desempeño vs meta",
        )
        format_plot(fig, 410, True)
        st.plotly_chart(fig, use_container_width=True)

    causes = (
        machine_data.groupby("Causa_parada", as_index=False)["Tiempo_parada_h"]
        .sum()
        .sort_values("Tiempo_parada_h", ascending=False)
    )
    causes = causes[causes["Causa_parada"] != "Sin parada relevante"]

    if not causes.empty:
        fig = px.bar(
            causes,
            x="Causa_parada",
            y="Tiempo_parada_h",
            text_auto=".1f",
            title=f"Causas de parada - {selected_machine}",
            labels={"Causa_parada": "Causa", "Tiempo_parada_h": "Horas de parada"},
        )
        fig.update_traces(marker_color=RED)
        format_plot(fig, 380, False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DATOS DE PRODUCCIÓN
# ============================================================
elif section == "Datos de Producción":
    show_header(
        "Datos de Producción",
        "Carga, edición, validación y descarga de la base utilizada por toda la aplicación.",
    )

    show_alert(
        "Los cambios guardados en esta sección recalculan automáticamente todos los módulos, KPIs, gráficos, alertas y recomendaciones.",
        "ok",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        template_csv = template_dataframe().to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar plantilla CSV",
            data=template_csv,
            file_name="plantilla_operaciones.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "Descargar plantilla Excel",
            data=to_excel_bytes(template_dataframe()),
            file_name="plantilla_operaciones.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with c3:
        st.download_button(
            "Descargar base actual",
            data=to_excel_bytes(data_base),
            file_name="base_operaciones_actual.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.subheader("Editor de registros")
    st.caption(
        "Puede modificar celdas, pegar datos desde Excel o agregar/eliminar filas. "
        "Después pulse Guardar cambios."
    )

    edited = st.data_editor(
        data_base,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="production_editor",
        column_config={
            "Fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "Horas_programadas": st.column_config.NumberColumn("Horas programadas", min_value=0.01),
            "Tiempo_operativo_h": st.column_config.NumberColumn("Tiempo operativo (h)", min_value=0.0),
            "Tiempo_parada_h": st.column_config.NumberColumn("Tiempo parada (h)", min_value=0.0),
            "Produccion_total": st.column_config.NumberColumn("Producción total", min_value=0),
            "Produccion_buena": st.column_config.NumberColumn("Producción buena", min_value=0),
            "Produccion_defectuosa": st.column_config.NumberColumn("Producción defectuosa", min_value=0),
            "Capacidad_esperada": st.column_config.NumberColumn("Capacidad esperada", min_value=0.01),
            "Tiempo_ciclo_real_s": st.column_config.NumberColumn("Ciclo real (s)", min_value=0.01),
            "Tiempo_ciclo_ideal_s": st.column_config.NumberColumn("Ciclo ideal (s)", min_value=0.01),
        },
    )

    left, right = st.columns([1, 4])

    with left:
        if st.button("Guardar cambios", type="primary", use_container_width=True):
            try:
                cleaned = clean_base_data(edited)
                st.session_state["data"] = cleaned
                st.session_state["source_name"] = "Datos editados en la aplicación"
                st.session_state["upload_hash"] = None
                st.success("Cambios guardados correctamente.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with right:
        st.caption(
            f"Registros activos: {len(data_base):,} | "
            f"Desde {data_base['Fecha'].min().strftime('%d/%m/%Y')} "
            f"hasta {data_base['Fecha'].max().strftime('%d/%m/%Y')}"
        )

    with st.expander("Diccionario de datos y reglas"):
        dictionary = pd.DataFrame(
            {
                "Campo": REQUIRED_COLUMNS,
                "Descripción": [
                    "Fecha del registro",
                    "Sede o planta industrial",
                    "Línea de producción",
                    "Código de máquina",
                    "Turno de trabajo",
                    "Horas planificadas para producir",
                    "Horas reales de operación",
                    "Horas de parada",
                    "Unidades totales producidas",
                    "Unidades conformes",
                    "Unidades no conformes",
                    "Producción esperada para el periodo",
                    "Tiempo real por unidad en segundos",
                    "Tiempo estándar o ideal por unidad",
                    "Motivo principal de parada",
                ],
            }
        )
        st.dataframe(dictionary, use_container_width=True, hide_index=True)


# ============================================================
# SIMULADOR
# ============================================================
elif section == "Simulador de Escenarios":
    show_header(
        "Simulador de Escenarios",
        "Evaluación de mejoras potenciales antes de decidir acciones de inversión o mejora continua.",
    )

    st.markdown(
        '<div class="section-note">Ajuste las mejoras esperadas. El escenario se compara con el desempeño filtrado actual.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        improvement_availability = st.slider(
            "Mejora en disponibilidad (pp)", 0.0, 20.0, 5.0, 0.5
        )
    with c2:
        improvement_performance = st.slider(
            "Mejora en rendimiento (pp)", 0.0, 20.0, 4.0, 0.5
        )
    with c3:
        improvement_quality = st.slider(
            "Mejora en calidad (pp)", 0.0, 10.0, 2.0, 0.5
        )

    new_a = min(100, summary["Disponibilidad"] + improvement_availability)
    new_p = min(100, summary["Rendimiento"] + improvement_performance)
    new_q = min(100, summary["Calidad"] + improvement_quality)
    new_oee = (new_a / 100) * (new_p / 100) * (new_q / 100) * 100

    ratio = new_oee / max(summary["OEE"], 0.01)
    estimated_good = summary["Produccion_buena"] * ratio
    extra_units = max(0, estimated_good - summary["Produccion_buena"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OEE actual", f"{summary['OEE']:.1f}%")
    c2.metric("OEE escenario", f"{new_oee:.1f}%", f"{new_oee-summary['OEE']:+.1f} pp")
    c3.metric("Producción buena estimada", f"{estimated_good:,.0f}", f"{extra_units:+,.0f} unid.")
    c4.metric("Mejora relativa OEE", f"{((new_oee/max(summary['OEE'],0.01))-1)*100:.1f}%")

    scenario = pd.DataFrame(
        {
            "Indicador": ["Disponibilidad", "Rendimiento", "Calidad", "OEE"],
            "Actual": [
                summary["Disponibilidad"],
                summary["Rendimiento"],
                summary["Calidad"],
                summary["OEE"],
            ],
            "Escenario": [new_a, new_p, new_q, new_oee],
        }
    )
    long_s = scenario.melt(
        id_vars="Indicador",
        var_name="Escenario",
        value_name="Valor",
    )

    fig = px.bar(
        long_s,
        x="Indicador",
        y="Valor",
        color="Escenario",
        barmode="group",
        text_auto=".1f",
        title="Comparación actual vs escenario mejorado",
        labels={"Valor": "Porcentaje (%)"},
        color_discrete_sequence=[MID, GREEN],
    )
    fig.update_yaxes(range=[0, 105])
    format_plot(fig, 430, True)
    st.plotly_chart(fig, use_container_width=True)

    if new_oee >= goal_oee and summary["OEE"] < goal_oee:
        show_alert(
            f"El escenario propuesto permitiría superar la meta de OEE de {goal_oee:.1f}%.",
            "ok",
        )
    elif new_oee < goal_oee:
        show_alert(
            f"Aun con las mejoras simuladas, el OEE quedaría en {new_oee:.1f}%, por debajo de la meta.",
            "warning",
        )


# ============================================================
# RECOMENDACIONES
# ============================================================
elif section == "Recomendaciones":
    show_header(
        "Recomendaciones",
        "Motor de reglas para convertir desviaciones operativas en acciones priorizadas de mejora.",
    )

    recommendations = recommendation_engine(filtered, goals)

    priority_order = {"Alta": 0, "Media": 1, "Baja": 2}
    recommendations["Orden"] = recommendations["Prioridad"].map(priority_order)
    recommendations = recommendations.sort_values("Orden").drop(columns="Orden")

    for _, row in recommendations.iterrows():
        level = "critical" if row["Prioridad"] == "Alta" else "warning" if row["Prioridad"] == "Media" else "ok"
        show_alert(
            f"<strong>{row['Prioridad']}</strong> | {row['Hallazgo']}<br>"
            f"<strong>Acción sugerida:</strong> {row['Acción']}",
            level,
        )

    st.subheader("Matriz de priorización")
    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Máquinas que requieren atención")
    attention = machine_metrics.copy()
    attention["Estado OEE"] = attention["OEE"].apply(lambda x: metric_status(x, goal_oee))
    attention["Estado calidad"] = attention["Calidad"].apply(lambda x: metric_status(x, goal_quality))
    attention["Estado disponibilidad"] = attention["Disponibilidad"].apply(lambda x: metric_status(x, goal_availability))

    st.dataframe(
        attention.sort_values("OEE")[
            ["Maquina", "OEE", "Disponibilidad", "Rendimiento", "Calidad", "Estado OEE", "Estado disponibilidad", "Estado calidad"]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "OEE": st.column_config.NumberColumn("OEE (%)", format="%.1f"),
            "Disponibilidad": st.column_config.NumberColumn("Disponibilidad (%)", format="%.1f"),
            "Rendimiento": st.column_config.NumberColumn("Rendimiento (%)", format="%.1f"),
            "Calidad": st.column_config.NumberColumn("Calidad (%)", format="%.1f"),
        },
    )


# ============================================================
# PIE GENERAL
# ============================================================
st.markdown(
    f"""
    <div class="footer">
        Operations Intelligence Center | Python + Streamlit + Plotly + Pandas + NumPy |
        Actualización de vista: {datetime.now().strftime('%d/%m/%Y %H:%M')}
    </div>
    """,
    unsafe_allow_html=True,
)
