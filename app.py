import streamlit as st
from datetime import datetime, timezone
import pandas as pd
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
import locale
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

def show_data_grid(df: pd.DataFrame):

    with st.expander("Explorar datos", expanded=True):
        st.dataframe(df.iloc[:, 5:-2], hide_index=True,
                     use_container_width=True)


def calculate_avg_and_delta(df, year_filter, column_name):
    if len(year_filter) == 2:
        current_year = max(year_filter)
        previous_year = min(year_filter)
        current_year_avg = df[df['Year'] == current_year][column_name].mean()
        previous_year_avg = df[df['Year'] == previous_year][column_name].mean()
        delta = current_year_avg - previous_year_avg
        return current_year_avg, delta
    elif len(year_filter) == 1:
        current_year_avg = df[df['Year'] == year_filter[0]][column_name].mean()
        return current_year_avg, None
    else:
        return None, None


def main():
    st.set_page_config(page_title="Miraflores Respira", page_icon="🌬️", layout="wide", menu_items={
        'About': "MirafloresRespira – 📊 Monitoreo de Calidad del Aire en Miraflores, Lima, Perú | Dashboard en \
        Tiempo Real 🌍. Consulta en tiempo real los niveles de contaminación en el aire de Miraflores: \
        PM2.5, PM10, CO, NO2, O3, y otros indicadores clave para la salud. Conoce cómo la calidad del \
        aire impacta tu bienestar y toma decisiones informadas para protegerte a ti y a tu familia. \
        🔍 Actualizaciones continuas y alertas para que estés siempre al tanto. Ideal para residentes, visitantes, y profesionales en salud ambiental"
    })
    
    st.title("Miraflores Respira 🍃")

    df = pd.read_csv('data/air_qa.csv')

    with st.sidebar:
        st.header("Filtros: ")
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df['Year'] = df['Fecha'].dt.year
        df['Month'] = df['Fecha'].dt.month
        year_filter = st.multiselect(
            "Año: ", options=df["Year"].unique(), default=df["Year"].unique())
        district_filter = st.multiselect(
            "Estación: ", options=df["Distrito"].unique(), default=df["Distrito"].unique())

        temp_min, temp_max = st.slider(
            "Temperatura °C:", min_value=df["Temperatura (C)"].min(), max_value=df["Temperatura (C)"].max(), value=(df["Temperatura (C)"].min(), df["Temperatura (C)"].max()))
        st.markdown("""
            <style>
                .stSlider > div > div > div > div > div {
                    background: #ffffff; 
                    border-radius: 10%; 
                    
                }
            </style>
            """, unsafe_allow_html=True)
        if not year_filter:
            st.info("Selecciona un año.")
            year_filter = df["Year"].unique()
        if not district_filter:
            st.info("Selecciona una estación.")
            district_filter = df["Distrito"].unique()

    df_filtered = df[df["Year"].isin(year_filter) & (df["Distrito"].isin(district_filter)) & (
        df["Temperatura (C)"] >= temp_min) & (df["Temperatura (C)"] <= temp_max)]

    show_data_grid(df_filtered)

    average_temp = round(df_filtered["Temperatura (C)"].mean(), 1)
    average_pres = round(df_filtered["Presion (hPa)"].mean(), 2)

    left_col, mid_col = st.columns(2)
    with left_col:
        average_temp, delta_temp = calculate_avg_and_delta(
            df, year_filter, "Temperatura (C)")
        if average_temp is not None:
            st.metric(label="Temperatura Promedio", value=f"{average_temp:.2f} °C", delta=(
                f"{delta_temp:.2f} °C" if delta_temp is not None else None))

    with mid_col:
        average_pres, delta_pres = calculate_avg_and_delta(
            df, year_filter, "Presion (hPa)")
        if average_pres is not None:
            st.metric(label="Presión Promedio", value=f"{average_pres:.2f} hPa", delta=(
                f"{delta_pres:.2f} hPa" if delta_pres is not None else None))

    grouped = df_filtered.groupby(["Year", "Month", "Distrito"])[
        ["PM2.5 (ug/m3)", "PM10 (ug/m3)", "O3 (ug/m3)", "CO (ug/m3)"]].mean().reset_index()
    grouped['Formatted Date'] = pd.to_datetime(
        grouped[['Year', 'Month']].assign(Day=1)).dt.strftime('%b %y').str.capitalize()
    left_col, right_col = st.columns(2)

    fig_bars = px.bar(grouped, x="Year", y="O3 (ug/m3)", color="Distrito", title="Promedio de Ozono por estación").update_layout(
        xaxis_title="Año", yaxis_title="Ozono (ug/m^3)", xaxis=dict(tickmode='linear', dtick=1))
    fig_co = px.line(grouped, x="Formatted Date", y="CO (ug/m3)", color="Distrito", markers=True,
                     title="Promedio de CO (Monoxido de carbono) por estación").update_layout(xaxis_title="Mes", yaxis_title="CO (ug/m^3)")
    left_col.plotly_chart(fig_bars, use_container_width=True)
    right_col.plotly_chart(fig_co, use_column_width=True)

    pollulants = ['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'CO (ug/m3)',
                  'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)', 'SO2 (ug/m3)']
    df_avg = df_filtered[['Fecha'] + pollulants].copy()
    for col in pollulants:
        df_avg.loc[:, col] = pd.to_numeric(df_avg[col], errors='coerce')

    df_monthly_avg = df_avg.resample('ME', on='Fecha').mean()
    pollutants_graph = go.Figure()
    for p in pollulants:
        pollutants_graph.add_trace(go.Scatter(
            x=df_monthly_avg.index, y=df_monthly_avg[p], mode='lines', name=p.split(' ')[0]))
    pollutants_graph.update_layout(title='Promedio mensual de contaminantes en Miraflores',
                                   xaxis_title='Fecha', yaxis_title='Concentración (ug/m3)')

    st.plotly_chart(pollutants_graph)

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
