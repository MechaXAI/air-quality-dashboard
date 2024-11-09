import streamlit as st
from datetime import datetime, timezone
import pandas as pd
from dateutil.relativedelta import relativedelta
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from folium.plugins import MarkerCluster

def initialize_session_state():
    if "location" not in st.session_state:
        st.session_state.location = None

def show_data_grid(df: pd.DataFrame):

    with st.expander("Explorar datos", expanded=True):
        st.dataframe(df.iloc[:, 5:], hide_index=True, use_container_width=True)


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
    initialize_session_state()
    st.title("Miraflores Respira 🍃")

    df = pd.read_csv('C:/Users/Pam/Desktop/Pam/Diplomado_Aplicaciones_IA/MODULO 1/Proyectos/air-quality-dashboard/data/air_qa.csv')
    df_1 = pd.read_csv('C:/Users/Pam/Desktop/Pam/Diplomado_Aplicaciones_IA/MODULO 1/Proyectos/air-quality-dashboard/data/air_qa.csv')
    with st.sidebar:
        st.header("Filtros: ")
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df['Year'] = df['Fecha'].dt.year
        year_filter = st.multiselect(
            "Año: ", options=df["Year"].unique(), default=df["Year"].unique())
        temp_min, temp_max = st.slider(
            "Temperature °C:", min_value=df["Temperatura (C)"].min(), max_value=df["Temperatura (C)"].max(), value=(df["Temperatura (C)"].min(), df["Temperatura (C)"].max()))
        
        
        
    df_filtered = df[df["Year"].isin(year_filter) & (
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
    
    df['YearMonth'] = df["Fecha"].dt.to_period('M').dt.to_timestamp()
    
    monthly_pm25 = df.groupby('YearMonth')['PM2.5 (ug/m3)'].mean().reset_index()
    fig = px.line(monthly_pm25, x='YearMonth', y='PM2.5 (ug/m3)', title='Miraflores - Calidad del aire PM2.5 (ug/m3) Promedio Mensual', labels={'YearMonth': 'Fecha', 'PM2.5 (ug/m3)': 'PM2.5 (ug/m3)'})
    st.plotly_chart(fig)
    
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
  
    st.title('PruebaPam')
    
    
    
    
    
    
    df_map = df_1[['Latitud', 'Longitud']]
    df_map.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'}, inplace=True)
    df_map[['lat', 'lon']] = df_map[['lat', 'lon']].round(4)
    #print(df_map)
    #st.map(df_map)

    
    ubicaciones_unicas = df_map.drop_duplicates(subset=['lat', 'lon'])
    distritos = ['San Isidro',  'San Miguel', 'Miraflores', 'Lima']
    print('ubi')
    print(ubicaciones_unicas)
    ubicaciones_unicas['distrito']=distritos
    
    
    st.map(ubicaciones_unicas)
    """
    st.title('Mapa Geográfico Interactivo')
    
    distritos_unicos = ubicaciones_unicas['distrito'].unique()
    distrito_seleccionado = st.selectbox("Selecciona un distrito para filtrar:", ['Todos'] + list(distritos_unicos))
    
    if distrito_seleccionado != 'Todos':
        df = ubicaciones_unicas[ubicaciones_unicas['distrito'] == distrito_seleccionado]
    
    mapa = folium.Map(location=[ubicaciones_unicas['lat'].mean(), ubicaciones_unicas['lon'].mean()], zoom_start=12)
    
    marker_cluster = MarkerCluster().add_to(mapa)
    
    for _, row in ubicaciones_unicas.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['distrito'],
            tooltip=row['distrito']
        ).add_to(marker_cluster)
    
    st.markdown(mapa._repr_html_(), unsafe_allow_html=True)
    """
    # Crear un DataFrame con ubicaciones, latitudes, longitudes y etiquetas
    
    
    # Crear un mapa interactivo con Plotly
    fig = px.scatter_mapbox(ubicaciones_unicas, lat="lat", lon="lon", hover_name="distrito",
                            color="distrito", size_max=30, zoom=10)
    
    # Usar una plantilla de mapa de Plotly
    fig.update_layout(mapbox_style="carto-positron")
    
    # Mostrar el mapa en Streamlit
    st.plotly_chart(fig)

    #Filtro con los gases
    df_1['Fecha_Filtrado']= pd.to_datetime(df_1['Fecha'])
    df_1['Fecha_Filtrado'] = df_1['Fecha_Filtrado'].dt.date
    df_1['Fecha_Filtrado']= pd.to_datetime(df_1['Fecha_Filtrado'])


    st.title('Analisis de  gases en el aire')
    
    anios = df_1['Fecha_Filtrado'].dt.year.unique()
    meses = df_1['Fecha_Filtrado'].dt.month.unique()
    
    
    anio, mes = st.columns(2)
    with anio:
        anio_seleccionado = st.selectbox('Selecciona el Año:', anios)
    with mes:
        mes_seleccionado = st.selectbox('Selecciona el Mes:', sorted(meses))
    
    df_filtrado = df_1[(df_1['Fecha_Filtrado'].dt.year == anio_seleccionado) & (df_1['Fecha_Filtrado'].dt.month == mes_seleccionado)]
    
    if df_filtrado.empty:
        st.warning(f'No hay datos disponibles para {mes_seleccionado}/{anio_seleccionado}.')
    else:
        # Mostrar los primeros registros para ver los datos
        df_filtrado['H2S (ug/m3)'] = df_filtrado['H2S (ug/m3)'].replace("-","0.00")
        df_filtrado['H2S (ug/m3)'] = df_filtrado['H2S (ug/m3)'].astype(float)
        df_filtrado=df_filtrado[['Fecha', 'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)']]
        print(df_filtrado)
        st.write(f'Datos para {mes_seleccionado}/{anio_seleccionado}:')
        st.dataframe(df_filtrado)
    
   
    # Si el DataFrame filtrado no está vacío, generar el gráfico
    if not df_filtrado.empty:
        # Crear un gráfico con Plotly Express
        fig = px.bar(df_filtrado, 
                      x='Fecha', 
                      y=['O3 (ug/m3)', 'NO2 (ug/m3)', 'H2S (ug/m3)', 'PM10 (ug/m3)'],
                      labels={'Fecha': 'Fecha', 
                              'value': 'Concentración (ug/m3)', 
                              'variable': 'Gaseoso'},
                      title=f'Concentraciones de Gases en {anio_seleccionado}-{mes_seleccionado}')
        
        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)
    else:
        st.write("No hay datos para el año y mes seleccionados.")
        
if __name__ == "__main__":
    main()
