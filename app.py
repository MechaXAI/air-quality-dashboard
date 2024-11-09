import calendar
import base64
from folium.plugins import MarkerCluster
import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timezone
import pandas as pd
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
import locale
locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')


def initialize_session_state():
    if "location" not in st.session_state:
        st.session_state.location = None


def show_data_grid(df: pd.DataFrame):

    with st.expander("Explorar datos", expanded=False):
        st.dataframe(df.iloc[:, 5:-2], hide_index=True,
                     use_container_width=True)


def crear_mapa_de_gases(df):
    # Copia del DataFrame para evitar modificaciones no deseadas
    df_copy = df.copy()

    # Asegurarse de que las concentraciones de gases sean numéricas (en caso de datos sucios)
    gases = ['CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)',
             'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)']
    for gas in gases:
        df_copy[gas] = pd.to_numeric(df_copy[gas], errors='coerce').fillna(0)

    # No convertimos las concentraciones de gases a ppm, ya que se deben mostrar en µg/m³
    # PM10 y PM2.5 se mantienen en µg/m³

    # Crear un selectbox para que el usuario seleccione el gas a mostrar
    gas_seleccionado = st.selectbox(
        'Selecciona el gas a mostrar:',
        ['CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)',
         'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)']
    )

    # Escalar el color según el gas seleccionado (sin preocuparse por los tamaños)
    min_val = df_copy[gas_seleccionado].min()
    max_val = df_copy[gas_seleccionado].max()

    # Crear el gráfico usando scatter_mapbox con color basado en el gas seleccionado
    fig = px.scatter_mapbox(
        df_copy,
        lat='Latitud',
        lon='Longitud',
        color=gas_seleccionado,  # Usamos la concentración en µg/m³ para color
        color_continuous_scale=px.colors.cyclical.IceFire,  # Escala de colores cíclica
        # Rango de color basado en la concentración
        range_color=[min_val, max_val],
        hover_name='Distrito',  # Mostrar el nombre del distrito al pasar el ratón
        text='Distrito',         # Mostrar el nombre del distrito dentro de los círculos
        zoom=10,
        mapbox_style='carto-positron',
        title=f'Concentración de {gas_seleccionado} reportada por estación',
    )

    # Actualizar el tamaño y forma de los puntos (marcadores)
    fig.update_traces(
        marker=dict(
            size=12,  # Tamaño de los círculos
            # Color basado en la concentración de gas seleccionado
            color=df_copy[gas_seleccionado],
            colorscale=px.colors.cyclical.IceFire,
            colorbar=dict(title=f'{gas_seleccionado}'),
            opacity=0.6,  # Transparencia de los círculos
            # Forma de los puntos (puedes probar 'square', 'star', etc.)
            symbol='circle',
        ),
        textposition='top center',  # Centrado dentro de los círculos
    )

    # Personalizar el hovertemplate para mostrar el distrito y la concentración del gas seleccionado
    # Aquí, Plotly accede correctamente al valor de la columna y lo formatea correctamente
    hovertemplate = (
        '<b>%{text}</b><br>'  # Nombre del distrito
        + '%{customdata[0]} '  # Valor del gas, con dos decimales
        + 'µg/m³'  # Mantener las unidades como µg/m³
        + '<br><extra></extra>'
    )

    fig.update_traces(
        hovertemplate=hovertemplate,
        # Pasamos los valores de la columna de concentración como customdata
        customdata=df_copy[[gas_seleccionado]].values
    )

    # Mostrar el gráfico
    st.plotly_chart(fig)
    with st.expander("🤔 Ver más", expanded=False):
        st.write(
            "Se muestra en el mapa la concentración de los gases reportada por los sensores de cada estación.")


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


def grafico_acumulacion_gases(df_1):

    df_1['Fecha_Filtrado'] = pd.to_datetime(df_1['Fecha'])
    df_1['Fecha_Filtrado'] = df_1['Fecha_Filtrado'].dt.date
    df_1['Fecha_Filtrado'] = pd.to_datetime(df_1['Fecha_Filtrado'])

    st.title('Analisis de  gases en el aire')

    anios = df_1['Fecha_Filtrado'].dt.year.unique()
    meses = df_1['Fecha_Filtrado'].dt.month.unique()
    meses_nombres = {calendar.month_name[mes]: mes for mes in meses}
    anio_seleccionado = anios[-1]
    anio, mes = st.columns(2)
    with anio:
        anio_seleccionado = st.selectbox('Selecciona el Año:', anios)

    with mes:
        mes_nombre_seleccionado = st.selectbox(
            'Selecciona el Mes:',
            list(meses_nombres.keys())
        )
    mes_seleccionado = meses_nombres[mes_nombre_seleccionado]
    df_filtrado = df_1[(df_1['Fecha_Filtrado'].dt.year == anio_seleccionado) & (
        df_1['Fecha_Filtrado'].dt.month == mes_seleccionado)]

    if df_filtrado.empty:
        st.warning(
            f'No hay datos disponibles para {mes_nombre_seleccionado}/{anio_seleccionado}.')
    else:
        # Mostrar los primeros registros para ver los datos
        df_filtrado['H2S (ug/m3)'] = df_filtrado['H2S (ug/m3)'].replace("-", "0.00")
        df_filtrado['H2S (ug/m3)'] = df_filtrado['H2S (ug/m3)'].astype(float)
        df_filtrado = df_filtrado[[
            'Fecha', 'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)']]
        st.write(
            f'Datos para {mes_nombre_seleccionado} de {anio_seleccionado}:')
        st.dataframe(df_filtrado)

    if not df_filtrado.empty:
        fig = px.bar(df_filtrado,
                     x='Fecha',
                     y=['O3 (ug/m3)', 'NO2 (ug/m3)',
                        'H2S (ug/m3)', 'PM10 (ug/m3)'],
                     labels={'Fecha': 'Fecha',
                             'value': 'Concentración (ug/m3)',
                             'variable': 'Gases'},
                     title=f'Concentraciones de Gases en {mes_nombre_seleccionado} del {anio_seleccionado}')

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)
    else:
        st.write("No hay datos para el año y mes seleccionados.")
    with st.expander("🤔 Ver más", expanded=False):
        st.write(
            "Se muestra en el mapa la concentración de los gases reportada por los sensores de cada estación.")


def crear_grafico_de_barras_gases(df):
    df_copy = df.copy()

    contaminants = ['CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)',
                    'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM2.5 (ug/m3)', 'SO2 (ug/m3)']

    for contaminant in contaminants:
        df_copy[contaminant] = pd.to_numeric(
            df_copy[contaminant], errors='coerce')

    total_values = df_copy[contaminants].sum().reset_index()
    total_values.columns = ['Contaminante', 'Concentración Total (ug/m3)']

    fig = px.bar(total_values, x='Contaminante', y='Concentración Total (ug/m3)',
                 title='Concentraciones totales de contaminantes')
    st.plotly_chart(fig)


def crear_grafico_comparativo_semanal(df):

    df_copy = df.copy()

    df_copy['Fecha'] = pd.to_datetime(df_copy['Fecha'], errors='coerce')

    df_copy['YearWeek'] = df_copy['Fecha'].dt.to_period('W').dt.to_timestamp()

    df_copy = df_copy[df_copy['Distrito'].isin(
        ['Lima', 'Miraflores', 'San Isidro', 'San Miguel'])]

    df_copy['Ruido (dB)'] = pd.to_numeric(
        df_copy['Ruido (dB)'], errors='coerce')

    weekly_noise = df_copy.groupby(['YearWeek', 'Distrito'])[
        'Ruido (dB)'].mean().reset_index()

    fig = px.line(
        weekly_noise,
        x='YearWeek',
        y='Ruido (dB)',
        color='Distrito',
        title='Niveles promedio semanales de ruido por distrito',
        labels={'Distrito': 'Distrito',
                'Ruido (dB)': 'Nivel de Ruido (dB)', 'YearWeek': 'Semana'}
    )

    max_noise_level = 85  # dB
    fig.add_scatter(
        x=weekly_noise['YearWeek'].unique(),
        y=[max_noise_level] * len(weekly_noise['YearWeek'].unique()),
        mode='lines',
        name='Máximo Permitido (85 dB)',
        line=dict(dash='dash', color='red')
    )

    st.plotly_chart(fig)


def crear_grafico_uv_por_hora(df):
    df_copy = df.copy()

    df_copy['Fecha'] = pd.to_datetime(df_copy['Fecha'], errors='coerce')

    df_copy['Hora'] = df_copy['Fecha'].dt.hour

    hourly_uv = df_copy.groupby('Hora')['UV'].mean().reset_index()

    reference_uv_level = 6

    fig = px.line(
        hourly_uv,
        x='Hora',
        y='UV',
        title='Niveles de rayos UV en diferentes horas del día',
        labels={'Hora': 'Hora del Día', 'UV': 'Índice UV'}
    )

    fig.add_scatter(
        x=hourly_uv['Hora'],
        y=[reference_uv_level] * len(hourly_uv),
        mode='lines',
        name='Valor de Referencia (UV 6)',
        line=dict(dash='dash', color='red')
    )

    st.plotly_chart(fig)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def main():

    st.set_page_config(page_title="Miraflores Respira", page_icon="🌬️", layout="wide", menu_items={
        'About': "MirafloresRespira – 📊 Monitoreo de Calidad del Aire en Miraflores, Lima, Perú | Dashboard en \
        Tiempo Real 🌍. Consulta en tiempo real los niveles de contaminación en el aire de Miraflores: \
        PM2.5, PM10, CO, NO2, O3, y otros indicadores clave para la salud. Conoce cómo la calidad del \
        aire impacta tu bienestar y toma decisiones informadas para protegerte a ti y a tu familia. \
        🔍 Actualizaciones continuas y alertas para que estés siempre al tanto. Ideal para residentes, visitantes, y profesionales en salud ambiental"
    })

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://ibb.co/d70C1dn);
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Miraflores Respira 🍃")

    df = pd.read_csv('data/air_qa.csv')
    df_1 = pd.read_csv('data/air_qa.csv')
    with st.sidebar:
        st.markdown(
            """
                <style>
                .rounded-image {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 150px;
                    height: 150px;
                    border-radius: 50%; 
                    object-fit: cover;
                }
                </style>
                """,
            unsafe_allow_html=True
        )
        st.markdown(
            f'<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTIwzSGjLmVJXhHO2r8MmF4mzvpcIyWoyUyg&s" class="rounded-image">',
            unsafe_allow_html=True
        )
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
                MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    grafico_acumulacion_gases(df_1)
    st.title('Visualización de niveles de contaminantes por ubicación de sensor')

    crear_mapa_de_gases(df_filtered)
    st.title('Otros Factores')

    crear_grafico_comparativo_semanal(df_filtered)

    crear_grafico_uv_por_hora(df_filtered)

    image_url = 'https://www.sostenibilidad.com/media/560200/efectos-contaminacion-atmosferica-salud.jpg?width=900&height=1973'

    st.write(
        """
        <div style='text-align:center;'>
            <h5>⚠️ Altas concentraciones de PM2.5, PM10, CO, and O3, incrementan la contaminación del aire. ⚠️</h5>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, mid, col3 = st.columns(3)
    with mid:
        st.image(image_url, width=500)


if __name__ == "__main__":

    main()
