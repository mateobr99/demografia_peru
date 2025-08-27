import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os

RUTA_BASE = r"D:\Mateo\ICSI\Proyecto\Data\Raw\Poblacion\Data\Processing"

# Función para cargar datos con rutas absolutas
@st.cache_data
def load_data():
    # Construir rutas absolutas
    ruta_ubigeo = os.path.join(RUTA_BASE, 'tabla_ubigeo.csv')
    ruta_1993 = os.path.join(RUTA_BASE, 'pob_censo_edades_sexo_1993.csv')
    ruta_2005 = os.path.join(RUTA_BASE, 'pob_censo_edades_sexo_2005.csv')
    ruta_2017 = os.path.join(RUTA_BASE, 'pob_censo_edades_sexo_2017.csv')
    ruta_area_2005 = os.path.join(RUTA_BASE, 'pob_censo_area_2005.csv')
    ruta_area_2017 = os.path.join(RUTA_BASE, 'pob_censo_area_2017.csv')

    # Verificar que los archivos existan
    for ruta, nombre in zip([ruta_ubigeo, ruta_1993, ruta_2005, ruta_2017, ruta_area_2005, ruta_area_2017], 
                           ['tabla_ubigeo.csv', '1993.csv', '2005.csv', '2017.csv', 'area_2005.csv', 'area_2017.csv']):
        if not os.path.exists(ruta):
            st.error(f"No se encuentra: {nombre} en {ruta}")
            return None, None, None, None, None, None  # Retorna None si falta algún archivo
    
    # Cargar datos
    try:
        df_ubigeo = pd.read_csv(ruta_ubigeo, delimiter=',', encoding='latin1')
        df_censo_1993 = pd.read_csv(ruta_1993, delimiter=',', encoding='latin1')
        df_censo_2005 = pd.read_csv(ruta_2005, delimiter=',', encoding='latin1')
        df_censo_2017 = pd.read_csv(ruta_2017, delimiter=',', encoding='latin1')
        df_censo_area_2005 = pd.read_csv(ruta_area_2005, delimiter=',', encoding='latin1')
        df_censo_area_2017 = pd.read_csv(ruta_area_2017, delimiter=',', encoding='latin1')
        
        st.success("Archivos cargados correctamente")
        
        # CORRECCIÓN: Retornar las variables correctas y dentro del bloque try
        return df_ubigeo, df_censo_1993, df_censo_2005, df_censo_2017, df_censo_area_2005, df_censo_area_2017
        
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}")
        st.stop()
        return None, None, None, None, None, None  # Retorna None si hay error
    

    # Función para filtrar datos duplicados (totales)
def filtrar_datos_duplicados(df):
        """
        Filtra categorías duplicadas o de resumen de un DataFrame
        si las columnas correspondientes existen.
        """
        categorias_excluir = ['Total', 'total', 'TOTAL', 'Ambos sexos', 'Ambos Sexos']

        # Filtra por 'sexo' y 'grupo_edad' si las columnas existen
        if 'sexo' in df.columns:
            df = df[~df['sexo'].isin(categorias_excluir)]
        if 'grupo_edad' in df.columns:
            df = df[~df['grupo_edad'].isin(categorias_excluir)]
            
        if 'tipo_area' in df.columns:
            df = df[~df['tipo_area'].isin(categorias_excluir)]
            
        return df

# Función para optimizar datos
def optimize_dataframe(df):
    if 'Ubigeo' in df.columns:
        df['Ubigeo'] = df['Ubigeo'].astype('string')
    if 'grupo_edad' in df.columns:
        df['grupo_edad'] = df['grupo_edad'].astype('category')
    if 'sexo' in df.columns:
        df['sexo'] = df['sexo'].astype('category')
    if 'tipo_area' in df.columns:
        df['tipo_area'] = df['tipo_area'].astype('category')
    if 'valor' in df.columns:
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    if 'poblacion' in df.columns:
        df['poblacion'] = pd.to_numeric(df['poblacion'], errors='coerce')
    return df

@st.cache_data
def setup_data(df_ubigeo, df_1993, df_2005, df_2017, df_area_2005, df_area_2017):
    """
    Esta función configura las estructuras
    de datos para acceso eficiente
    """

    # Diccionario principal de datos
    data_store = {
        'ubigeo': df_ubigeo,
        'censos': {
            1993: df_1993,
            2005: df_2005,
            2017: df_2017
        },
        'areas': {
            2005: df_area_2005,
            2017: df_area_2017
        }

    }

    departamentos = sorted(df_ubigeo['Departamento'].unique().tolist())
    provincias = sorted(df_ubigeo['Provincia'].unique().tolist())
    distritos = sorted(df_ubigeo['Distrito'].unique().tolist())

    data_store['filtros'] = {
        'departamentos': ['Todos'] + departamentos,
        'provincias': ['Todos'] + provincias,
        'distritos': ['Todos'] + distritos
    }

    return data_store

# Función para obtener los datos filtrados
def datos_filtrados(data_store, anio, departamento='Todos', provincia='Todos', distrito='Todos'):
    """
    Esta función obtiene datos filtrados sin hacer
    el merge completo
    """

    df_censo = data_store['censos'][anio].copy()
    df_censo = filtrar_datos_duplicados(df_censo)
    df_areas = data_store['areas'].get(anio)
    if df_areas is not None:
        df_areas = df_areas.copy()
    df_ubigeo = data_store['ubigeo'].copy()

    # Si no hay filtros devuelve datos del agregado nacional
    if departamento == 'Todos' and provincia == "Todos" and distrito == "Todos":
        resultado_piramide = df_censo.groupby(['grupo_edad','sexo'])['valor'].sum().reset_index()

        if df_areas is not None:
            resultado_areas = df_areas.groupby('tipo_area')['poblacion'].sum().reset_index()
        else:
            resultado_areas = pd.DataFrame()
        return{'piramide': resultado_piramide, 'area': resultado_areas}
    
    # Aplicar primer filtro - ubigeo
    ubigeo_filtrado = df_ubigeo.copy()

    if departamento != 'Todos':
        ubigeo_filtrado = ubigeo_filtrado[ubigeo_filtrado['Departamento'] == departamento]
    
    if provincia != 'Todos':
        ubigeo_filtrado = ubigeo_filtrado[ubigeo_filtrado['Provincia'] == provincia]

    if distrito != 'Todos':
        ubigeo_filtrado = ubigeo_filtrado[ubigeo_filtrado['Distrito'] == distrito]

    # Merge solo con los ubigeos filtrados
    df_filtrado = pd.merge(
        df_censo,
        ubigeo_filtrado[['Ubigeo']],
        on='Ubigeo',
        how='inner'
    )

    # CAMBIO CLAVE: Estandarizar el nombre de la columna en df_areas antes del merge
    if df_areas is not None:
        # Renombrar la columna de ubigeo si es necesario
        if 'Ubigeo' not in df_areas.columns:
            # Encuentra la columna que contiene un nombre de ubigeo y la renombra
            ubigeo_col = next((col for col in df_areas.columns if 'ubigeo' in col.lower()), None)
            if ubigeo_col:
                df_areas.rename(columns={ubigeo_col: 'Ubigeo'}, inplace=True)
            else:
                st.error("Error: No se encontró la columna 'Ubigeo' en el archivo de área para el censo. Asegúrate de que existe y está bien escrita.")
                return {'piramide': pd.DataFrame(), 'area': pd.DataFrame()}

    if df_areas is not None:
        df_areas_filtrado = pd.merge(
            df_areas,
            ubigeo_filtrado[['Ubigeo']],
            on='Ubigeo',
            how='inner'
        )

        resultado_areas = df_areas_filtrado.groupby('tipo_area')['poblacion'].sum().reset_index()
    else:
        resultado_areas = pd.DataFrame()


    # Agrupar por grupo_edad y sexo
    resultado_piramide = df_filtrado.groupby(['grupo_edad', 'sexo'])['valor'].sum().reset_index()

    return {'piramide': resultado_piramide, 'area': resultado_areas}

def crear_piramide_poblacional(df, anio, title_suffix=""):
    """
    Crea una pirámide poblacional a partir de datos grupados 
    """
    # 1. FILTRAR: Excluir la categoría "Total" si existe
    df = df[df['grupo_edad'] != 'Total']
    
    # 2. DEFINIR ORDEN de grupos de edad
    orden_edades = [
        'De 0 a 4 años', 'De 5 a 9 años', 'De 10 a 14 años', 'De 15 a 19 años',
        'De 20 a 24 años', 'De 25 a 29 años', 'De 30 a 34 años', 'De 35 a 39 años',
        'De 40 a 44 años', 'De 45 a 49 años', 'De 50 a 54 años', 'De 55 a 59 años',
        'De 60 a 64 años', 'De 65 a 69 años', 'De 70 a 74 años', 'De 75 a 79 años',
        'De 80 a 84 años', 'De 85 a 89 años', 'De 90 a 94 años', 'De 95 a 99 años',
        'De 100 y más años'
    ]

    # 3. Filtrar solo los grupos que existen en los datos
    grupos_existentes = [grupo for grupo in orden_edades if grupo in df['grupo_edad'].unique()]

    # 4. ORDENAR los datos según el orden definido
    df['grupo_edad_ordenado'] = pd.Categorical(
        df['grupo_edad'], 
        categories=grupos_existentes, 
        ordered=True
    )
    df = df.sort_values('grupo_edad_ordenado')

    # Separar por sexo
    hombres = df[df['sexo'] == 'Hombre']
    mujeres = df[df['sexo'] == 'Mujer']

    # Verificar que tengamos datos
    if hombres.empty or mujeres.empty:
        st.warning("No hay datos suficientes para crear la pirámide")
        return None
    
    # Crear gráfico 
    fig, ax = plt.subplots(figsize=(10, 8))

    # Posiciones en el eje y 
    y_pos = np.arange(len(hombres))

    # Barras horizontales
    ax.barh(y_pos, -hombres['valor'], height=0.8, alpha=0.7,
            color='#3498bd', edgecolor='white')
    ax.barh(y_pos, mujeres['valor'], height=0.8, alpha=0.7,
            color='#e74c3c', edgecolor='white')
    
    # Configurar ejes y etiquetas
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(edad) for edad in hombres['grupo_edad']], fontsize=10)
    ax.set_xlabel('Población', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grupo de Edad', fontsize=12, fontweight='bold')
    
    # Formatear eje X para valores absolutos
    max_val = max(abs(ax.get_xlim()[0]), ax.get_xlim()[1])
    ax.set_xlim(-max_val * 1.1, max_val * 1.1)
    ax.set_xticks(np.arange(-max_val, max_val + 1, max_val // 5))
    ax.set_xticklabels([str(int(abs(x))) for x in ax.get_xticks()])
    
    # Título y leyenda
    title = f'Pirámide Poblacional {anio}'
    if title_suffix:
        title += f' - {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    
    # Grid suave
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def crear_grafico_pastel(df, anio, title_suffix, columna_area='tipo_area'):
    """
    Args:
        df: DataFrame con los datos
        anio: Año del censo
        title_suffix: Sufijo para el título
        columna_area: Nombre de la columna que contiene el tipo de área
    """
    # Verificar si el DataFrame está vacío
    if df is None or df.empty:
        st.warning(f"DataFrame vacío para el año {anio}")
        return None
    
    # Verificar si la columna existe
    if columna_area not in df.columns:
        st.error(f"La columna '{columna_area}' no existe. Columnas disponibles: {list(df.columns)}")
        return None
    
    # Verificar si existe la columna 'poblacion'
    if 'poblacion' not in df.columns:
        st.error(f"La columna 'poblacion' no existe. Columnas disponibles: {list(df.columns)}")
        return None
    
    # Filtrar excluyendo 'total'
    df_filtrado = df[df[columna_area].str.lower() != 'total']
    
    # Verificar si hay datos después del filtro
    if df_filtrado.empty:
        st.warning(f"No hay datos de áreas específicas para el año {anio}")
        return None
    
    categorias_ordenadas = ['Urbano', 'Rural']
    grupos_existentes = [cat for cat in categorias_ordenadas if cat in df_filtrado[columna_area].unique()]
    
    # Verificar que hay categorías para mostrar
    if not grupos_existentes:
        st.warning(f"No hay categorías válidas de área para el año {anio}")
        return None
    
    # Crear una copia para no modificar el original
    df_plot = df_filtrado.copy()
    df_plot[columna_area] = pd.Categorical(df_plot[columna_area], categories=grupos_existentes, ordered=True)
    df_plot = df_plot.sort_values(columna_area)
    
    try:
        fig = px.pie(
            df_plot,
            names=columna_area,
            values='poblacion',
            title=f'Población por Tipo de Área {anio} - {title_suffix}',
            hole=0.4
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        
        return fig
        
    except Exception as e:
        st.error(f"Error al crear gráfico de pastel: {e}")
        return None

# Aplicación Streamlit principal
def main():
    st.set_page_config(page_title='Pirámides poblacionales Perú', layout='wide')
    st.title('Pirámides poblacionales del Perú')
    st.markdown("Visualización de datos censales 1993, 2005 y 2017")

    # Cargar datos
    df_ubigeo, df_1993, df_2005, df_2017, df_area_2005, df_area_2017 = load_data()
    data_store = setup_data(df_ubigeo, df_1993, df_2005, df_2017, df_area_2005, df_area_2017)

    # Sidebar con controles
    with st.sidebar:
        st.header("Filtros")

        # Selector del año
        anio = st.selectbox(
            'Seleccione el año del censo:',
            options=[1993, 2005, 2017],
            index=2
        )

        # Selector del departamento
        departamento = st.selectbox(
            'Departamento:',
            options=data_store['filtros']['departamentos'],
            index=0
        )

        # Selector de provincia
        provincias_filtradas = ['Todos']
        if departamento != 'Todos':
            provincias_departamento = data_store['ubigeo'][
                data_store['ubigeo']['Departamento'] == departamento
            ]['Provincia'].unique().tolist()
            provincias_filtradas += sorted(provincias_departamento)
        
        provincia = st.selectbox(
            "Provincia:",
            options=provincias_filtradas,
            index=0
        )

        # Selector de distrito (dependiente de departamento y provincia)
        distritos_filtrados = ['Todos']
        if departamento != 'Todos' and provincia != 'Todos':
            distritos_provincia = data_store['ubigeo'][
                (data_store['ubigeo']['Departamento'] == departamento) &
                (data_store['ubigeo']['Provincia'] == provincia)
            ]['Distrito'].unique().tolist()
            distritos_filtrados += sorted(distritos_provincia)
        
        distrito = st.selectbox(
            "Distrito:",
            options=distritos_filtrados,
            index=0
        )

    with st.spinner('Cargando datos...'):
        resultados_filtrados = datos_filtrados(data_store, anio, departamento,provincia, distrito)
        df_piramide = resultados_filtrados['piramide']
        df_area = resultados_filtrados['area']

    # Título descriptivo
    title_suffix = "Total Nacional"
    if departamento != "Todos":
        title_suffix = f"Departamento: {departamento}"
    if provincia != "Todos":
        title_suffix += f", Provincia: {provincia}"
    if distrito != 'Todos':
        title_suffix += f", Distrito: {distrito}"

    col1, col2 = st.columns(2)

    with col1:
        if not df_piramide.empty:
            fig = crear_piramide_poblacional(df_piramide, anio, title_suffix)
            st.pyplot(fig)
        else:
            st.warning("No hay datos de pirámide disponibles para los filtros seleccionados.")

    with col2:
        if not df_area.empty:
            fig_pastel = crear_grafico_pastel(df_area, anio, title_suffix)
            st.plotly_chart(fig_pastel)
        else:
            st.warning("No hay datos de área disponibles para los filtros seleccionados.")
            
    if not df_piramide.empty:
        total_poblacion_general = df_piramide[df_piramide['sexo'].isin(['Hombre', 'Mujer'])]['valor'].sum()
        total_hombres = df_piramide[df_piramide['sexo'] == 'Hombre']['valor'].sum()
        total_mujeres = df_piramide[df_piramide['sexo'] == 'Mujer']['valor'].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Población", f"{total_poblacion_general:,.0f}")
        with col2:
            st.metric("Hombres", f"{total_hombres:,.0f}")
        with col3:
            st.metric("Mujeres", f"{total_mujeres:,.0f}")
    else:
        st.warning("No hay datos disponibles para los filtros seleccionados")

if __name__ == '__main__':
    main()