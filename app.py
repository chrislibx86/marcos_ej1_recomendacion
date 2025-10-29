import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="**Eduardo Mendieta** - Recomendador de libros - Arquitectura Lambda", layout="wide")


# 1. SIMULACIÓN DE DATOS ---------------------------------------------------------------------

datos_libros = {
    "Cien Años de Soledad": {"categoria": "Ficción Mágica", "complejidad": 5, "popularidad": 4},
    "1984": {"categoria": "Ciencia Ficción", "complejidad": 4, "popularidad": 5},
    "El Principito": {"categoria": "Fábula", "complejidad": 1, "popularidad": 5},
    "Sapiens": {"categoria": "No Ficción", "complejidad": 3, "popularidad": 3},
    "Don Quijote de la Mancha": {"categoria": "Clásico", "complejidad": 5, "popularidad": 3},
    "Dune": {"categoria": "Ciencia Ficción", "complejidad": 4, "popularidad": 4},
    "Orgullo y Prejuicio": {"categoria": "Romance Clásico", "complejidad": 2, "popularidad": 4},
    "El Hobbit": {"categoria": "Fantasía", "complejidad": 3, "popularidad": 5},
    "Crimen y Castigo": {"categoria": "Novela Filosófica", "complejidad": 5, "popularidad": 3},
    "Cumbres Borrascosas": {"categoria": "Romance Gótico", "complejidad": 3, "popularidad": 4},
    "Moby Dick": {"categoria": "Aventura", "complejidad": 4, "popularidad": 2},
    "Matar a un Ruiseñor": {"categoria": "Drama Social", "complejidad": 2, "popularidad": 5},
    "La Carretera": {"categoria": "Post-apocalíptico", "complejidad": 3, "popularidad": 3},
    "Rayuela": {"categoria": "Experimental", "complejidad": 5, "popularidad": 2},
    "El Nombre del Viento": {"categoria": "Fantasía Épica", "complejidad": 4, "popularidad": 4},
    "Una Breve Historia del Tiempo": {"categoria": "Divulgación Científica", "complejidad": 4, "popularidad": 5},
    "Ensayo sobre la Ceguera": {"categoria": "Novela Distópica", "complejidad": 3, "popularidad": 4},
    " Siddhartha": {"categoria": "Filosofía", "complejidad": 2, "popularidad": 3},
    "Drácula": {"categoria": "Terror Clásico", "complejidad": 3, "popularidad": 4},
    "El Padrino": {"categoria": "Crimen", "complejidad": 2, "popularidad": 5},
    "Psicoanalista": {"categoria": "Thriller", "complejidad": 3, "popularidad": 3},
    "Fahrenheit 451": {"categoria": "Ciencia Ficción", "complejidad": 2, "popularidad": 4},
    "La Odisea": {"categoria": "Épica Antigua", "complejidad": 5, "popularidad": 2},
    "El Código Da Vinci": {"categoria": "Misterio", "complejidad": 1, "popularidad": 5},
    "Los Miserables": {"categoria": "Novela Histórica", "complejidad": 5, "popularidad": 3},
    "Las Uvas de la Ira": {"categoria": "Drama Social", "complejidad": 4, "popularidad": 2},
    "Harry Potter 1": {"categoria": "Fantasía Juvenil", "complejidad": 1, "popularidad": 5},
}

nombres_libros = list(datos_libros.keys())
usuarios = ["Eduardo", "Alexander", "Jimmy", "Karina", "Freddy", "Nube", "Anthony", "Ariel", "Joseph"]
columnas_caracteristicas = ["Complejidad de Lectura", "Popularidad Global"]


def calcular_similitud(df_calificaciones, df_caracteristicas_libro):

    calificaciones_transpuestas = df_calificaciones.T
    
    libros_comunes = list(set(calificaciones_transpuestas.index) & set(df_caracteristicas_libro.index))
    
    calificaciones_alineadas = calificaciones_transpuestas.loc[libros_comunes]
    caracteristicas_alineadas = df_caracteristicas_libro.loc[libros_comunes]
    
    caracteristicas_combinadas = pd.concat([calificaciones_alineadas, caracteristicas_alineadas], axis=1)
    
    matriz_similitud = pd.DataFrame(
        cosine_similarity(caracteristicas_combinadas),
        index=caracteristicas_combinadas.index,
        columns=caracteristicas_combinadas.index
    )
    return matriz_similitud

if 'calificaciones' not in st.session_state:
    st.session_state.calificaciones = pd.DataFrame(
        np.random.randint(1, 6, size=(len(usuarios), len(nombres_libros))),
        index=usuarios,
        columns=nombres_libros
    )
    
    lista_caracteristicas = []
    for libro, datos in datos_libros.items():
        lista_caracteristicas.append({
            "Libro": libro,
            "Complejidad de Lectura": datos["complejidad"],
            "Popularidad Global": datos["popularidad"],
        })
    st.session_state.caracteristicas_libro = pd.DataFrame(lista_caracteristicas).set_index("Libro")
    
    st.session_state.similitud = calcular_similitud(
        st.session_state.calificaciones, 
        st.session_state.caracteristicas_libro
    )


# 2. CAPA BATCH ------------------------------------------------------------------------------

st.title("📚 Recomendador de Libros con Arquitectura Lambda")
st.caption("Simulación educativa: Capa Batch + Capa de Velocidad + Capa de Servicio")

st.subheader("🧩 Capa Batch - Datos Base (Lentos)")
st.write("La capa batch procesa los datos históricos (calificaciones) y las características estáticas de los libros para construir el modelo de similitud base.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Calificaciones Históricas (Dinámico)**")
    st.dataframe(st.session_state.calificaciones)

with col2:
    st.markdown("**Características Estáticas de Libros (Lento)**")
    st.write("Estas variables se integran al cálculo de similitud. Son datos de lento cambio.")
    st.dataframe(st.session_state.caracteristicas_libro)


st.subheader("🤖 Modelo Batch - Similitud Combinada entre Libros")
st.write(
    "Calculada mediante Similitud Coseno sobre un vector que incluye las calificaciones de todos los usuarios "
    "**MÁS** la 'Complejidad de Lectura' y la 'Popularidad Global' del libro."
)
st.dataframe(st.session_state.similitud.round(3), use_container_width=True)


# 3. CAPA DE VELOCIDAD -----------------------------------------------------------------------

st.header("⚡ Capa de Velocidad - Ingreso en Tiempo Real")

usuario_velocidad = st.selectbox("Selecciona un usuario para calificar:", usuarios)
libro_velocidad = st.selectbox("Selecciona un libro para calificar:", nombres_libros)
calificacion_velocidad = st.slider("Nueva calificación (1-5)", 1, 5, 3, key="new_rating")

def actualizar_modelo():

    st.session_state.calificaciones.loc[usuario_velocidad, libro_velocidad] = calificacion_velocidad
    
    st.session_state.similitud = calcular_similitud(
        st.session_state.calificaciones, 
        st.session_state.caracteristicas_libro
    )

if st.button("➕ Agregar Nueva Calificación (Actualización Instantánea)"):
    actualizar_modelo()
    st.success(f"✅ ¡Éxito! {usuario_velocidad} ha calificado '{libro_velocidad}' con {calificacion_velocidad} estrellas. El modelo se ha actualizado.")


# 4. CAPA DE SERVICIO ------------------------------------------------------------------------

st.header("🛰️ Capa de Servicio - Recomendaciones Actualizadas")

libro_seleccionado = st.selectbox("Selecciona un libro base para encontrar similares:", nombres_libros)

recomendaciones = st.session_state.similitud[libro_seleccionado].sort_values(ascending=False)[1:5]

st.write("🔍 Libros más similares a:", libro_seleccionado)

df_recomendaciones = pd.DataFrame(recomendaciones)
df_recomendaciones.columns = ["Similitud Combinada"]
st.dataframe(df_recomendaciones.round(4), use_container_width=True)

st.info("**Estudiante:** Eduardo Mendieta.")