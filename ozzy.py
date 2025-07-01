# -*- coding: utf-8 -*-

import streamlit as st
import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT (Tema Gótico) ---
st.set_page_config(
    page_title="Ozzy - Asistente de Actium31",
    page_icon="💀", 
    layout="centered",
)

# --- CONFIGURACIÓN DEL MODELO DE GOOGLE ---
try:
    load_dotenv()
    API_KEY = os.getenv('GOOGLE_API_KEY')
    if not API_KEY:
        st.error("No se encontró la GOOGLE_API_KEY en el archivo .env. Por favor, asegúrate de que el archivo existe y la clave está configurada.")
        st.stop()
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Error al configurar la API de Google: {e}")
    st.stop()

# --- INSTRUCCIÓN MAESTRA DE OZZY (System Prompt) ---
INSTRUCCION_MAESTRA = """Eres Ozzy, el asistente de IA de Actium31, una productora de vídeo y foto para músicos. Tu personalidad es cercana, creativa y profesional. Entiendes perfectamente a los artistas porque, como Actium31, vienes del mundo de la música.

Tus reglas son:
1. Tono: Trata siempre de tú. Sé amable y colaborador. Tu objetivo es ayudar y resolver dudas.
2. Base de Conocimiento: Basa tus respuestas únicamente en la información del contexto que te proporciono. No añadas información que no esté ahí. Si la respuesta no está en el contexto, di honestamente que no tienes esa información y proporciona el email de contacto (info@actium31.com). Nunca inventes respuestas.
3. Objetivo Principal: Ayuda a los usuarios a entender los servicios, explícales cómo contactar y anímales a ver el portfolio si es relevante.
4. Estilo: Da respuestas claras y concisas. Evita los párrafos muy largos, pero mantén siempre un toque creativo y personal."""

# --- FUNCIONES DEL MOTOR RAG (CON CACHÉ DE STREAMLIT) ---

@st.cache_resource
def cargar_y_procesar_kb():
    """Carga, trocea y genera embeddings para la base de conocimiento."""
    ruta_archivo = "base_de_conocimiento_actium31.txt"
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        contenido_kb = archivo.read()
    
    fragmentos = contenido_kb.split('---')
    fragmentos_limpios = [fragmento.strip() for fragmento in fragmentos if fragmento.strip()]
    
    model_embedding = 'models/embedding-001'
    embeddings = genai.embed_content(model=model_embedding, content=fragmentos_limpios)['embedding']
    
    print("Base de conocimiento cargada y procesada exitosamente.")
    return fragmentos_limpios, embeddings

def encontrar_fragmento_mas_relevante(pregunta, fragmentos_kb, embeddings_kb):
    model_embedding = 'models/embedding-001'
    embedding_pregunta = genai.embed_content(model=model_embedding, content=pregunta)['embedding']
    puntuaciones_similitud = np.dot(np.array(embedding_pregunta), np.array(embeddings_kb).T)
    indice_mas_relevante = np.argmax(puntuaciones_similitud)
    return fragmentos_kb[indice_mas_relevante]

def generar_respuesta(pregunta, contexto):
    """Genera una respuesta conversacional usando el modelo generativo de Gemini."""
    prompt = f"""
    {INSTRUCCION_MAESTRA}
    ---CONTEXTO---
    {contexto}
    ---FIN DEL CONTEXTO---
    PREGUNTA DEL USUARIO:
    {pregunta}
    RESPUESTA DE OZZY:
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        respuesta = model.generate_content(prompt)
        return respuesta.text
    except Exception as e:
        st.error(f"Error al generar respuesta: {e}")
        return None

# --- INTERFAZ DE LA APLICACIÓN ---

st.title("Ozzy 💀 - Asistente de Actium31")
st.write("¡Hola! Soy Ozzy, el asistente de IA de Actium31. Estoy aquí para ayudarte con tus dudas sobre producción de vídeo, foto y marketing para músicos. ¿En qué puedo ayudarte?")

try:
    chunks, embeddings_kb = cargar_y_procesar_kb()
except Exception as e:
    st.error(f"No se pudo cargar la base de conocimiento. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LÍNEA CORREGIDA ---
# Ahora, al dibujar el historial, nos aseguramos de asignar el avatar correcto a cada mensaje.
for message in st.session_state.messages:
    avatar = "🤘" if message["role"] == "user" else "💀"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres saber?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🤘"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="💀"):
        with st.spinner("Ozzy está pensando..."):
            contexto = encontrar_fragmento_mas_relevante(prompt, chunks, embeddings_kb)
            respuesta = generar_respuesta(prompt, contexto)
            # Asegurarse de que la respuesta no sea None
            if respuesta is None:
                respuesta = "Lo siento, tuve un problema. Inténtalo de nuevo."
            st.markdown(respuesta)
    
    st.session_state.messages.append({"role": "assistant", "content": respuesta})