import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Modelo CAPM", layout="wide")

# Estilos personalizados con fondo negro para el resultado
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #e6f0ff; /* azul claro */
    }
    .stButton>button {
        background-color: #ff7f00; /* naranja intenso */
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
    .stMetric {
        background-color: #000000; /* negro */
        color: #ffffff; /* blanco */
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #4caf50 !important; /* verde fuerte */
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Modelo CAPM Interactivo")

# Inputs en el marco lateral
st.sidebar.header("🔧 Parámetros de entrada")
rf = st.sidebar.number_input("Tasa libre de riesgo (%)", value=2.0, step=0.1)
beta = st.sidebar.number_input("Beta", value=1.0, step=0.1)
rm = st.sidebar.number_input("Rendimiento de mercado (%)", value=8.0, step=0.1)

# Botón para correr el cálculo
if st.sidebar.button("Calcular CAPM"):
    capm = rf + beta * (rm - rf)
    st.subheader("📈 Resultado del CAPM")
    st.metric(label="Rendimiento esperado (%)", value=round(capm, 2))
    st.success("✅ Cálculo completado")
else:
    st.info("Introduce los parámetros y presiona **Calcular CAPM**")
