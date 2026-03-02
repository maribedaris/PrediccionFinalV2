import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# CONFIGURACIÓN
# ==============================

st.set_page_config(page_title="Clasificador - Red Neuronal", layout="centered")

# ==============================
# CARGAR ARTEFACTOS
# ==============================

@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("modelo_red_neuronal.joblib")
    scaler = joblib.load("scaler.joblib")
    cols_escalar = joblib.load("columnas_escalar.joblib")
    cols_categoricas = joblib.load("columnas_categoricas.joblib")
    le_binarios = joblib.load("label_encoders_binarios.joblib")
    ohe = joblib.load("one_hot_encoder.joblib")
    feature_columns = joblib.load("feature_columns.joblib")

    return modelo, scaler, cols_escalar, cols_categoricas, le_binarios, ohe, feature_columns


try:
    modelo, scaler, cols_escalar, cols_categoricas, le_binarios, ohe, feature_columns = cargar_artefactos()
except Exception as e:
    st.error(f"Error cargando artefactos: {e}")
    st.stop()

# ==============================
# INTERFAZ
# ==============================

st.title("Clasificador - Red Neuronal")
st.markdown("Suba un archivo Excel con las variables del modelo.")

archivo = st.file_uploader("Subir archivo Excel (.xlsx)", type=["xlsx"])

if archivo is not None:

    try:
        df_original = pd.read_excel(archivo)
        df_input = df_original.copy()

        st.subheader("Vista previa")
        st.dataframe(df_input.head())

        # ==============================
        # VALIDAR COLUMNAS
        # ==============================

        columnas_necesarias = (
            cols_escalar
            + cols_categoricas
            + list(le_binarios.keys())
        )

        faltantes = [col for col in columnas_necesarias if col not in df_input.columns]

        if faltantes:
            st.error(f"Faltan estas columnas en el archivo: {faltantes}")
            st.stop()

        # ==============================
        # TRANSFORMACIONES
        # ==============================

        # 1️⃣ Escalar numéricas
        df_input[cols_escalar] = scaler.transform(df_input[cols_escalar])

        # 2️⃣ Codificar binarias
        for col, le in le_binarios.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col])

        # 3️⃣ One Hot Encoding
        df_cat = df_input[cols_categoricas]
        cat_encoded = ohe.transform(df_cat)

        if hasattr(cat_encoded, "toarray"):
            cat_encoded = cat_encoded.toarray()

        try:
            ohe_col_names = ohe.get_feature_names_out(cols_categoricas)
        except:
            ohe_col_names = ohe.get_feature_names(cols_categoricas)

        df_cat_encoded = pd.DataFrame(cat_encoded, columns=ohe_col_names)

        # 4️⃣ Eliminar categóricas originales
        df_input = df_input.drop(columns=cols_categoricas)

        # 5️⃣ Unir todo
        df_final = pd.concat(
            [df_input.reset_index(drop=True),
             df_cat_encoded.reset_index(drop=True)],
            axis=1
        )

        # 6️⃣ Orden correcto de columnas
        df_final = df_final.reindex(columns=feature_columns, fill_value=0)

        # ==============================
        # PREDICCIÓN
        # ==============================

        predicciones = modelo.predict(df_final)
        probabilidades = modelo.predict_proba(df_final)[:, 1]

        # ==============================
        # RESULTADOS
        # ==============================

        df_resultado = df_original.copy()
        df_resultado["Prediccion"] = predicciones
        df_resultado["Probabilidad_Ingreso"] = probabilidades

        st.subheader("Resultados")
        st.dataframe(df_resultado.head())

        # Descargar resultados
        archivo_salida = "resultado_predicciones.xlsx"
        df_resultado.to_excel(archivo_salida, index=False)

        with open(archivo_salida, "rb") as f:
            st.download_button(
                label="Descargar resultados",
                data=f,
                file_name="resultado_predicciones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error procesando archivo: {e}")