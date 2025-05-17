import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar modelos y escaladores
modelos = {
    'Regresión Logística': joblib.load('modelo_reg_log.pkl'),
    'Maquina de Apoyo Vectorial': joblib.load('modelo_svm.pkl'),
    'Red Neuronal (MLP)': joblib.load('modelo_rna.pkl'),
    'Mapas Cognitivos Difusos': joblib.load('modelo_fcm.pkl')
}

escaladores = {
    'Regresión Logística': joblib.load('escalador_reg_log.pkl'),
    'Maquina de Apoyo Vectorial': joblib.load('escalador_svm.pkl'),
    'Red Neuronal (MLP)': joblib.load('escalador_rna.pkl'),
    'Mapas Cognitivos Difusos': joblib.load('escalador_fcm.pkl')
}

# Diccionario con etiquetas y rangos sugeridos
columnas = {
    'C1': ('Edad', (18, 45)),
    'C2': ('Índice de Masa Corporal (IMC)', (15, 40)),
    'C3': ('Edad gestacional al momento del parto (semanas)', (20, 42)),
    'C4': ('Gravidez', (1, 10)),
    'C5': ('Paridad', (0, 5)),
    'C6': ('Síntomas iniciales de inicio (0: Edema, 1: Hipertensión, 2: RCIU)', (0, 2)),
    'C7': ('Edad gestacional al inicio de los síntomas iniciales (semanas)', (10, 42)),
    'C8': ('Intervalo desde el inicio de los síntomas iniciales hasta el parto (días)', (0, 100)),
    'C9': ('Edad gestacional al inicio de la hipertensión (semanas)', (10, 42)),
    'C10': ('Intervalo desde el inicio de la hipertensión hasta el parto (días)', (0, 100)),
    'C11': ('Edad gestacional al inicio del edema (semanas)', (10, 42)),
    'C12': ('Intervalo desde el inicio del edema hasta el parto (días)', (0, 100)),
    'C13': ('Edad gestacional al inicio de la proteinuria (semanas)', (10, 42)),
    'C14': ('Intervalo desde el inicio de la proteinuria hasta el parto (días)', (0, 100)),
    'C15': ('Tratamiento expectante', (0, 1)),
    'C16': ('Terapia antihipertensiva antes de la hospitalización', (0, 1)),
    'C17': ('Antecedentes (0: No, 1: Hipertensión, 2: SOP)', (0, 2)),
    'C18': ('Presión arterial sistólica máxima', (90, 200)),
    'C19': ('Presión arterial diastólica máxima', (50, 130)),
    'C20': ('Razones para el parto (0-5)', (0, 5)),
    'C21': ('Modo de parto (0: Cesárea, 1: Parto vaginal)', (0, 1)),
    'C22': ('Valor máximo de BNP', (0, 3000)),
    'C23': ('Valores máximos de creatinina', (0.2, 5.0)),
    'C24': ('Valor máximo de ácido úrico', (2, 10)),
    'C25': ('Valor máximo de proteinuria', (0, 10)),
    'C26': ('Valor máximo de proteína total', (3, 8)),
    'C27': ('Valor máximo de albúmina', (2, 5.5)),
    'C28': ('Valor máximo de ALT', (0, 300)),
    'C29': ('Valor máximo de AST', (0, 300)),
    'C30': ('Valor máximo de plaquetas', (50, 450))
}
st.title("Predicción de Crecimiento Fetal (FGR)")

modelo_seleccionado = st.selectbox("Selecciona el modelo de predicción", list(modelos.keys()))
modelo = modelos[modelo_seleccionado]
escalador = escaladores[modelo_seleccionado]

# Ajuste de columnas según el modelo
if modelo_seleccionado in ['Regresión Logística']:
    columnas_requeridas = {k: v for k, v in columnas.items() if k not in ['C20', 'C21']}
else:
    columnas_requeridas = columnas

modo = st.radio("Selecciona el modo de predicción:", ["Predicción Individual", "Predicción por Lote"])

if modo == "Predicción Individual":
    st.subheader("Ingreso de datos manual")
    entrada = []
    for cod, (desc, (min_val, max_val)) in columnas_requeridas.items():
        valor = st.number_input(f"{cod} - {desc} (Rango sugerido: {min_val}-{max_val})", min_value=float(min_val), max_value=float(max_val), value=float((min_val + max_val) // 2))
        entrada.append(valor)

    if st.button("Predecir"):
        entrada_np = np.array(entrada).reshape(1, -1)
        entrada_esc = escalador.transform(entrada_np)
        pred = modelo.predict(entrada_esc)
        st.success(f"Resultado de la predicción: {'FGR Positivo' if int(pred[0]) == 1 else 'Negativo'}")

else:
    st.subheader("Carga de archivo (.csv o .xlsx)")
    archivo = st.file_uploader("Selecciona un archivo que contenga las columnas correspondientes", type=["xlsx", "csv"])
    if archivo:
        try:
            df = pd.read_excel(archivo) if archivo.name.endswith('.xlsx') else pd.read_csv(archivo)
            columnas_esperadas = list(columnas_requeridas.keys())

            if set(columnas_esperadas).issubset(df.columns):
                X = df[columnas_esperadas]
                y_true = df['C31'] if 'C31' in df.columns else None
                X_scaled = escalador.transform(X)
                pred = modelo.predict(X_scaled)
                df['Predicción'] = np.where(pred == 1, 'FGR', 'Normal')
                st.dataframe(df[['Predicción']])

                if y_true is not None:
                    acc = accuracy_score(y_true, pred)
                    cm = confusion_matrix(y_true, pred)
                    report = classification_report(y_true, pred, target_names=['Normal', 'FGR'], output_dict=True)

                    st.subheader("Métricas del modelo")
                    st.markdown(f"**Exactitud general:** {acc:.2f}")

                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'FGR'], yticklabels=['Normal', 'FGR'], ax=ax)
                    ax.set_xlabel("Predicción")
                    ax.set_ylabel("Valor Real")
                    ax.set_title("Matriz de Confusión")
                    st.pyplot(fig)

                    tn, fp, fn, tp = cm.ravel()
                    st.markdown("**Interpretación:**")
                    st.write(f"- Verdaderos Negativos (Normal predicho como Normal): {tn}")
                    st.write(f"- Falsos Positivos (Normal predicho como FGR): {fp}")
                    st.write(f"- Falsos Negativos (FGR predicho como Normal): {fn}")
                    st.write(f"- Verdaderos Positivos (FGR predicho como FGR): {tp}")
                    st.markdown("""
                    Una buena predicción minimiza los falsos positivos (FP) y falsos negativos (FN). 
                    Los **FN** son especialmente críticos, ya que implican casos de FGR no detectados.
                    """)

                    st.subheader("Reporte de Clasificación")
                    reporte_df = pd.DataFrame(report).transpose().round(2)
                    st.dataframe(reporte_df)

                    st.markdown("""
                    - **Precisión (precision):** Proporción de verdaderos positivos sobre el total de predicciones positivas.
                    - **Sensibilidad (recall):** Proporción de verdaderos positivos sobre el total de casos reales positivos.
                    - **F1-score:** Media armónica entre precisión y recall. Útil cuando hay desbalance de clases.
                    """)

                st.download_button("Descargar resultados", df.to_csv(index=False).encode(), file_name="predicciones_fgr.csv")
            else:
                st.error("Las columnas del archivo no coinciden con las esperadas.")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
