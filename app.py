from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el pipeline unificado
model_pipeline = joblib.load('pipeline_estudiantes.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error_message = None
    
    # Lista de variables requeridas para validación
    fields = [
        'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
        'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
        'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]

    if request.method == 'POST':
        try:
            # 1. Recolectar datos del formulario
            input_data = {}
            for field in fields:
                val = request.form.get(field)
                if val is None or val.strip() == "":
                    raise ValueError(f"El campo '{field}' es obligatorio.")
                input_data[field] = [val.strip()]
            
            # 2. Convertir tipos de datos numéricos explícitamente
            df_input = pd.DataFrame(input_data)
            num_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
            for col in num_cols:
                df_input[col] = pd.to_numeric(df_input[col])
                
            # Validaciones lógicas básicas de negocio
            if not (0 <= df_input['Attendance'].iloc[0] <= 100):
                raise ValueError("La asistencia debe estar entre 0% y 100%.")
            if not (0 <= df_input['Previous_Scores'].iloc[0] <= 100):
                raise ValueError("Las calificaciones previas deben estar entre 0 y 100.")

            # 3. Generar predicción pasando el DataFrame limpio por el pipeline
            pred = model_pipeline.predict(df_input)[0]
            # Limitar el resultado en un rango lógico de calificaciones (0 - 100)
            prediction = round(float(np.clip(pred, 0, 100)), 2)

        except Exception as e:
            error_message = f"Error en los datos ingresados: {str(e)}"

    return render_template('index.html', prediction=prediction, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)