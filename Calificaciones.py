import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificaciones  ''')
st.image("calificaciones.jpeg", caption="Predicción de la calificación de una persona.")

st.header('Datos de estudiante')

def user_input_features():
  # Entrada
  horas = st.number_input('Horas:', min_value=1, max_value=100, value = 1, step = 1)
  horas_sueño = st.number_input("Horas_sueño:",  min_value=4, max_value=9, value = 4, step = 1)
  asistencia = st.number_input('porcentaje de asistencia:', min_value=50, max_value=100, value = 50, step = 1)
  calificaciones_previas = st.number_input('Calificaciones previas:', min_value=40, max_value=95, value = 40, step = 1)


  user_input_data = {'hours_studied': horas,
                     'sleep_hours': horas_sueño,
                     'attendance_percent': asistencia,
                     'previous_scores': calificaciones_previas,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
prediccion=0
datos =  pd.read_csv('calificaciones.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Calificación del alumno')
st.write('La calificación que se predice es ', prediccion)
