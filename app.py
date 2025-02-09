import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import streamlit as st

# Load dataset
data = pd.read_csv('datapendidikan.csv', sep=';')

# Data Cleaning
data['Presentase peserta didik'] = data['Presentase peserta didik'].replace('-', np.nan).astype(float)
data.dropna(inplace=True)
data = data[data['Provinsi'] != 'INDONESIA']

# Train Linear Regression Model
def train_model(filtered_data):
    X = filtered_data[['Tahun']].values
    y = filtered_data['Presentase peserta didik'].values
    model = LinearRegression()
    model.fit(X, y)
    return model

# Streamlit UI
st.title("Prediksi Presentase Peserta Didik Berdasarkan Tahun")

# Pilihan dropdown
provinces = sorted(data['Provinsi'].unique())
education_levels = sorted(data['Jenjang Pendidikan'].unique())

provinsi = st.selectbox("Pilih Provinsi:", provinces)
jenjang = st.selectbox("Pilih Jenjang Pendidikan:", education_levels)
tahun_prediksi = st.number_input("Masukkan Tahun Prediksi:", min_value=2023, step=1)

# Tombol prediksi
if st.button("Prediksi"):
    # Filter data
    filtered_data = data[(data['Provinsi'] == provinsi) & (data['Jenjang Pendidikan'] == jenjang)]

    if filtered_data.empty:
        st.warning("Data tidak tersedia untuk kombinasi tersebut.")
    else:
        # Train model
        model = train_model(filtered_data)

        # Predict
        prediksi = model.predict([[tahun_prediksi]])[0]
        
        # Hasil prediksi
        st.success(f"Prediksi Presentase Peserta Didik pada Tahun {tahun_prediksi} di {provinsi} untuk Jenjang {jenjang}: {prediksi:.2f}%")

        # Plot grafik
        fig = px.line(filtered_data, x='Tahun', y='Presentase peserta didik',
                       title=f'Trend {jenjang} di {provinsi}')
        fig.add_scatter(x=[tahun_prediksi], y=[prediksi], mode='markers',
                         name='Prediksi', marker=dict(color='red'))
        st.plotly_chart(fig)
