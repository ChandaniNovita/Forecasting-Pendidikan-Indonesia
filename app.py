# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    provinces = sorted(data['Provinsi'].unique())
    education_levels = sorted(data['Jenjang Pendidikan'].unique())
    if request.method == 'POST':
        provinsi = request.form['provinsi']
        jenjang = request.form['jenjang']
        tahun_prediksi = int(request.form['tahun'])

        # Filter data
        filtered_data = data[(data['Provinsi'] == provinsi) & (data['Jenjang Pendidikan'] == jenjang)]

        # Train model
        model = train_model(filtered_data)

        # Predict
        prediksi = model.predict([[tahun_prediksi]])[0]

        # Create Plot
        fig = px.line(filtered_data, x='Tahun', y='Presentase peserta didik', title=f'Trend {jenjang} di {provinsi}')
        fig.add_scatter(x=[tahun_prediksi], y=[prediksi], mode='markers', name='Prediksi', marker=dict(color='red'))
        plot_html = pio.to_html(fig, full_html=False)

        return render_template('result.html', provinsi=provinsi, jenjang=jenjang, tahun=tahun_prediksi, prediksi=prediksi, plot_html=plot_html)

    return render_template('index.html', provinces=provinces, education_levels=education_levels)

if __name__ == '__main__':
    app.run(debug=True)