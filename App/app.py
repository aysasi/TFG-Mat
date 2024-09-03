from flask import Flask, request, render_template_string

from waitress import serve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from tensorflow import keras

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix


import pickle
import pandas as pd


app = Flask(__name__)


logisticModel = None
decisionTree = None
randomForest = None
kNeighbours = None
SVM = None
NaiveBayes = None
ANN = None
RNN = None

def load_logistic():
    global logisticModel
    loaded_model = pickle.load(open('Logistic.sav', 'rb'))
    logisticModel = loaded_model


def load_decision():
    global decisionTree
    loaded_model = pickle.load(open('Decision tree.sav', 'rb'))
    decisionTree = loaded_model


def load_forest():
    global randomForest
    loaded_model = pickle.load(open('Random Forest.sav', 'rb'))
    randomForest = loaded_model


def load_k_neigh():
    global kNeighbours
    loaded_model = pickle.load(open('K-vecinos.sav', 'rb'))
    kNeighbours = loaded_model


def load_SVM():
    global SVM
    loaded_model = pickle.load(open('SVM.sav', 'rb'))
    SVM = loaded_model


def load_naive():
    global NaiveBayes
    loaded_model = pickle.load(open('Naive Bayes.sav', 'rb'))
    NaiveBayes = loaded_model


def load_ANN():
    global ANN
    loaded_model = pickle.load(open('ANN.sav', 'rb'))
    ANN = loaded_model


def load_RNN():
    global RNN
    RNN = keras.models.load_model('RNN.keras')


def predict_RNN(dataset):
    global RNN

    test_ds = np.array(dataset)
    test_ds = test_ds.reshape(test_ds.shape[0], 1, test_ds.shape[1])

    y_pred_aux = RNN.predict(test_ds)
    y_pred_aux = (y_pred_aux > 0.5).astype(int)

    return y_pred_aux[0][0]



def determine_background_color(pred):
    # loaded_model.predict([[781, 0.097, 0.129, 0.184, 0.061]])
    # area	smoothness	compactness	symmetry	fractal_dimension
    print(pred)
    log = int(logisticModel.predict(pred)[0])
    dec = int(decisionTree.predict(pred)[0])
    forest = int(randomForest.predict(pred)[0])
    kvecinos = int(kNeighbours.predict(pred)[0])
    svm = int(SVM.predict(pred)[0])
    nb = int(NaiveBayes.predict(pred)[0])
    ann = int(ANN.predict(pred)[0])
    rnn = predict_RNN(pred)

    return [log, dec, forest, kvecinos, svm, nb, ann, rnn]

@app.route('/', methods=['GET', 'POST'])
def index():
    datos = None
    background_color = None
    if request.method == 'POST':
        dato1 = float(request.form.get('dato1'))
        dato2 = float(request.form.get('dato2'))
        dato3 = float(request.form.get('dato3'))
        dato4 = float(request.form.get('dato4'))
        dato5 = float(request.form.get('dato5'))

        dict = {'area': [dato1], 'smoothness': [dato2], 'compactness': [dato3], 'symmetry': [dato4], 'fractal_dimension': [dato5]}
        pred = pd.DataFrame.from_dict(dict)

        datos = determine_background_color(pred)
        background_color = 'green' if sum(datos) < 4 else 'red'
        datos = ['M' if p == 1 else 'B' for p in datos]

    return render_template_string(html_code, datos=datos, background_color=background_color)


html_code = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción tumor de próstata</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
        }
        
        table { width: 320px; border-spacing: 10px; } 
       
        
        .container {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .panel, .form-container, .result-container {
            flex: 1;
            margin: 10px;
        }
        .panel {
            background-color: #e9ecef;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            max-height: 600px;  /* Asegura que tenga una altura máxima similar a la del panel izquierdo */
            overflow-y: auto;
        }
        .panel button {
            display: block;
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            font-size: 16px;
            color: #fff;
            background-color: #007BFF;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .panel button:hover {
            background-color: #0056b3;
        }
        .input-group {
            margin-bottom: 15px;
        }
        .input-group input {
            width: calc(100% - 20px);
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .input-group label {
            display: block;
            margin-bottom: 5px;
        }
        .result-container {
            background-color: {{ background_color }};
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: left;
            display: {% if datos %} block {% else %} none {% endif %};
        }
        button.submit-button {
            padding: 10px 20px;
            font-size: 16px;
            color: #fff;
            background-color: #007BFF;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button.submit-button:hover {
            background-color: #0056b3;
        }
    </style>
    <script>
        function loadNumbers(datos) {
            document.getElementById('dato1').value = datos[0];
            document.getElementById('dato2').value = datos[1];
            document.getElementById('dato3').value = datos[2];
            document.getElementById('dato4').value = datos[3];
            document.getElementById('dato5').value = datos[4];
        }
    </script>
</head>
<body>

<div class="container">
    <div class="panel">
        <h2>Observaciones predefinidas</h2>
            <button onclick="loadNumbers([1203, 0.125, 0.16, 0.207, 0.06])">Observación 1</button>
            <button onclick="loadNumbers([1297, 0.141, 0.133, 0.181, 0.059])">Observación 2</button>
            <button onclick="loadNumbers([1040, 0.095, 0.109, 0.179, 0.057])">Observación 3</button>
            <button onclick="loadNumbers([781, 0.097, 0.129, 0.184, 0.061])">Observación 4</button>
            <button onclick="loadNumbers([1162, 0.094, 0.172, 0.185, 0.063])">Observación 5</button>
            <button onclick="loadNumbers([870, 0.096, 0.134, 0.19, 0.057])">Observación 6</button>
            <button onclick="loadNumbers([524, 0.09, 0.038, 0.147, 0.059])">Observación 7</button>
            <button onclick="loadNumbers([559, 0.102, 0.126, 0.172, 0.064])">Observación 8</button>
            <button onclick="loadNumbers([1104, 0.091, 0.219, 0.231, 0.063])">Observación 9</button>
            <button onclick="loadNumbers([545, 0.104, 0.144, 0.197, 0.068])">Observación 10</button>
            <button onclick="loadNumbers([449, 0.103, 0.091, 0.168, 0.06])">Observación 11</button>
            <button onclick="loadNumbers([561, 0.088, 0.077, 0.181, 0.057])">Observación 12</button>
            <button onclick="loadNumbers([312, 0.113, 0.081, 0.274, 0.07])">Observación 13</button>
            <button onclick="loadNumbers([269, 0.104, 0.078, 0.172, 0.069])">Observación 14</button>
            <button onclick="loadNumbers([818, 0.092, 0.084, 0.18, 0.054])">Observación 15</button>
            <button onclick="loadNumbers([1245, 0.129, 0.345, 0.291, 0.081])">Observación 16</button>
            <button onclick="loadNumbers([652, 0.113, 0.134, 0.212, 0.063])">Observación 17</button>
            <button onclick="loadNumbers([728, 0.092, 0.104, 0.172, 0.061])">Observación 18</button>
            <button onclick="loadNumbers([555, 0.102, 0.082, 0.164, 0.057])">Observación 19</button>
            <button onclick="loadNumbers([451, 0.105, 0.071, 0.19, 0.066])">Observación 20</button>  
    </div>
    <div class="form-container">
        <h1>Introduzca los datos del tumor</h1>
<form method="post">
    <div class="input-group">
        <label for="dato1">Área</label>
        <input type="number" id="dato1" name="dato1" step="any" value="{{ request.form.get('dato1', '') }}" required>
    </div>
    <div class="input-group">
        <label for="dato2">Suavidad</label>
        <input type="number" id="dato2" name="dato2" step="any" value="{{ request.form.get('dato2', '') }}" required>
    </div>
    <div class="input-group">
        <label for="dato3">Compacidad</label>
        <input type="number" id="dato3" name="dato3" step="any" value="{{ request.form.get('dato3', '') }}" required>
    </div>
    <div class="input-group">
        <label for="dato4">Simetría</label>
        <input type="number" id="dato4" name="dato4" step="any" value="{{ request.form.get('dato4', '') }}" required>
    </div>
    <div class="input-group">
        <label for="dato5">Dimensión fractal</label>
        <input type="number" id="dato5" name="dato5" step="any" value="{{ request.form.get('dato5', '') }}" required>
    </div>
    <button type="submit" class="submit-button">Enviar Datos</button>
</form>
    </div>
    <div class="result-container">
        <h2>Resultados modelos:</h2>
        <table>
            <tr><td><b>Regresión logística:</b></td>  <td>{{ datos[0] }}</td></tr>
            <tr><td><b>Árbol de decisión:</b></td>  <td>{{ datos[1] }}</td></tr>
            <tr><td><b>Random forest:</b></td>  <td>{{ datos[2] }}</td></tr>
            <tr><td><b>K-vecinos:</b></td>  <td>{{ datos[3] }}</td></tr>
            <tr><td><b>SVM:</b></td>  <td>{{ datos[4] }}</td></tr>
            <tr><td><b>Naive Bayes:</b></td> <td>{{ datos[5] }}</td></tr>
            <tr><td><b>ANN:</b> </td> <td>{{ datos[6] }}</td></tr>
            <tr><td><b>RNN: </b></td>  <td>{{ datos[7] }}</td></tr>

        </table>
    </div>
</div>

</body>
</html>
"""


if __name__ == '__main__':
    # CARGAR MODELOS
    load_logistic()
    load_decision()
    load_forest()
    load_k_neigh()
    load_SVM()
    load_naive()
    load_ANN()
    load_RNN()

    # EJECUTAR APLICACIÓN
    serve(app, host="127.0.0.1", port=5000)
