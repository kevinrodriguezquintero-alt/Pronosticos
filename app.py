from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

""" SERVIDOR / API """
app = Flask(__name__)

""" FUNCIONES DE PYTHON """

""" Esta funcion carga el archivo .csv para su lectura en las funciones (def) """
def load_data():
    try:
        return pd.read_csv("venta_historicas.csv")
    except Exception as e:
        return None

def pronosticar(datos, columna, N):
    if columna not in datos.columns:
        return None
        
    df = datos[[columna]].copy()
    df.columns = ["ventas"]
    
    # Calculate Forecast and Errors
    df["Pronostico"] = df["ventas"].rolling(window=N).mean().shift(1)
    df["Error"] = df["ventas"] - df["Pronostico"]
    df["ABS_Error"] = df["Error"].abs()
    
    # Handle division by zero
    df["APE"] = np.where(df["ventas"] != 0, (df["ABS_Error"] / df["ventas"]) * 100, 0)
    df["APE_Prima"] = np.where(df["Pronostico"] != 0, (df["ABS_Error"] / df["Pronostico"]) * 100, 0)
    df["Error_cuadrado"] = df["Error"] ** 2
    
    # Calculate global metrics, ignoring NaNs from the rolling window
    MAPE = df["APE"].mean()
    MAPE_Prima = df["APE_Prima"].mean()
    MSE = df["Error_cuadrado"].mean()
    RMSE = MSE ** 0.5 if not np.isnan(MSE) else 0

    return {
        "ventas": df["ventas"].fillna("").tolist(),
        "pronostico": df["Pronostico"].fillna("").tolist(),
        "labels": list(range(1, len(df) + 1)),
        "metrics": {
            "MAPE": round(MAPE, 2) if not np.isnan(MAPE) else 0,
            "MAPE_Prima": round(MAPE_Prima, 2) if not np.isnan(MAPE_Prima) else 0,
            "MSE": round(MSE, 2) if not np.isnan(MSE) else 0,
            "RMSE": round(RMSE, 2) if not np.isnan(RMSE) else 0
        }
    }

""" ENDPOINTS DE LA API """

""" METHOD: GET """
@app.route("/", methods=["GET"])
def index():
    datos = load_data()
    """ ENCABEZADOS DEL CSV """
    productos = datos.columns.tolist() if datos is not None else []

    """ RENDERIZADO DEL HTML y le pasa los productos como parametro """
    return render_template("Pronostico.html", productos=productos)

""" METHOD: POST """
@app.route("/api/forecast", methods=["POST"])
def forecast():
    """ CARGA EL CSV Y SUS DATOS """
    datos = load_data()
    if datos is None:
        return jsonify({"error": "No se pudo cargar el archivo CSV"}), 500
        
    """ RECIBO LA INFORMACION DEL HTML, lo que viene en el body del fetch en json """
    data = request.json
    N = int(data.get("n", 3))
    producto = data.get("producto")
    
    if not producto:
        return jsonify({"error": "Producto no especificado"}), 400
        
    """ Ejcuta la funcion pronosticar y guarda el resultado """
    resultado = pronosticar(datos, producto, N)
    if resultado is None:
        return jsonify({"error": "Producto no encontrado"}), 404
        
    """ RETORNA EL RESULTADO EN JSON """
    return jsonify(resultado)

""" INICIA EL SERVIDOR """
if __name__ == "__main__":
    app.run(debug=True)