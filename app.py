from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import sys
import os
import logging

# Configurar ruta para librerías locales
sys.path.append(os.path.abspath("prophet_lib"))
try:
    from prophet import Prophet
    # Desactivar logs ruidosos de Prophet/CmdStanPy
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
except ImportError:
    print("Error: No se pudo cargar Prophet desde prophet_lib")

""" SERVIDOR / API """
app = Flask(__name__)

""" FUNCIONES DE PYTHON """

""" Esta funcion carga el archivo .csv para su lectura en las funciones (def) """
def load_data():
    try:
        return pd.read_csv("venta_historicas.csv")
    except Exception as e:
        return None

def pronosticar(datos, columna, N, steps=1):
    if columna not in datos.columns or "Fecha" not in datos.columns:
        return None
        
    df = datos[["Fecha", columna]].copy()
    df.columns = ["fecha", "ventas"]
    
    if df.empty:
        return {
            "ventas": [],
            "pronostico": [],
            "labels": [],
            "metrics": {"MAPE": 0, "MAPE_Prima": 0, "MSE": 0, "RMSE": 0}
        }
    
    # Generate future dates (next months on the 15th)
    last_date = pd.to_datetime(df["fecha"].iloc[-1])
    future_labels = []
    future_rows = []
    for i in range(1, steps + 1):
        # Calculate next date
        month = last_date.month + i
        year = last_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        curr_future_date = last_date.replace(year=year, month=month, day=15)
        future_labels.append(curr_future_date.strftime("%Y-%m-%d"))
        future_rows.append({"fecha": curr_future_date.strftime("%Y-%m-%d"), "ventas": np.nan})
    
    # Append rows for the future forecast
    df_extended = pd.concat([df, pd.DataFrame(future_rows)], ignore_index=True)
    
    # Calculate Forecast and Errors on original data
    df["Pronostico"] = df["ventas"].rolling(window=N).mean().shift(1)
    
    # The future forecast for multiple steps in Moving Average 
    # will just use the last calculated rolling mean for simplicity
    last_ma = df["Pronostico"].iloc[-1] if not df["Pronostico"].empty else 0
    if np.isnan(last_ma) and len(df) >= N:
        # If the last one is NaN but we have enough data, it's because it was the last shift
        # we calculate it manually
        last_ma = df["ventas"].tail(N).mean()

    pronostico_final = df["Pronostico"].tolist() + [last_ma] * steps
    
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

    # Prepare lists for output
    labels = df["fecha"].tolist() + future_labels
    ventas_list = df["ventas"].tolist() + [np.nan] * steps

    return {
        "ventas": [x if not np.isnan(x) else "" for x in ventas_list],
        "pronostico": [int(round(x)) if not np.isnan(x) else "" for x in pronostico_final],
        "labels": labels,
        "metrics": {
            "MAPE": round(MAPE, 2) if not np.isnan(MAPE) else 0,
            "MAPE_Prima": round(MAPE_Prima, 2) if not np.isnan(MAPE_Prima) else 0,
            "MSE": round(MSE, 2) if not np.isnan(MSE) else 0,
            "RMSE": round(RMSE, 2) if not np.isnan(RMSE) else 0
        }
    }

def calcular_metricas(ventas_reales, pronostico_serie):
    # Aseguramos que solo comparamos donde ambos existen y no son NaN
    mask = ~np.isnan(ventas_reales) & ~np.isnan(pronostico_serie)
    v = ventas_reales[mask]
    p = pronostico_serie[mask]
    
    if len(v) == 0:
        return {"MAPE": 0, "MAPE_Prima": 0, "MSE": 0, "RMSE": 0}
        
    error = v - p
    abs_error = np.abs(error)
    
    ape = np.where(v != 0, (abs_error / v) * 100, 0)
    ape_prima = np.where(p != 0, (abs_error / p) * 100, 0)
    error_cuadrado = error ** 2
    
    mape = np.mean(ape)
    mape_prima = np.mean(ape_prima)
    mse = np.mean(error_cuadrado)
    rmse = np.sqrt(mse)
    
    return {
        "MAPE": round(mape, 2),
        "MAPE_Prima": round(mape_prima, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2)
    }

def pronosticar_suavizacion_exponencial(datos, columna, steps=1):
    if columna not in datos.columns or "Fecha" not in datos.columns:
        return None
        
    df = datos[["Fecha", columna]].copy()
    df.columns = ["fecha", "ventas"]
    
    if df.empty:
        return {
            "ventas": [], "pronostico": [], "labels": [],
            "metrics": {"MAPE": 0, "MAPE_Prima": 0, "MSE": 0, "RMSE": 0}
        }

    # Preparar serie temporal
    df["fecha_dt"] = pd.to_datetime(df["fecha"])
    series = df.set_index("fecha_dt")["ventas"]
    
    try:
        # Ajustar modelo ETS (Aditivo por defecto)
        model = ETSModel(series, error="add", trend="add", seasonal=None)
        fit = model.fit(maxiter=1000, disp=False)
        
        # Valores ajustados (histórico)
        fitted_values = fit.fittedvalues
        
        # Predicción N pasos adelante
        forecast_steps = fit.forecast(steps=steps)
        
        # Generar fechas futuras
        last_date = df["fecha_dt"].iloc[-1]
        future_labels = []
        for i in range(1, steps + 1):
            next_date = last_date + pd.DateOffset(months=i)
            next_date = next_date.replace(day=15)
            future_labels.append(next_date.strftime("%Y-%m-%d"))
        
        # Construir listas para el gráfico
        labels = df["fecha"].tolist() + future_labels
        ventas_list = df["ventas"].tolist() + [np.nan] * steps
        
        # El pronóstico incluye los valores ajustados y los puntos futuros
        pronostico_list = fitted_values.tolist() + forecast_steps.tolist()
        
        return {
            "ventas": [x if not np.isnan(x) else "" for x in ventas_list],
            "pronostico": [int(round(x)) if not np.isnan(x) else "" for x in pronostico_list],
            "labels": labels,
            "metrics": calcular_metricas(df["ventas"].values, fitted_values.values)
        }
    except Exception as e:
        print(f"Error en ETS: {e}")
        return pronosticar(datos, columna, 3) # Fallback

def pronosticar_prophet(datos, columna, steps=1):
    if columna not in datos.columns or "Fecha" not in datos.columns:
        return None
        
    df = datos[["Fecha", columna]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    
    if len(df) < 2:
        return pronosticar(datos, columna, 1, steps)

    try:
        # Inicializar y ajustar modelo Prophet con parámetros más conservadores
        # yearly_seasonality=4 (menos ruidoso), escalas de prioridad bajas para evitar sobreajuste
        model = Prophet(
            yearly_seasonality=4, 
            weekly_seasonality=False, 
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=0.1
        )
        model.fit(df)
        
        # Generar dataframe para predicciones (incluye histórico)
        future = model.make_future_dataframe(periods=steps, freq='MS') 
        # Ajustamos el día al 15 para mantener consistencia si es necesario
        forecast = model.predict(future)
        
        # Asegurar que los pronósticos no sean negativos (Ventas no pueden ser < 0)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        
        # Extraer valores y fechas
        labels = forecast["ds"].dt.strftime("%Y-%m-%d").tolist()
        # Ajustar los labels futuros a día 15 si el original era día 15
        if pd.to_datetime(df["ds"].iloc[-1]).day == 15:
            labels = [l[:-2] + "15" if i >= len(df) else l for i, l in enumerate(labels)]
            
        ventas_list = df["y"].tolist() + [np.nan] * steps
        pronostico_list = forecast["yhat"].tolist()
        
        # Metricas sobre el histórico
        fitted_values = forecast["yhat"].iloc[:len(df)].values
        
        return {
            "ventas": [x if not np.isnan(x) else "" for x in ventas_list],
            "pronostico": [int(round(x)) if not np.isnan(x) else "" for x in pronostico_list],
            "labels": labels,
            "metrics": calcular_metricas(df["y"].values, fitted_values)
        }
    except Exception as e:
        print(f"Error en Prophet: {e}")
        return pronosticar_suavizacion_exponencial(datos, columna, steps)

""" ENDPOINTS DE LA API """

""" METHOD: GET """
@app.route("/", methods=["GET"])
def index():
    datos = load_data()
    """ ENCABEZADOS DEL CSV, excluyendo 'Fecha' """
    productos = [col for col in datos.columns if col != "Fecha"] if datos is not None else []

    """ RENDERIZADO DEL HTML y le pasa los productos como parametro """
    return render_template("Pronostico.html", productos=productos)

@app.route("/api/forecast", methods=["POST"])
def forecast():
    """ CARGA EL CSV Y SUS DATOS """
    datos = load_data()
    if datos is None:
        return jsonify({"error": "No se pudo cargar el archivo CSV"}), 500
        
    # RECIBO LA INFORMACION DEL HTML
    data = request.json
    N = int(data.get("n", 3))
    future_steps = int(data.get("future_steps", 1))
    method = data.get("method", "promedio_movil")
    
    # Identify all product columns (exclude "Fecha")
    productos = [col for col in datos.columns if col != "Fecha"]
    
    resultados = {}
    for producto in productos:
        if method == "suavizacion_exponencial":
            res = pronosticar_suavizacion_exponencial(datos, producto, future_steps)
        elif method == "otro": # Prophet
            res = pronosticar_prophet(datos, producto, future_steps)
        else:
            res = pronosticar(datos, producto, N, future_steps)
            
        if res:
            resultados[producto] = res
            
    """ RETORNA TODOS LOS RESULTADOS EN JSON """
    return jsonify(resultados)

""" INICIA EL SERVIDOR """
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        # Overwrite the existing historical data
        file.save("venta_historicas.csv")
        return jsonify({"message": "File uploaded successfully"}), 200
        
    return jsonify({"error": "Invalid file format, CSV required"}), 400

if __name__ == "__main__":
    app.run(debug=True)