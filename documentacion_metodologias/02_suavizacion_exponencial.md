# Metodología 2: Suavización Exponencial (ETS)

## ¿Qué es la Suavización Exponencial?

La **Suavización Exponencial** es una familia de modelos de series de tiempo donde las observaciones más recientes reciben un **mayor peso exponencial** que las observaciones antiguas. A diferencia del Promedio Móvil (que da igual peso a todos los N períodos), aquí la influencia decrece exponencialmente hacia el pasado.

El modelo utilizado en este proyecto es un **ETS (Error, Trend, Seasonality)** con componentes **aditivos de error y tendencia**, implementado con la librería `statsmodels`.

### Fórmula simplificada (Holt's Linear / ETS Aditivo con Tendencia)

```
Nivel(t)   = α · Real(t) + (1 - α) · [Nivel(t-1) + Tendencia(t-1)]
Tendencia(t) = β · [Nivel(t) - Nivel(t-1)] + (1 - β) · Tendencia(t-1)
Pronóstico(t+h) = Nivel(t) + h · Tendencia(t)
```

Donde `α` y `β` son parámetros de suavización estimados automáticamente.

### Supuestos del modelo
- Puede capturar una **tendencia lineal** (creciente o decreciente).
- Asume que el **error es aditivo** (se suma al valor, no se multiplica).
- No modela estacionalidad en esta implementación.

---

## ¿Cómo funciona en el código?

### Función principal: `pronosticar_suavizacion_exponencial()` — `app.py` líneas 130–181

```python
def pronosticar_suavizacion_exponencial(datos, columna, steps=1):
```

| Parámetro | Descripción |
|-----------|-------------|
| `datos`   | DataFrame con las ventas históricas |
| `columna` | Nombre del producto |
| `steps`   | Número de períodos futuros a pronosticar |

#### Paso a paso

**1. Preparación de la serie temporal (líneas 144–145)**
```python
df["fecha_dt"] = pd.to_datetime(df["fecha"])
series = df.set_index("fecha_dt")["ventas"]
```
La columna de fechas se convierte al tipo `datetime` y se usa como índice del DataFrame, requisito de `statsmodels`.

**2. Ajuste del modelo ETS (líneas 149–150)**
```python
model = ETSModel(series, error="add", trend="add", seasonal=None)
fit = model.fit(maxiter=1000, disp=False)
```
- `error="add"`: el error del modelo es aditivo.
- `trend="add"`: la tendencia es aditiva (lineal).
- `seasonal=None`: no se modela estacionalidad.
- `maxiter=1000`: máximo de iteraciones para la optimización.
- `disp=False`: suprime la salida de diagnóstico en consola.

Los parámetros `α` y `β` son **estimados automáticamente** por máxima verosimilitud.

**3. Valores ajustados sobre el histórico (línea 153)**
```python
fitted_values = fit.fittedvalues
```
Estos son los valores que el modelo "reconstruye" para los períodos pasados, útiles para calcular métricas de error.

**4. Pronóstico futuro (línea 156)**
```python
forecast_steps = fit.forecast(steps=steps)
```
Genera `steps` predicciones hacia adelante, extrapolando la tendencia estimada.

**5. Generación de fechas futuras (líneas 159–164)**
```python
for i in range(1, steps + 1):
    next_date = last_date + pd.DateOffset(months=i)
    next_date = next_date.replace(day=15)
    future_labels.append(next_date.strftime("%Y-%m-%d"))
```
Calcula las fechas de los meses futuros, ajustando siempre al día 15 para consistencia con el dataset original.

**6. Construcción de resultado (líneas 167–177)**
```python
labels = df["fecha"].tolist() + future_labels
ventas_list = df["ventas"].tolist() + [np.nan] * steps
pronostico_list = fitted_values.tolist() + forecast_steps.tolist()
```
Se combinan el histórico y los pasos futuros en una sola lista para su visualización en el gráfico. El pronóstico final se redondea al entero más cercano (`int(round(x))`) para representar unidades físicas completas.

**7. Manejo de errores — Fallback (línea 181)**
```python
except Exception as e:
    return pronosticar(datos, columna, 3)  # Fallback
```
Si el modelo ETS falla (datos insuficientes, no convergencia), se usa el Promedio Móvil con N=3 como respaldo.

---

## Métrica de calidad: `calcular_metricas()` — líneas 102–128

Esta función auxiliar compara los `fitted_values` (ajustados) con las ventas reales para calcular:

| Métrica | Fórmula |
|---------|---------|
| **MAPE** | `mean(|e| / Real × 100)` |
| **MAPE'** | `mean(|e| / Pronóstico × 100)` |
| **MSE** | `mean(e²)` |
| **RMSE** | `√MSE` |

Solo compara donde **ambas series tienen valores válidos** (línea 104):
```python
mask = ~np.isnan(ventas_reales) & ~np.isnan(pronostico_serie)
```

---

## ¿Cuándo conviene usarlo?

✅ **Útil cuando**: Hay una tendencia visible (crecimiento o caída gradual en ventas).  
✅ **Ventaja**: Más adaptable que el promedio móvil; captura dirección de la tendencia.  
⚠️ **Limitación**: El modelo actual no captura estacionalidad mensual o anual.
