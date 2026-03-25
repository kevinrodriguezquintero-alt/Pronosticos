# Metodología 1: Promedio Móvil (Moving Average)

## ¿Qué es el Promedio Móvil?

El **Promedio Móvil** es una técnica de suavización de series de tiempo. La idea central es que el pronóstico para el próximo período es simplemente el **promedio de los últimos N períodos reales**. Al "mover" esta ventana a lo largo del tiempo, el modelo reduce el impacto de ruido o variaciones aleatorias puntuales.

### Fórmula

```
Pronóstico(t) = (Real(t-1) + Real(t-2) + ... + Real(t-N)) / N
```

donde `N` es el tamaño de la ventana (número de períodos a promediar).

### Supuestos del modelo
- La demanda futura se parece al comportamiento promedio reciente.
- No existe una tendencia creciente o decreciente fuerte.
- No hay estacionalidad marcada dentro de la ventana N.

---

## ¿Cómo funciona en el código?

### Función principal: `pronosticar()` — `app.py` líneas 32–100

```python
def pronosticar(datos, columna, N, steps=1):
```

| Parámetro | Descripción |
|-----------|-------------|
| `datos`   | DataFrame con las ventas históricas del CSV |
| `columna` | Nombre del producto a pronosticar (ej. "Producto_A") |
| `N`       | Tamaño de la ventana del promedio móvil |
| `steps`   | Número de períodos futuros a pronosticar |

#### Paso a paso

**1. Preparación de datos (líneas 36–37)**
```python
df = datos[["Fecha", columna]].copy()
df.columns = ["fecha", "ventas"]
```
Se extrae solo la columna de fechas y la del producto seleccionado.

**2. Generación de fechas futuras (líneas 48–61)**
```python
for i in range(1, steps + 1):
    month = last_date.month + i
    ...
    curr_future_date = last_date.replace(year=year, month=month, day=15)
```
Se calculan las fechas futuras (mes a mes, siempre el día 15) y se agregan como filas vacías al DataFrame.

**3. Cálculo del pronóstico (línea 64)**
```python
df["Pronostico"] = df["ventas"].rolling(window=N).mean().shift(1)
```
- `rolling(window=N).mean()`: calcula el promedio de los últimos N valores en cada punto.
- `.shift(1)`: desplaza 1 período hacia adelante, porque el pronóstico del período `t` usa datos hasta `t-1` (no incluye el valor real de `t`).

**4. Pronóstico para períodos futuros (líneas 68–74)**
```python
last_ma = df["ventas"].tail(N).mean()
pronostico_final = df["Pronostico"].tolist() + [last_ma] * steps
```
Para los pasos futuros, se repite el último promedio móvil calculado. El resultado final se redondea al entero más cercano (`int(round(x))`) ya que no tiene sentido pronosticar unidades fraccionarias de un producto físico.

**5. Métricas de error (líneas 76–88)**

| Métrica | Fórmula | Descripción |
|---------|---------|-------------|
| **MAPE** | `mean(|Error| / Real × 100)` | Error porcentual medio respecto a valor real |
| **MAPE'** | `mean(|Error| / Pronóstico × 100)` | Error porcentual medio respecto al pronóstico |
| **MSE** | `mean(Error²)` | Error cuadrático medio |
| **RMSE** | `√MSE` | Raíz del error cuadrático medio |

---

## Parámetros ajustables desde la interfaz

| Parámetro | Dónde se recibe | Efecto |
|-----------|-----------------|--------|
| `N` | `request.json["n"]` (línea 249) | Tamaño de ventana; más grande = más suave pero más lento |
| `future_steps` | `request.json["future_steps"]` (línea 250) | Cuántos meses futuros pronosticar |

---

## ¿Cuándo conviene usarlo?

✅ **Útil cuando**: La demanda es relativamente estable y sin tendencia fuerte.  
✅ **Ventaja**: Muy simple de interpretar y computacionalmente eficiente.  
⚠️ **Limitación**: No captura tendencias ni estacionalidades. El pronóstico futuro siempre es "plano".
