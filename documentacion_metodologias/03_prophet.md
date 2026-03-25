# Metodología 3: Prophet (Meta)

## ¿Qué es Prophet?

**Prophet** es un modelo de pronóstico de series de tiempo desarrollado por Meta (antes Facebook). Fue diseñado para datos de negocios con patrones complejos: tendencias no lineales, estacionalidades múltiples (anual, semanal, diaria) y efectos de días festivos. Su principal ventaja es que es **robusto y fácil de configurar** para analistas sin profundos conocimientos de series de tiempo.

### Componentes del modelo

Prophet descompone la serie en:

```
y(t) = g(t) + s(t) + h(t) + ε(t)
```

| Componente | Descripción |
|------------|-------------|
| `g(t)` | **Tendencia**: crecimiento lineal o logístico, con posibles puntos de cambio |
| `s(t)` | **Estacionalidad**: patrones periódicos anuales, semanales, etc. (modelados con series de Fourier) |
| `h(t)` | **Días festivos**: efectos especiales en fechas definidas |
| `ε(t)` | **Error**: ruido residual |

### Series de Fourier para estacionalidad

La estacionalidad se aproxima mediante combinaciones de senos y cosenos:

```
s(t) = Σ [aₙ·cos(2πnt/P) + bₙ·sin(2πnt/P)]  donde n = 1, ..., N_fourier
```

Más términos de Fourier → más "ondas" → mayor flexibilidad (pero también más riesgo de sobreajuste).

---

## ¿Cómo funciona en el código?

### Función principal: `pronosticar_prophet()` — `app.py` líneas 183–226

```python
def pronosticar_prophet(datos, columna, steps=1):
```

| Parámetro | Descripción |
|-----------|-------------|
| `datos`   | DataFrame con ventas históricas |
| `columna` | Nombre del producto |
| `steps`   | Número de meses futuros a pronosticar |

#### Paso a paso

**1. Preparación de datos (líneas 187–189)**
```python
df = datos[["Fecha", columna]].copy()
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])
```
Prophet exige que el DataFrame tenga **exactamente** dos columnas: `ds` (fecha) e `y` (valor). El renombramiento es obligatorio.

**2. Inicialización del modelo (líneas 197–204)**
```python
model = Prophet(
    yearly_seasonality=4,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=0.1
)
```

**3. Entrenamiento (línea 205)**
```python
model.fit(df)
```
Prophet ajusta internamente los parámetros de tendencia y estacionalidad mediante Stan (un framework de inferencia bayesiana). Los parámetros de la sección anterior controlan qué tan flexible es este ajuste.

**4. Generación del dataframe futuro (línea 208)**
```python
future = model.make_future_dataframe(periods=steps, freq='MS')
```
- `freq='MS'`: frecuencia mensual, inicio de mes.
- Incluye tanto el histórico como los períodos futuros.

**5. Predicción (línea 210)**
```python
forecast = model.predict(future)
```
Genera columnas como `yhat` (pronóstico), `yhat_lower` y `yhat_upper` (intervalos de confianza).

**6. Límite inferior no negativo (línea 213) ← AJUSTE REALIZADO**
```python
forecast["yhat"] = forecast["yhat"].clip(lower=0)
```
Se aplica un piso de 0, ya que las ventas no pueden ser negativas. Esto evita que los valores bajos del intervalo de confianza se reflejen como pronósticos negativos.

**7. Corrección de fechas (líneas 215–217)**
```python
if pd.to_datetime(df["ds"].iloc[-1]).day == 15:
    labels = [l[:-2] + "15" if i >= len(df) else l for i, l in enumerate(labels)]
```
Si los datos originales tienen el día 15, las fechas futuras (que Prophet genera como inicio de mes `MS`) se ajustan al día 15 para mantener consistencia en el gráfico.

**8. Fallback (línea 226)**
```python
except Exception as e:
    return pronosticar_suavizacion_exponencial(datos, columna, steps)
```
Si Prophet falla, se usa Suavización Exponencial como respaldo.

---

## Ajustes realizados para corregir el comportamiento errático

### Problema original
El modelo con configuración por defecto generaba pronósticos con oscilaciones extremas (p.ej. de -1500 a +3000 en datos que normalmente están entre 100 y 450). Esto se debía a **sobreajuste (overfitting)** del componente estacional sobre un dataset relativamente corto y ruidoso (~54 observaciones mensuales).

### Parámetros modificados

| Parámetro | Valor anterior | Valor nuevo | Efecto |
|-----------|---------------|-------------|--------|
| `yearly_seasonality` | `True` (= 10 términos de Fourier) | `4` | Menos ondas en el ciclo anual → patrón más suave |
| `changepoint_prior_scale` | `0.05` (defecto) | `0.01` | La tendencia es menos sensible a cambios bruscos locales |
| `seasonality_prior_scale` | `10.0` (defecto) | `0.1` | La estacionalidad tiene menor magnitud → no sobrecompensa el ruido |
| `clip(lower=0)` | No existía | Añadido | Pronósticos nunca negativos |
| `round()` | Decimales | Enteros | Representación de unidades reales |

### Resultado de la corrección
Tras los ajustes, los pronósticos para todos los productos se mantienen dentro de rangos físicamente plausibles (ej. Producto_A: mínimo `188`, máximo `~290` para las próximas proyecciones), eliminando los picos artificiales y redondeando a unidades enteras.

---

## ¿Cuándo conviene usarlo?

✅ **Útil cuando**: Hay patrones estacionales anuales claros y suficientes datos (al menos 2 años).  
✅ **Ventaja**: Maneja automáticamente tendencias y estacionalidades; muy robusto ante valores faltantes.  
⚠️ **Limitación**: Con datos cortos y ruidosos, puede sobreajustarse si no se calibran los parámetros de prior. Computacionalmente más costoso que ETS o Promedio Móvil.
