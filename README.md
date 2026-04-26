# Agente de Analítica Predictiva Automatizada

Plataforma inteligente que transforma cualquier dataset tabular en un análisis predictivo completo — desde la carga del archivo hasta un reporte ejecutivo en PDF — sin requerir intervención técnica avanzada.

---

## ¿Qué hace?

El agente ejecuta automáticamente todo el ciclo analítico:

Detecta la variable objetivo y el tipo de problema (clasificación o regresión)
Limpia y preprocesa los datos sin configuración manual
Entrena y compara 6 modelos de Machine Learning en paralelo
Selecciona el mejor modelo usando validación cruzada robusta
Opcionalmente optimiza hiperparámetros con Optuna
Genera explicaciones interpretables con SHAP y LIME
Produce un reporte ejecutivo PDF listo para compartir

---

## Preprocesamiento automático

El sistema analiza cada columna del dataset y la clasifica sin intervención manual:

| Tipo detectado | Criterio | Transformación aplicada |
|---|---|---|
| `NUMERIC` | dtype numérico | IterativeImputer → Winsorización 1%-99% → StandardScaler |
| `CATEGORICAL_ORDINAL` | categórica ≤ 10 valores únicos | SimpleImputer → OrdinalEncoder |
| `CATEGORICAL_NOMINAL` | categórica 11–50 valores únicos | SimpleImputer → OneHotEncoder (drop="first") |
| `HIGH_CARDINALITY` | categórica > 50 valores únicos | Descartada |
| `DROP_MISSING` | > 40% nulos | Descartada |
| `ID_LIKE` | nombre contiene "id/key/code" o > 95% valores únicos | Descartada |

> **Sin hardcoding.** El sistema no conoce ningún nombre de columna de antemano.

Los transformers personalizados (`WinsorizationTransformer`, `CorrelationFilter`, `DropHighCardinalityTransformer`) implementan `get_feature_names_out()` para que los nombres de features se propaguen correctamente hacia SHAP y LIME.

---

## Modelos disponibles

| Modelo | Tipo |
|---|---|
| Random Forest | Ensemble de árboles |
| LightGBM | Gradient boosting |
| SVM | Máquina de soporte vectorial |
| KNN | K-vecinos más cercanos |
| Decision Tree | Árbol de decisión simple |
| MLP | Red neuronal multicapa |

---

## Selección de modelo (clave metodológica)

El modelo se selecciona exclusivamente usando el conjunto de entrenamiento, mediante validación cruzada:
**Criterio de selección:** `robust_score = mean_CV − std_CV`
Se penaliza la varianza entre folds para preferir modelos estables sobre modelos que puntúan alto pero de forma inconsistente.

## Interpretación

El agente no selecciona el modelo solo por tener menor varianza, sino por:

Alto rendimiento promedio
Baja variabilidad (más robusto)

Esto evita elegir modelos que:

Puntúan alto pero son inestables
Sobreajustan a ciertos folds
Punto crítico (correcto metodológicamente)
El test set NO participa en la selección del modelo
El test set se usa únicamente para evaluación final

Esto garantiza:

Evaluación honesta (sin data leakage)
Generalización realista

---

## Optimización con Optuna (opcional)

Cuando el usuario activa la optimización, el sistema busca los mejores hiperparámetros para el modelo ganador usando búsqueda bayesiana. El modelo tuneado solo reemplaza al baseline si supera su score en validación cruzada.

```
Baseline CV:  0.8145
Tuned CV:     0.8282  ✔ Modelo optimizado seleccionado
```

---

## Explicabilidad

### SHAP — Importancia global
Usa `TreeExplainer` para modelos basados en árboles (RandomForest, LightGBM, DecisionTree) y `Explainer` genérico para el resto. Los valores se normalizan a shape `(n_samples, n_features)` para manejar correctamente clasificación binaria, multiclase y regresión.

Permite entender:

Qué variables importan más globalmente
Magnitud del impacto en el modelo

### LIME — Explicación local
Explica la predicción de una instancia específica en el espacio transformado, usando nombres de features legibles (no `feature_0`, `feature_1`).

> Las explicaciones indican **influencia estadística**, no causalidad.

---

## Reporte PDF

El reporte generado automáticamente incluye:

- **Cabecera ejecutiva** con metadata del análisis (target, modelo, duración, fecha)
- **Tarjetas de métricas** destacadas (Accuracy, F1-Score o R², RMSE, MAE)
- **Tabla comparativa** de todos los modelos en test set, con el modelo seleccionado marcado con ★
- **Gráfico de barras** de comparación visual (modelo ganador en naranja)
- **Gráfico SHAP** con las top features por importancia media absoluta
- **Tabla LIME** con dirección de influencia (▲ Aumenta / ▼ Reduce) por feature
- **Resumen de preprocesamiento** con columnas usadas y descartadas por grupo
- **Conclusión** con recomendación de uso

---

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd agent

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar la API
uvicorn app.main:app --reload

# Iniciar el dashboard (en otra terminal)
streamlit run dashboard/streamlit_app.py
```

---

## Uso

1. Abre el dashboard en `http://localhost:8501`
2. Sube un archivo CSV o XLSX
3. Activa "Optimizar con Optuna" si quieres mayor precisión (tarda más)
4. Haz clic en **▶ Ejecutar análisis**
5. Espera el resultado — el dashboard se actualiza automáticamente
6. Descarga el PDF o revisa las métricas en pantalla

---

## Formatos de entrada soportados

| Formato | Requisitos |
|---|---|
| CSV | Separado por coma, cualquier encoding |
| XLSX | Primera hoja del libro |

**Requisitos del dataset:**
- Mínimo 50 filas
- Al menos una columna identificable como variable objetivo
- Estructura tabular (sin texto libre, imágenes ni audio)

---

## Resultado de cada análisis

Cada ejecución genera en `outputs/tasks_output/{task_id}/`:

```
results.json    # Métricas, preprocesamiento, SHAP, LIME, parámetros Optuna
report.pdf      # Reporte ejecutivo listo para compartir
```

---

## Limitaciones conocidas

- No procesa datos no estructurados (texto libre, imágenes, audio)
- El rendimiento depende directamente de la calidad del dataset
- Datasets muy pequeños (< 50 filas) o muy ruidosos pueden producir resultados poco confiables
- El modelo seleccionado se basa en métricas estadísticas, no en criterio de negocio
- Se recomienda validar los resultados con expertos del dominio antes de usar en producción

---

## Casos de uso sugeridos

- Predicción de abandono de clientes (churn)
- Clasificación de riesgo crediticio
- Detección de fraude
- Pronóstico de ventas o demanda
- Segmentación de usuarios
- Cualquier problema de clasificación binaria, multiclase o regresión sobre datos tabulares

---

> **Este sistema apoya la toma de decisiones, no la reemplaza.**
> Se recomienda integrar los modelos en procesos controlados y monitorear su desempeño en producción.
