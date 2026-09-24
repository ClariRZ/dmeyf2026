# Informe de ensayo — EXXX

## 1. Identificación

- ID:
- Fecha:
- Commit:
- Autor:
- Rama:
- Script/notebook:
- Dataset:
- Hash/version:
- foto_mes:
- Población:
- Representación: agnóstica / EDA-informed / longitudinal
- Método:

## 2. Pregunta del ensayo

Describir exactamente qué se intenta evaluar.

## 3. Hipótesis previa

### Lo que esperamos que el método pueda detectar

### Lo que NO sería evidencia suficiente

## 4. Datos

- n inicial:
- n final:
- filtros:
- variables:
- variables excluidas:
- motivo de exclusión:
- missing:
- ceros:
- outliers:

## 5. Preprocesamiento

Describir paso a paso:
1.
2.
3.

Incluir:
- imputación;
- escalado;
- transformaciones;
- reducción dimensional;
- muestreo.

## 6. Configuración del algoritmo

- algoritmo:
- librería:
- versión:
- parámetros:
- semilla:
- k o equivalente:

## 7. Procedimiento ejecutado

Enumerar literalmente los pasos realizados.

## 8. Resultado bruto

- cantidad de clusters:
- tamaño por cluster:
- ruido:
- convergencia:
- warnings:
- errores:

## 9. Métricas internas

| Métrica | Valor | Interpretación limitada |
|---|---:|---|
| Silhouette | | |
| Calinski-Harabasz | | |
| Davies-Bouldin | | |
| SSE/BIC/AIC | | |

## 10. Estabilidad

### Semillas

### Submuestras

### Parámetros

### Meses

### Concordancia

## 11. Perfilado de clusters

Sin nombres semánticos.

Para cada cluster:
- n;
- %;
- % ceros;
- mediana;
- mediana positivos;
- missing;
- principales diferencias one-vs-rest;
- familias dominantes.

## 12. Comparación con z501

- similitud:
- diferencias:
- ARI si es posible:
- clientes que cambian:
- clientes núcleo:

## 13. Comparación con otros ensayos

## 14. Lo observado

Solo hechos.

## 15. Lo compatible con los datos

Interpretaciones prudentes.

## 16. Lo que este ensayo NO demuestra

Obligatorio.

## 17. Posibles sesgos o artefactos

- selección de variables;
- escalado;
- redundancia;
- dimensionalidad;
- missing;
- ceros;
- outliers;
- tamaño muestral;
- algoritmo;
- parámetros;
- visualización.

## 18. Resultado negativo o inesperado

Registrar también si:
- no hay separación;
- hay un cluster dominante;
- hay exceso de ruido;
- la solución cambia mucho;
- la interpretación es pobre.

## 19. Conclusión del ensayo

Conclusión limitada a la evidencia.

## 20. Estado

- [ ] replicado
- [ ] estable
- [ ] interpretable
- [ ] temporalmente consistente
- [ ] apto para comparación final
- [ ] requiere repetición

## 21. Archivos generados

- config:
- métricas:
- labels:
- figuras:
- logs:
