# Inventario inicial

## Material encontrado en Drive

- **TP2_clustering**: trabajo previo sobre validación interna/externa; usa PCA, K-means, Ward y Spectral Clustering; discute silhouette, SSE y el riesgo de interpretar clusters como verdad externa.
- **Informe_EDA.md**: documenta el razonamiento target → EDA → clustering → FE y describe z501 como una exploración de heterogeneidad dentro de BAJA+2.
- **clusters_tendencias.pdf**: resultado previo relacionado con tendencias de clusters; pendiente de incorporar al benchmark de forma reproducible.
- **Clase 9 - Clustering Algoritmos DMCyT**: material docente disponible para revisar variantes y supuestos.

## Hallazgo conceptual de DMEyF

La pregunta correcta no es solo “¿cuántos clusters hay?”, sino:
> ¿se mantienen los mismos perfiles cuando cambiamos algoritmo, semilla, k y mes?

## Solución previa a contrastar

El material de DMEyF menciona una solución z501 con perfiles vinculados a:
- Visa;
- payroll, pagos y cajeros;
- actividad y estados de tarjeta;
- canales y cuentas;
- préstamos.

Estos nombres no se tomarán como etiquetas definitivas. Se usarán únicamente como hipótesis para contrastar contra nuevas soluciones.

## Pendiente crítico

Encontrar el artefacto exacto que produjo z501:
- notebook/script;
- variables incluidas;
- transformación/escalado;
- mes;
- número de clusters;
- semilla;
- algoritmo.

Sin eso no es válido afirmar que una solución nueva “replica” o “mejora” la anterior.
