# Benchmark de clustering — DMEyF 2026

Objetivo: contrastar distintas formas de clusterización sobre el problema de BAJA+2 sin confundir una partición geométrica con una segmentación sustantiva.

## Regla central

Todos los métodos de una misma pista deben usar:
- el mismo universo de clientes;
- las mismas variables;
- el mismo tratamiento de missing y ceros;
- el mismo escalado;
- el mismo muestreo;
- las mismas métricas de evaluación.

No se elige un método por “verse mejor”.

## Lo que ya existe

El EDA de DMEyF ya motivó la hipótesis de heterogeneidad dentro de BAJA+2 y z501 produjo perfiles asociados a mecanismos como Visa, payroll/pagos/cajeros, actividad/estado de tarjetas, canales/cuentas y préstamos. Esos grupos deben tratarse como hipótesis hasta comprobar estabilidad frente a algoritmo, k, semilla y mes.

Además, en trabajos previos ya se usaron K-means, Ward, Spectral Clustering, PCA, SSE y silhouette, con una conclusión metodológica importante: una buena métrica interna no demuestra por sí sola que exista una clase “real”.

## Tres pistas de comparación

### A. Foto transversal numérica
Para responder: “¿qué tipos de BAJA+2 existen en un mes dado?”

Métodos:
- KMeans
- MiniBatchKMeans
- Agglomerative: Ward, average, complete
- Gaussian Mixture
- Birch
- DBSCAN
- HDBSCAN
- OPTICS
- Spectral Clustering
- Mean Shift
- Affinity Propagation

### B. Datos mixtos
Para responder: “¿qué perfiles aparecen si preservamos variables categóricas/estados además de las numéricas?”

Métodos:
- K-Prototypes
- distancia de Gower + clustering jerárquico
- distancia de Gower + PAM/K-Medoids

### C. Trayectorias temporales
Para responder: “¿existen distintas formas de apagarse?”

Métodos:
- TimeSeriesKMeans euclídeo
- TimeSeriesKMeans + DTW
- Soft-DTW KMeans
- K-Shape
- clustering jerárquico sobre distancias entre trayectorias

Esta pista no debe mezclarse en la misma tabla con A/B: cambia la representación del objeto a clusterizar.

## Métricas

### Calidad geométrica
- silhouette
- Calinski-Harabasz
- Davies-Bouldin
- SSE/inercia cuando corresponda

### Robustez
- estabilidad entre semillas
- estabilidad por bootstrap/submuestras
- concordancia ARI entre soluciones
- sensibilidad a k / parámetros
- estabilidad temporal por foto_mes

### Utilidad sustantiva
- tamaño de cada cluster
- % ceros por familia
- mediana entre positivos
- missing estructural
- variables que más diferencian cluster vs resto
- posibilidad de describir el perfil sin inventar etiquetas

## Salidas mínimas por método

1. tabla de métricas;
2. tamaños de clusters;
3. PCA/UMAP solo para visualizar;
4. heatmap de perfil cluster × variable;
5. estabilidad;
6. comparación con la solución z501;
7. interpretación provisional.

## Criterio de decisión

No habrá un “ganador” por una sola métrica. Una solución pasa a la siguiente etapa si combina:
- separación razonable;
- estabilidad;
- tamaños no degenerados;
- interpretabilidad;
- persistencia temporal.

## Estructura

```
clustering_benchmark/
├── README.md
├── 00_inventario_existente.md
├── config/
│   └── metodos.yaml
└── src/
    └── benchmark_clustering.py
```

## Próximos pasos

1. localizar el notebook/script exacto de z501 y reconstruir sus variables de entrada;
2. congelar un dataset de benchmark;
3. correr primero la pista A;
4. auditar estabilidad;
5. recién después abrir pistas B y C.
