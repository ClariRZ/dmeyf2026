# Protocolo general de clustering — DMEyF 2026

## Objetivo

Evaluar si dentro de la población BAJA+2 existen estructuras de agrupamiento reproducibles, estables e interpretables, distinguiendo cuidadosamente entre:

1. estructura geométrica;
2. estabilidad estadística;
3. dependencia de la representación;
4. utilidad descriptiva;
5. evidencia suficiente para hablar de perfiles.

El objetivo NO es encontrar “el mejor algoritmo” ni producir clusters atractivos.

---

# Algoritmo de procedimiento

## Fase 0 — Congelar la pregunta

**Pregunta principal**
> ¿Existen subgrupos estables dentro de BAJA+2 definidos por combinaciones distintas de señales bancarias?

**Preguntas secundarias**
- ¿La estructura depende del algoritmo?
- ¿La estructura depende de las variables elegidas?
- ¿Se mantiene al cambiar semilla o muestra?
- ¿Se mantiene entre meses?
- ¿Los clusters son interpretables sin forzar etiquetas?
- ¿Los mismos clientes permanecen juntos entre soluciones?

No se cambia esta pregunta después de mirar resultados.

---

## Fase 1 — Reconstruir antecedentes

Antes de ejecutar un nuevo clustering:

1. recuperar el experimento z501;
2. identificar población;
3. identificar mes;
4. identificar variables;
5. identificar transformaciones;
6. identificar algoritmo;
7. identificar k;
8. identificar semilla;
9. registrar resultados disponibles;
10. registrar limitaciones.

Resultado:
`02_antes_del_clustering/reconstruccion_z501.md`

Si faltan elementos, deben quedar explícitamente como “no recuperado”.

---

## Fase 2 — Definir dos representaciones independientes

### Línea A — Agnóstica

Objetivo:
reducir al mínimo la influencia de la narrativa previa del EDA.

Reglas:
- usar una representación amplia del cliente;
- controlar redundancia extrema;
- evitar seleccionar variables solo porque diferencian BAJA+2;
- documentar exclusiones objetivas;
- no usar etiquetas semánticas de clusters anteriores.

### Línea B — Informada por EDA

Objetivo:
probar explícitamente la hipótesis de mecanismos encontrada en el EDA.

Incluye dimensiones como:
- actividad general;
- tarjetas;
- payroll;
- cuentas;
- transferencias;
- canales;
- préstamos;
- estados/inactividad;
- ceros;
- mediana entre positivos;
- cambios temporales cuando corresponda.

La comparación A vs B es parte del experimento.

---

## Fase 3 — Congelar dataset de benchmark

Crear un manifiesto con:

- archivo fuente;
- hash del archivo si es posible;
- fecha;
- foto_mes;
- población;
- filtros;
- n original;
- n final;
- variables;
- tipos;
- missing;
- tratamiento de infinitos;
- tratamiento de outliers;
- tratamiento de ceros;
- escalado;
- semilla de muestreo.

No se permite cambiar el dataset a mitad de una comparación sin abrir un nuevo experimento.

---

## Fase 4 — Registrar hipótesis antes de ejecutar

Cada experimento debe contener:

- qué método se prueba;
- por qué;
- qué estructura favorece ese método;
- qué esperamos que pueda detectar;
- qué resultado NO se considerará evidencia;
- parámetros fijados antes de ejecutar.

Archivo:
`00_pre_registro.md`

---

## Fase 5 — Ejecutar

Cada experimento debe generar:

- configuración;
- log;
- asignación cluster por cliente;
- métricas;
- tamaños;
- ruido/noise cuando exista;
- centroides/prototipos cuando aplique;
- tiempo de ejecución;
- errores o warnings.

Nunca borrar un experimento fallido.

---

## Fase 6 — Validación interna

Calcular cuando corresponda:

- silhouette;
- Calinski-Harabasz;
- Davies-Bouldin;
- SSE/inercia;
- BIC/AIC para modelos probabilísticos;
- proporción de ruido para métodos de densidad.

Regla:
ninguna métrica interna valida por sí sola la existencia de perfiles reales.

---

## Fase 7 — Estabilidad

### 7.1 Semillas
Repetir con múltiples semillas cuando el método sea estocástico.

### 7.2 Submuestras / bootstrap
Repetir sobre muestras parciales.

### 7.3 Sensibilidad a hiperparámetros
Cambiar k, eps, min_samples, covariance_type u otros parámetros relevantes.

### 7.4 Concordancia
Calcular:
- ARI;
- NMI opcional;
- matriz de co-asignación;
- porcentaje de pares de clientes que permanecen juntos.

### 7.5 Temporal
Repetir en distintos foto_mes cuando la definición del target lo permita.

---

## Fase 8 — Perfilado sin nombrar

Primero usar:
- C0;
- C1;
- C2;
- etc.

Para cada cluster:

- n;
- %;
- % ceros;
- mediana;
- mediana positivos;
- IQR;
- missing;
- effect size one-vs-rest;
- variables más diferenciadoras;
- familias sobrerrepresentadas.

No asignar nombres interpretativos en esta fase.

---

## Fase 9 — Interpretación

Separar obligatoriamente:

### Lo observado
Hechos medidos.

### Lo compatible con los datos
Interpretaciones plausibles.

### Lo no demostrado
Hipótesis no confirmadas.

### Posibles explicaciones alternativas
Dependencia de escala, correlación, producto ausente, outliers, distribución, etc.

---

## Fase 10 — Comparación entre métodos

Construir una tabla maestra con:

- método;
- representación;
- k;
- métricas;
- estabilidad;
- ruido;
- tamaños;
- interpretabilidad;
- similitud con z501;
- similitud entre métodos;
- estabilidad temporal.

No crear ranking único.

---

## Fase 11 — Núcleos estables

Buscar grupos de clientes que permanecen juntos en múltiples soluciones.

Esto puede ser más informativo que cualquier partición individual.

Producto:
- matriz de co-clustering;
- consenso;
- clientes ambiguos;
- clientes núcleo.

---

## Fase 12 — Decisión sobre etiquetas

Solo después de todas las fases anteriores se puede evaluar si una descripción semántica es defendible.

Una etiqueta debe:
- apoyarse en diferencias cuantificadas;
- no exagerar;
- no implicar causalidad;
- no ocultar heterogeneidad;
- poder ser refutada por los datos.

Si no se cumple, el cluster queda como C0/C1/etc.

---

## Fase 13 — Informe de experimento

Cada experimento debe terminar con un informe completo y auditable.

Ver:
`protocolo/plantilla_informe_ensayo.md`

---

## Fase 14 — Informe final

El informe final NO reemplaza los informes individuales.

Debe responder:

- qué estructura se repite;
- qué depende del algoritmo;
- qué depende de la representación;
- qué resultados son inestables;
- qué perfiles tienen evidencia suficiente;
- qué hipótesis quedan abiertas;
- qué no puede concluirse.

---

# Regla de oro

> Un clustering es una propuesta de organización de los datos, no una verdad revelada.

La evidencia aumenta cuando una estructura:
- aparece con distintas representaciones;
- aparece con distintos algoritmos;
- es estable a perturbaciones;
- es interpretable;
- persiste temporalmente;
- no depende de una sola decisión analítica.
