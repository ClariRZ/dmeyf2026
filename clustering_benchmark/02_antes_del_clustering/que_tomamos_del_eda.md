# Qué se toma del EDA para clustering

## El EDA sí se usa para

### 1. Comprender el significado de las variables
Evita tratar de igual manera:
- conteos;
- montos;
- saldos;
- estados;
- ausencia de producto;
- missing;
- ceros.

### 2. Detectar redundancia
Variables muy correlacionadas pueden sobrerrepresentar una dimensión y dominar distancias.

### 3. Separar cero de bajo uso
Se preserva la distinción entre:
- proporción de ceros;
- nivel de uso entre positivos.

### 4. Definir familias
Tarjetas, cuentas, payroll, canales, transferencias, préstamos, etc.

### 5. Generar hipótesis
El EDA sugiere mecanismos posibles de desvinculación.

### 6. Diseñar la línea EDA-informed
Se construye una representación explícitamente basada en mecanismos hallados.

---

# Qué NO se toma del EDA como verdad previa

## 1. Rankings como selección automática
Una variable discriminante de BAJA+2 no es automáticamente una buena variable para clustering dentro de BAJA+2.

## 2. Etiquetas de clusters
No se presuponen “activo”, “intermedio”, “apagado”, etc.

## 3. Número de clusters
El EDA no fija k.

## 4. Trayectorias inferidas de comparación transversal
CONTINUA > BAJA+2 > BAJA+1 en un mismo mes no demuestra una trayectoria individual.

## 5. Causalidad
El EDA describe asociaciones.

---

# Riesgo principal

Si seleccionamos solo variables que ya expresan la narrativa de “apagado”, podemos inducir clusters que reproduzcan esa narrativa.

Por eso se comparan:

- Línea A: agnóstica;
- Línea B: EDA-informed.

La diferencia entre ambas es un resultado del estudio.
