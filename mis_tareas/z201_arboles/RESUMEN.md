# Tarea `z201_Sobre_Árboles` — resultados

Resumen de resultados. La resolucion explicada esta en `z201_resolucion.ipynb`; esto es la version corta.

Corrido local sobre `monday/competencia_01_crudo.csv` (6 períodos, 202103–202108).
Scripts en orden: `01_target` → `02_grid_arboles` → `03_plots` → `04_metricas_corte`
→ `05_eda_periodos` → `06_features` → `07_eda_graficos`. Salidas en `outputs/`.

Entorno: `.venv` del repo (`duckdb`, `scikit-learn`, `seaborn`, `matplotlib` instalados).

```bash
.venv/bin/python mis_tareas/z201_arboles/01_target.py
```

---

## 0. El target (lo que `z101` deja abierto)

```sql
case
    when mes_1 is null then null      -- no hay mes+1 en la ventana
    when mes_1 = 0    then 'BAJA+1'
    when mes_2 is null then null      -- hay mes+1 pero no mes+2
    when mes_2 = 0    then 'BAJA+2'
    else 'CONTINUA'
end
```

Los dos `is null` son el punto fino: sin ellos, 202107 y 202108 caen en el `else`
y quedan etiquetados CONTINUA cuando en realidad **no se pueden saber**.

| foto_mes | BAJA+1 | BAJA+2 | CONTINUA | % BAJA+2 |
|---|---|---|---|---|
| 202103 | 1019 | 960 | 160921 | 0,589% |
| 202104 | 964 | 1139 | 161181 | **0,698%** |
| 202105 | 1143 | 870 | 161755 | 0,531% |
| 202106 | 874 | 1098 | 162142 | 0,669% |
| 202107 | 1103 | — | — | sin ventana |
| 202108 | — | — | — | sin ventana |

**Solo 202103–202106 son entrenables.** Y ojo: la tasa de BAJA+2 oscila entre
0,53% y 0,70% — un ±25% mes a mes que no es del modelo sino del mundo.

---

## 1. ¿Qué modelo dio más ganancia? ¿Cuál el peor?

De 24 configuraciones (`outputs/grid_modelos.csv`), medidas **sobre el mismo train**:

| modelo | hojas | ganancia | corte | AUC |
|---|---|---|---|---|
| sin límites (`max_depth=None`, `min_samples_leaf=1`) | 2712 | **$1.221.577.500** | 1,0000 | **1,0000** |
| prof 12, hoja≥1 | 203 | $565.070.000 | 0,0283 | 0,919 |
| prof 12, hoja≥200 | 123 | $548.020.000 | 0,0253 | 0,936 |
| prof 5, hoja≥1 *(el de la clase)* | 23 | $431.447.500 | 0,0275 | 0,880 |
| prof 3 | 8 | $363.962.500 | 0,0729 | 0,813 |

El "mejor" por ganancia **y** por AUC es el mismo: el árbol sin frenos, con AUC = 1,000
y ganancia $1.221.577.500. Ese número es exactamente 1.139 × $1.072.500: aisló los
1.139 BAJA+2 en hojas puras, sin un solo falso positivo. **No aprendió, memorizó.**
Es el peor modelo de los 24, disfrazado del mejor, y solo se ve porque estamos midiendo
sobre los mismos datos con los que entrenó. Un AUC de 1 en este problema es una alarma,
no un premio.

Entre los modelos sanos: gana **prof 12 / hoja≥1** por ganancia, pero **prof 12 / hoja≥200**
tiene mejor AUC con la mitad de hojas — la ganancia y el AUC no ordenan igual.

La variante `sin_id` (sacando `numero_de_cliente` y `foto_mes` de X, que el notebook deja
adentro) da prácticamente lo mismo: el árbol no los estaba usando. Sacarlos igual, son ruido.

## 2. La relación con el corte óptimo

`costo_estimulo / ganancia_acierto` = 27.500 / 1.072.500 = **0,025641**

Los cortes empíricos de los modelos sanos: 0,0253 · 0,0256 · 0,0257 · 0,0260 · 0,0273.
Caen todos encima del valor teórico. Tiene sentido: estimular a un cliente conviene
cuando `p × 1.072.500 > 27.500`, o sea `p > 0,025641`. La curva de ganancia acumulada
no descubre nada nuevo, **confirma** el umbral.

Los que se desvían son los malos: prof 3 corta en 0,0729 porque con 8 hojas ninguna
probabilidad cae cerca del umbral.

## 3. Punto de corte óptimo por métrica

Modelo prof 12 / hoja≥200 (`outputs/cortes_optimos.csv`):

| métrica | valor óptimo | clientes a estimular | ganancia | % de la máxima |
|---|---|---|---|---|
| accuracy | 0,9930 | **0** | $0 | 0% |
| especificidad | 1,0000 | **0** | $0 | 0% |
| sensibilidad | 1,0000 | **143.157** | **−$2.683.917.500** | −490% |
| F1 | 0,2111 | 1.324 | $249.590.000 | 45,5% |
| **ganancia** | — | 12.832 | **$548.020.000** | 100% |

Esto es el corazón de la tarea:

- **Accuracy y especificidad se maximizan no haciendo nada.** Con 0,70% de eventos,
  el clasificador "nadie se va" acierta el 99,30%. Cualquier campaña que hagas
  *empeora* la accuracy.
- **Sensibilidad se maximiza estimulando a todo el banco** y funde $2.684 millones.
- **F1 es la única con un óptimo interior** (1.324 clientes) y aun así deja el 55%
  de la plata en la mesa: se para en el corte 0,134, cinco veces más exigente que el
  0,0256 que pide el negocio.
- En este dataset **accuracy y especificidad son la misma curva**, porque con 0,7% de
  positivos los verdaderos negativos dominan el numerador.

Ninguna métrica estadística estándar encuentra el corte que le importa a Miranda.
La función de ganancia no es una métrica más: es la única.

## 4. EDA sobre los períodos descartados

### Meses rotos (data drifting)

| variable | 202103 | 202104 | 202105 | 202106 | 202107 | 202108 |
|---|---|---|---|---|---|---|
| `mpayroll2` | roto | roto | roto | **ok** | roto | roto |
| `cpayroll2_trx` | roto | roto | roto | **ok** | roto | roto |
| `ccajas_depositos` | ok | ok | **roto** | ok | ok | ok |

"Roto" = 100% nula o 100% cero ese mes. `mpayroll2` y `cpayroll2_trx` existen
**solo en 202106**: un modelo entrenado ahí aprende un patrón que no va a estar
en producción. `ccajas_depositos` se cae justo en 202105.

Muertas en los 6 meses, descartables: `mcuenta_corriente_adicional`,
`Master_madelantodolares`, `Visa_madelantodolares`.

### Qué separa a un BAJA+2 (AUC univariada, 202104)

| variable | AUC | lectura |
|---|---|---|
| `ctrx_quarter` | 0,143 | la más fuerte, y baja |
| `mcaja_ahorro` | 0,182 | baja |
| `mpasivos_margen` | 0,200 | baja |
| `mtarjeta_visa_consumo` | 0,207 | baja |
| `mactivos_margen` | 0,718 | **sube** |

Casi todas tienen AUC < 0,5: el cliente que se va **se apaga** — deja de transaccionar,
vacía la caja de ahorro, deja de consumir con la tarjeta. La excepción es
`mactivos_margen`, que sube: se endeuda más antes de irse.

`ctrx_quarter` cuantificado:

| transacciones del trimestre | % BAJA+2 | lift |
|---|---|---|
| 0 | 10,04% | **14,4×** |
| 1–2 | 8,10% | 11,6× |
| 3–5 | 5,72% | 8,2× |
| 11–20 | 3,05% | 4,4× |
| 41–80 | 0,54% | 0,77× |
| 80+ | 0,13% | 0,18× |

### Los nulos son señal, no ruido

El 4,89% de clientes tiene todo el bloque `Visa_*` en null. Entre ellos la tasa de
BAJA+2 es **3,84% contra 0,70% general: lift 5,5×**. No tener tarjeta ya predice la baja.
**Imputar esos nulos con la mediana borra la señal.** Doce variables `Visa_*` comparten
exactamente el mismo patrón, así que un solo flag `sin_visa` las resume.

## 5. Features de historia — el resultado honesto

71 features nuevas sobre 202103–202106: `lag1`, `delta1`, `vs_hist` (valor actual
contra su promedio de los 3 meses previos), `max_hist`, más `sin_visa` / `sin_master` /
`tarjetas_faltantes`.

Evaluación out-of-time: entrenar en 202104, medir en 202105 y 202106 al corte teórico.

| juego de variables | 202104 (train) | 202105 (test) | 202106 (test) |
|---|---|---|---|
| solo originales | $547.717.500 | $185.625.000 | $286.220.000 |
| originales + historia | $547.827.500 | $183.177.500 (−1,3%) | $290.895.000 (+1,6%) |

**No mejoran.** −1,3% en un mes, +1,6% en el otro: ruido, no señal. Con un solo árbol
las features de historia no aportan sobre las originales. Vale la pena reintentarlo con
LightGBM antes de darlas por muertas, pero por ahora el dato es ese.

Dos cosas sí salen de acá:

1. **`tarjetas_faltantes` se lleva el 9,2% de la importancia** — el flag de nulos
   funciona aunque el resto de las features no. Las 71 nuevas juntas se llevan 15,4%,
   y casi todo es ese flag más `cproductos__vs_hist` (2,5%).
2. **El costo real del overfitting, medido**: $547M en train contra $185M–$286M en test.
   El modelo rinde entre un tercio y la mitad de lo que promete. Y la diferencia entre
   los dos meses de test ($185M vs $286M, un 54% de brecha) es varianza del mes, no
   del modelo — lo que hace que evaluar en un solo mes sea engañoso.

---

## Para el video de Miranda

1. El cliente que se va **se apaga primero**: `ctrx_quarter` = 0 multiplica por 14 el
   riesgo. Hay ventana para actuar.
2. **No tener tarjeta Visa multiplica por 5,5 el riesgo** — y es un flag, no un modelo.
3. La campaña conviene con `p > 2,56%`, y eso sale de la aritmética del negocio
   (27.500 / 1.072.500), no del modelo.
4. **Optimizar accuracy sería no llamar a nadie.** Hay que decirlo antes de que alguien
   lo pida.
5. Lo que el modelo promete en train ($547M) no es lo que rinde ($185M–$286M).
