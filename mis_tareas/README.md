# Mis tareas

Resoluciones propias de las tareas de la materia. **Nada de esta carpeta viene de la
cátedra** — los notebooks originales están en `monday/`, `arboles/` y `ensembles/`, y
no los toco, así que traer cambios del repo oficial nunca genera conflictos acá.

| carpeta | consigna original | estado |
|---|---|---|
| `z201_arboles/` | `monday/z201_Sobre_Árboles.ipynb` | resuelta |
| `z402_feature_engineering_sql/` | `monday/z402_Feature_Engineering_en_SQL.ipynb` | resuelta |
| `z301_incertidumbre/` | `monday/z301_Sobre_la_incertidumbre.ipynb` | resuelta |
| `z401_random_forest/` | `monday/z401_Fuego_contra_fuego.ipynb` | resuelta |

Las cuatro tareas de la serie `monday/` están resueltas.

## Cómo leer cada carpeta

El archivo `<consigna>_resolucion.ipynb` es la entrega: copia cada consigna tal como
está en el original y debajo la resuelve, explicando qué se busca en ese punto y cómo
se resuelve. Está ejecutado, así que se lee sin necesidad de correr nada.

Los `.py` numerados al lado son el mismo código partido por etapa, para poder correr
una sola parte sin rehacer todo.

## Correrlo

Con el `.venv` del repo, desde la raíz:

```bash
.venv/bin/python mis_tareas/z201_arboles/01_target.py
```

Los datos (`competencia_01_crudo.csv` y el `competencia_01.parquet` derivado) viven en
`monday/` y no están versionados por tamaño. El primer script los regenera.
