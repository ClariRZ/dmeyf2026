"""Piezas compartidas por los scripts de la tarea."""
import pathlib
import duckdb
import numpy as np
import pandas as pd
from sklearn.tree import _tree

DIR = pathlib.Path(__file__).resolve().parent
DATA = DIR.parent.parent / "monday"   # los datos viven en la carpeta de la catedra
OUT = DIR / "outputs"; OUT.mkdir(exist_ok=True)
CRUDO = DATA / "competencia_01_crudo.csv"
PARQUET = DATA / "competencia_01.parquet"

GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500
CORTE_TEORICO = COSTO_ESTIMULO / GANANCIA_ACIERTO   # 0.025641
SEMILLA = 17


def cargar(foto_mes=None, columnas="*"):
    """Lee competencia_01.parquet con duckdb (evita depender de pyarrow)."""
    where = f"where foto_mes = {foto_mes}" if foto_mes else ""
    return duckdb.sql(f"select {columnas} from '{PARQUET}' {where}").df()


def get_leaf_info(tree):
    """Tabla de hojas: Samples y conteo por clase. Misma idea que la funcion de la clase."""
    t = tree.tree_
    filas = []
    for i in range(t.node_count):
        if t.children_left[i] == _tree.TREE_LEAF:
            cuentas = t.value[i][0] * int(t.n_node_samples[i])
            fila = {"Node": i, "Samples": int(t.n_node_samples[i])}
            for j, c in enumerate(tree.classes_):
                fila[c] = int(round(cuentas[j]))
            filas.append(fila)
    df = pd.DataFrame(filas)
    for c in ("BAJA+1", "BAJA+2", "CONTINUA"):
        if c not in df:
            df[c] = 0
    return df


def tabla_cortes(leaf_df):
    """Ordena hojas por P(BAJA+2) desc y arma ganancia acumulada + matriz de confusion por corte."""
    d = leaf_df.copy()
    d["ganancia"] = (GANANCIA_ACIERTO * d["BAJA+2"]
                     - COSTO_ESTIMULO * (d["BAJA+1"] + d["CONTINUA"]))
    d["prob_baja_2"] = d["BAJA+2"] / d["Samples"]
    d = d.sort_values("prob_baja_2", ascending=False).reset_index(drop=True)
    d["gan_acumulada"] = d["ganancia"].cumsum()

    d["evento"] = d["BAJA+2"]
    d["no_evento"] = d["CONTINUA"] + d["BAJA+1"]
    tot_e, tot_ne = d["evento"].sum(), d["no_evento"].sum()
    d["TP"] = d["evento"].cumsum()
    d["FP"] = d["no_evento"].cumsum()
    d["FN"] = tot_e - d["TP"]
    d["TN"] = tot_ne - d["FP"]
    d["enviados"] = d["Samples"].cumsum()
    return d


def metricas(d):
    """Metricas clasicas por punto de corte, sobre la tabla de tabla_cortes()."""
    m = pd.DataFrame({"prob_baja_2": d["prob_baja_2"], "enviados": d["enviados"]})
    m["TPR"] = m["sensibilidad"] = d["TP"] / (d["TP"] + d["FN"])
    m["especificidad"] = d["TN"] / (d["TN"] + d["FP"])
    m["FPR"] = 1 - m["especificidad"]
    m["precision"] = d["TP"] / (d["TP"] + d["FP"])
    m["accuracy"] = (d["TP"] + d["TN"]) / (d["TP"] + d["TN"] + d["FP"] + d["FN"])
    m["f1"] = 2 * m["precision"] * m["sensibilidad"] / (m["precision"] + m["sensibilidad"])
    m["ganancia"] = d["gan_acumulada"]
    return m


def auc_desde_cortes(d):
    """AUC por trapecios, agregando los extremos (0,0) y (1,1) que la tabla no trae."""
    tpr = np.r_[0.0, (d["TP"] / (d["TP"].iloc[-1] + d["FN"].iloc[-1])).to_numpy(), 1.0]
    fpr = np.r_[0.0, (d["FP"] / (d["FP"].iloc[-1] + d["TN"].iloc[-1])).to_numpy(), 1.0]
    return float(np.trapezoid(tpr, fpr)), fpr, tpr
