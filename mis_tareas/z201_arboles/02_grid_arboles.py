"""Tarea 1: variar parametros del arbol y comparar ganancia, corte optimo y AUC.

Salidas en outputs/: grid_modelos.csv, ganancia_y_roc.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import _comun as C

PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

data = C.cargar(202104)
y = data["clase_ternaria"]

GRILLA = [
    dict(criterion="gini",    max_depth=5,    min_samples_split=80,   min_samples_leaf=1),   # el de la clase
    dict(criterion="gini",    max_depth=3,    min_samples_split=80,   min_samples_leaf=1),
    dict(criterion="gini",    max_depth=8,    min_samples_split=80,   min_samples_leaf=1),
    dict(criterion="gini",    max_depth=12,   min_samples_split=80,   min_samples_leaf=1),
    dict(criterion="gini",    max_depth=None, min_samples_split=2,    min_samples_leaf=1),   # sin frenos
    dict(criterion="gini",    max_depth=8,    min_samples_split=1000, min_samples_leaf=200),
    dict(criterion="gini",    max_depth=12,   min_samples_split=1000, min_samples_leaf=200),
    dict(criterion="gini",    max_depth=None, min_samples_split=1000, min_samples_leaf=500),
    dict(criterion="entropy", max_depth=5,    min_samples_split=80,   min_samples_leaf=1),
    dict(criterion="entropy", max_depth=8,    min_samples_split=1000, min_samples_leaf=200),
    dict(criterion="entropy", max_depth=12,   min_samples_split=1000, min_samples_leaf=200),
    dict(criterion="gini",    max_depth=8,    min_samples_split=1000, min_samples_leaf=200, ccp_alpha=1e-5),
]

VARIANTES = {
    # fiel a la clase: X = todo menos clase_ternaria (arrastra numero_de_cliente y foto_mes)
    "clase":  data.drop("clase_ternaria", axis=1),
    # sin el ID del cliente ni el periodo: son ruido puro y el arbol igual los usa
    "sin_id": data.drop(["clase_ternaria", "numero_de_cliente", "foto_mes"], axis=1),
}

filas, curvas, ganancias = [], {}, {}
for var, X in VARIANTES.items():
    for k, params in enumerate(GRILLA):
        modelo = DecisionTreeClassifier(random_state=C.SEMILLA, **params).fit(X, y)
        d = C.tabla_cortes(C.get_leaf_info(modelo))
        auc, fpr, tpr = C.auc_desde_cortes(d)
        i = int(d["gan_acumulada"].idxmax())
        nombre = f"{var}|m{k:02d}"
        filas.append({
            "modelo": nombre, "variante": var,
            **{p: params.get(p) for p in ("criterion", "max_depth", "min_samples_split",
                                          "min_samples_leaf", "ccp_alpha")},
            "hojas": len(d),
            "ganancia_max": float(d.loc[i, "gan_acumulada"]),
            "corte_prob": float(d.loc[i, "prob_baja_2"]),
            "clientes_enviados": int(d.loc[i, "enviados"]),
            "baja2_capturados": int(d.loc[i, "TP"]),
            "recall": float(d.loc[i, "TP"] / (d["TP"].iloc[-1] + d["FN"].iloc[-1])),
            "auc": auc,
        })
        curvas[nombre] = (fpr, tpr)
        ganancias[nombre] = (d["enviados"].to_numpy(), d["gan_acumulada"].to_numpy())
        print(f"{nombre:12s} hojas={len(d):5d} gan={filas[-1]['ganancia_max']:>13,.0f} "
              f"corte={filas[-1]['corte_prob']:.4f} auc={auc:.4f}")

res = pd.DataFrame(filas).sort_values("ganancia_max", ascending=False)
res.to_csv(C.OUT / "grid_modelos.csv", index=False)
np.savez(C.OUT / "curvas.npz",
         **{f"roc__{n}__{i}": a for n, c in curvas.items() for i, a in zip("ft", c)},
         **{f"gan__{n}__{i}": a for n, g in ganancias.items() for i, a in zip("xg", g)})

cols = ["modelo", "criterion", "max_depth", "min_samples_split", "min_samples_leaf",
        "hojas", "ganancia_max", "corte_prob", "clientes_enviados", "recall", "auc"]
print("\n" + "=" * 100)
print("RANKING POR GANANCIA\n")
print(res[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print("\nRANKING POR AUC\n")
print(res.sort_values("auc", ascending=False)[["modelo", "max_depth", "min_samples_leaf",
                                               "hojas", "auc", "ganancia_max"]].to_string(index=False))
print(f"\ncorte teorico costo_estimulo/ganancia_acierto = {C.CORTE_TEORICO:.6f}")
print("\nok ->", C.OUT)
