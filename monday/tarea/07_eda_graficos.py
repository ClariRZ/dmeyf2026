"""Graficos del EDA: que variables separan, y como se ve la senal en ctrx_quarter."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _comun as C

# rampa secuencial de un solo tono: la barra codifica magnitud, no identidad
RAMPA = ["#184f95", "#1c5cab", "#2a78d6", "#3987e5", "#6da7ec", "#86b6ef", "#b7d3f6"]

uni = pd.read_csv(C.OUT / "eda_univariado.csv").dropna(subset=["fuerza"])
top = uni.head(15).iloc[::-1]
d = C.cargar(202104, "ctrx_quarter, clase_ternaria")
d["es_baja2"] = (d["clase_ternaria"] == "BAJA+2")
base = d["es_baja2"].mean()

fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# --- panel A: poder de separacion univariado
colores = [RAMPA[min(int(i * len(RAMPA) / len(top)), len(RAMPA) - 1)]
           for i in range(len(top))][::-1]
b = ax[0].barh(top["variable"], top["fuerza"], color=colores, height=.68)
ax[0].bar_label(b, labels=[f"{a:.3f}" for a in top["auc"]], padding=4,
                fontsize=8.5, color="#5c5b55")
ax[0].set(xlim=(0, .42), xlabel="poder de separacion  |AUC - 0.5|",
          title="Que variables distinguen un BAJA+2  ·  202104\n(el numero al lado es el AUC crudo)")
ax[0].tick_params(labelsize=9)
ax[0].grid(axis="x", alpha=.25, lw=.6)

# --- panel B: tasa de baja por tramo de actividad
cortes = [-1, 0, 2, 5, 10, 20, 40, 80, 10_000]
etiq = ["0", "1-2", "3-5", "6-10", "11-20", "21-40", "41-80", "80+"]
d["tramo"] = pd.cut(d["ctrx_quarter"], bins=cortes, labels=etiq)
g = d.groupby("tramo", observed=True).agg(tasa=("es_baja2", "mean"), n=("es_baja2", "size"))
b2 = ax[1].bar(g.index.astype(str), g["tasa"] * 100, color=RAMPA[2], width=.68)
ax[1].bar_label(b2, labels=[f"{v:.1f}%" for v in g["tasa"] * 100], padding=3,
                fontsize=9, color="#5c5b55")
ax[1].axhline(base * 100, color="#eb6834", lw=2, ls="--")
ax[1].annotate(f"tasa general: {base:.2%}", (len(g) - 1.4, base * 100),
               xytext=(0, -20), textcoords="offset points", fontsize=9, color="#eb6834")
ax[1].set(xlabel="ctrx_quarter  ·  transacciones del trimestre",
          ylabel="% de clientes que son BAJA+2",
          title="El cliente deja de operar ANTES de darse de baja")
ax[1].grid(axis="y", alpha=.25, lw=.6)

for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.savefig(C.OUT / "eda_variables.png", dpi=140)

print(g.assign(lift=g["tasa"] / base).to_string(float_format=lambda v: f"{v:,.4f}"))
print("\nok ->", C.OUT / "eda_variables.png")
