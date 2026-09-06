"""Grafica ganancia acumulada y ROC a partir de las curvas guardadas por 02."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _comun as C

PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"]
# un solo representante por configuracion, con nombre legible
SERIES = [
    ("sin_id|m01", "prof 3"),
    ("sin_id|m00", "prof 5, hoja>=1  (el de la clase)"),
    ("sin_id|m03", "prof 12, hoja>=1"),
    ("sin_id|m06", "prof 12, hoja>=200"),
    ("sin_id|m04", "sin limites  (memoriza el train)"),
]
z = np.load(C.OUT / "curvas.npz")
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

for color, (clave, etiqueta) in zip(PALETA, SERIES):
    x, g = z[f"gan__{clave}__x"], z[f"gan__{clave}__g"]
    ax[0].plot(x, g / 1e6, lw=2, color=color, label=etiqueta)
    j = int(np.argmax(g))
    ax[0].plot(x[j], g[j] / 1e6, "o", ms=8, color=color, mec="white", mew=2)
ax[0].axhline(0, color="#8a8a85", lw=1)
ax[0].set(xlim=(0, 30_000), ylim=(-100, 1350),
          xlabel="clientes estimulados (hojas ordenadas por P(BAJA+2))",
          ylabel="ganancia acumulada (millones $)",
          title="Ganancia acumulada segun punto de corte  ·  202104, medida sobre el mismo train")
ax[0].annotate("ganancia maxima posible:\ntodos los BAJA+2, cero costo\n(1.139 x $1.072.500)",
               xy=(1139, 1221), xytext=(6000, 1080), fontsize=8, color="#5c5b55",
               arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=1))
ax[0].grid(alpha=.25, lw=.6); ax[0].legend(fontsize=8.5, frameon=False, loc="center right")

for color, (clave, etiqueta) in zip(PALETA, SERIES):
    ax[1].plot(z[f"roc__{clave}__f"], z[f"roc__{clave}__t"], lw=2, color=color, label=etiqueta)
ax[1].plot([0, 1], [0, 1], "--", lw=1, color="#8a8a85")
ax[1].set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos",
          title="Curvas ROC  ·  el AUC = 1 es la senal de alarma, no el premio")
ax[1].grid(alpha=.25, lw=.6); ax[1].legend(fontsize=8.5, frameon=False, loc="lower right")

for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.savefig(C.OUT / "ganancia_y_roc.png", dpi=140)
print("ok ->", C.OUT / "ganancia_y_roc.png")
