"""Tarea 2: punto de corte optimo para accuracy, sensibilidad, especificidad y F1,
comparado contra el corte que maximiza la ganancia.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import _comun as C

PALETA = {"accuracy": "#2a78d6", "sensibilidad": "#eb6834",
          "especificidad": "#1baf7a", "f1": "#eda100"}

data = C.cargar(202104)
y = data["clase_ternaria"]
X = data.drop(["clase_ternaria", "numero_de_cliente", "foto_mes"], axis=1)

modelo = DecisionTreeClassifier(criterion="gini", random_state=C.SEMILLA,
                                max_depth=12, min_samples_split=1000,
                                min_samples_leaf=200).fit(X, y)
d = C.tabla_cortes(C.get_leaf_info(modelo))
m = C.metricas(d)

# fila 0 = "no estimular a nadie": TP=FP=0, todo negativo
tot_e = int(d["TP"].iloc[-1] + d["FN"].iloc[-1])
tot_ne = int(d["FP"].iloc[-1] + d["TN"].iloc[-1])
cero = pd.DataFrame([{
    "prob_baja_2": 1.0, "enviados": 0, "TPR": 0.0, "sensibilidad": 0.0,
    "especificidad": 1.0, "FPR": 0.0, "precision": np.nan,
    "accuracy": tot_ne / (tot_e + tot_ne), "f1": 0.0, "ganancia": 0.0}])
m = pd.concat([cero, m], ignore_index=True)

print(f"modelo: gini, prof 12, split>=1000, hoja>=200  |  hojas={len(d)}")
print(f"BAJA+2={tot_e}  no-evento={tot_ne}  ({tot_e/(tot_e+tot_ne):.4%} de eventos)\n")

filas = []
for met in ("accuracy", "sensibilidad", "especificidad", "f1", "ganancia"):
    i = int(m[met].idxmax())
    filas.append({
        "metrica": met,
        "valor_optimo": m.loc[i, met],
        "corte_prob": m.loc[i, "prob_baja_2"],
        "clientes_estimulados": int(m.loc[i, "enviados"]),
        "sensibilidad": m.loc[i, "sensibilidad"],
        "especificidad": m.loc[i, "especificidad"],
        "ganancia": m.loc[i, "ganancia"],
    })
opt = pd.DataFrame(filas)
opt.to_csv(C.OUT / "cortes_optimos.csv", index=False)
m.to_csv(C.OUT / "metricas_por_corte.csv", index=False)
print("PUNTO DE CORTE QUE MAXIMIZA CADA METRICA\n")
print(opt.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print(f"\ncorte teorico (costo/ganancia) = {C.CORTE_TEORICO:.6f}")
gan_max = float(m["ganancia"].max())
for met in ("accuracy", "sensibilidad", "especificidad", "f1"):
    g = float(opt.loc[opt.metrica == met, "ganancia"].iloc[0])
    print(f"  si cortaras por {met:14s}: ganancia = {g:>15,.0f}  "
          f"({g/gan_max:6.1%} de la maxima)")

# --- grafico --------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
x = m["enviados"].to_numpy() + 1          # +1 para poder usar escala log

for met, color in PALETA.items():
    # accuracy punteada: con 0,7% de eventos queda pegada a especificidad
    estilo = dict(ls="--", dashes=(5, 3)) if met == "accuracy" else {}
    ax[0].plot(x, m[met], lw=2, color=color, label=met, **estilo)
    i = int(m[met].idxmax())
    ax[0].plot(x[i], m[met].iloc[i], "o", ms=8, color=color, mec="white", mew=2)
ax[0].set(xscale="log", ylim=(-.03, 1.05),
          xlabel="clientes estimulados (escala log)", ylabel="valor de la metrica",
          title="Cada metrica pide un corte distinto")
ax[0].grid(alpha=.25, lw=.6); ax[0].legend(fontsize=9, frameon=False, loc="center left")

ax[1].plot(x, m["ganancia"] / 1e6, lw=2, color="#4a3aa7", label="ganancia")
ax[1].axhline(0, color="#8a8a85", lw=1)
# accuracy y especificidad caen en el mismo punto (no estimular a nadie): una sola etiqueta
ETIQUETAS = {"accuracy": ("accuracy y especificidad\ncoinciden: no estimular a nadie", (10, 14)),
             "sensibilidad": ("sensibilidad: estimular a todos", (-14, 16)),
             "f1": ("F1", (9, -3))}
for met, color in PALETA.items():
    i = int(m[met].idxmax())
    ax[1].plot(x[i], m["ganancia"].iloc[i] / 1e6, "o", ms=9, color=color, mec="white", mew=2)
    if met in ETIQUETAS:
        texto, desp = ETIQUETAS[met]
        ax[1].annotate(texto, (x[i], m["ganancia"].iloc[i] / 1e6), fontsize=8.5,
                       color="#3d3d38", xytext=desp, textcoords="offset points",
                       ha="right" if met == "sensibilidad" else "left")
i = int(m["ganancia"].idxmax())
ax[1].plot(x[i], m["ganancia"].iloc[i] / 1e6, "*", ms=17, color="#4a3aa7", mec="white", mew=1.5)
ax[1].annotate(f"optimo economico\n{int(m['enviados'].iloc[i]):,} clientes · p>={m['prob_baja_2'].iloc[i]:.4f}"
               .replace(",", "."),
               (x[i], m["ganancia"].iloc[i] / 1e6), fontsize=8.5, color="#3d3d38",
               xytext=(16, -46), textcoords="offset points", ha="left")
ax[1].set(xscale="log", xlabel="clientes estimulados (escala log)",
          ylabel="ganancia (millones $)",
          title="Lo que cuesta cortar por la metrica equivocada")
ax[1].grid(alpha=.25, lw=.6)

for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.savefig(C.OUT / "metricas_por_corte.png", dpi=140)
print("\nok ->", C.OUT)
