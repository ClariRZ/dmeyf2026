"""Tarea 4: features de historia (lags, deltas, tendencias) para detectar futuras bajas.

Evaluacion HONESTA: entrena en 202104 y mide la ganancia en 202105 y 202106,
que el modelo nunca vio. Comparar sobre el train solo premia al que memoriza.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import duckdb, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
import _comun as C

# variables rotas: mpayroll2/cpayroll2_trx solo viven en 202106, ccajas_depositos muere en 202105
ROTAS = ["mpayroll2", "cpayroll2_trx", "ccajas_depositos",
         "mcuenta_corriente_adicional", "Master_madelantodolares", "Visa_madelantodolares"]

# las que mas separan BAJA+2 segun el univariado de 05
CLAVE = ["ctrx_quarter", "mcaja_ahorro", "mpasivos_margen", "mtarjeta_visa_consumo",
         "ctarjeta_visa_transacciones", "mcuentas_saldo", "mautoservicio",
         "ctarjeta_debito_transacciones", "ccomisiones_otras", "cproductos",
         "mpayroll", "cpayroll_trx", "chomebanking_transacciones", "mrentabilidad",
         "mactivos_margen", "mtransferencias_recibidas", "ctransferencias_emitidas"]

todas = [c for c in duckdb.sql(f"select * from '{C.PARQUET}' limit 0").df().columns]
base_cols = [c for c in todas if c not in ROTAS + ["numero_de_cliente", "foto_mes", "clase_ternaria"]]

nuevas = []
for v in CLAVE:
    w = f'over (partition by numero_de_cliente order by foto_mes)'
    nuevas += [
        f'lag("{v}", 1) {w} as "{v}__lag1"',
        f'"{v}" - lag("{v}", 1) {w} as "{v}__delta1"',
        # cuanto vale hoy contra su propio promedio historico (1 = igual, <1 = se apago)
        f'"{v}" / nullif(avg("{v}") {w[:-1]} rows between 3 preceding and 1 preceding), 0) as "{v}__vs_hist"',
        f'max("{v}") {w[:-1]} rows between 3 preceding and 1 preceding) as "{v}__max_hist"',
    ]
# senales globales de "cliente apagandose"
nuevas += [
    'case when Visa_status is null then 1 else 0 end as sin_visa',
    'case when Master_status is null then 1 else 0 end as sin_master',
    '(case when Visa_status is null then 1 else 0 end +'
    ' case when Master_status is null then 1 else 0 end) as tarjetas_faltantes',
]

sql = f"""
with h as (
    select numero_de_cliente, foto_mes, clase_ternaria
         , {', '.join(f'"{c}"' for c in base_cols)}
         , {', '.join(nuevas)}
    from '{C.PARQUET}'
)
select * from h where foto_mes in (202104, 202105, 202106)
"""
print("construyendo features...")
df = duckdb.sql(sql).df()
nuevas_cols = [c for c in df.columns if c not in base_cols + ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
print(f"{len(df):,} filas · {len(base_cols)} variables originales + {len(nuevas_cols)} nuevas")

for c in df.columns:
    if df[c].dtype == "float64":
        df[c] = df[c].astype("float32")


def ganancia_en(modelo, X, y_real):
    """Ordena por P(BAJA+2) y devuelve ganancia maxima, corte optimo y ganancia al corte teorico."""
    i = list(modelo.classes_).index("BAJA+2")
    p = modelo.predict_proba(X)[:, i]
    orden = np.argsort(-p)
    es_baja2 = (y_real.to_numpy()[orden] == "BAJA+2")
    gan = np.cumsum(np.where(es_baja2, C.GANANCIA_ACIERTO, -C.COSTO_ESTIMULO))
    j = int(np.argmax(gan))
    k = int((p[orden] >= C.CORTE_TEORICO).sum())
    return {"gan_max": float(gan[j]), "enviados_opt": j + 1,
            "corte_opt": float(p[orden][j]),
            "gan_corte_teorico": float(gan[k - 1]) if k else 0.0,
            "enviados_teorico": k}


tr = df[df.foto_mes == 202104]
y_tr = tr["clase_ternaria"]
PARAMS = dict(criterion="gini", random_state=C.SEMILLA, max_depth=12,
              min_samples_split=1000, min_samples_leaf=200)

JUEGOS = {"solo originales": base_cols, "originales + historia": base_cols + nuevas_cols}
resultados = []
modelos = {}
for nombre, cols in JUEGOS.items():
    mod = DecisionTreeClassifier(**PARAMS).fit(tr[cols], y_tr)
    modelos[nombre] = (mod, cols)
    fila = {"juego": nombre, "n_vars": len(cols)}
    for mes in (202104, 202105, 202106):
        sub = df[df.foto_mes == mes]
        met = ganancia_en(mod, sub[cols], sub["clase_ternaria"])
        etiqueta = "train" if mes == 202104 else "test"
        fila[f"{mes}_{etiqueta}_gan_max"] = met["gan_max"]
        fila[f"{mes}_{etiqueta}_gan_corte_teorico"] = met["gan_corte_teorico"]
        fila[f"{mes}_{etiqueta}_enviados"] = met["enviados_teorico"]
    resultados.append(fila)

res = pd.DataFrame(resultados)
res.to_csv(C.OUT / "features_out_of_time.csv", index=False)
pd.set_option("display.width", 220)
print("\n" + "=" * 100)
print("GANANCIA  (202104 = train, 202105 y 202106 = meses que el modelo nunca vio)\n")
print(res.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

for mes in (202105, 202106):
    a = res.loc[0, f"{mes}_test_gan_corte_teorico"]
    b = res.loc[1, f"{mes}_test_gan_corte_teorico"]
    print(f"\n{mes} al corte teorico: {a:,.0f} -> {b:,.0f}   ({(b-a)/abs(a):+.1%})")

mod, cols = modelos["originales + historia"]
imp = (pd.DataFrame({"variable": cols, "importancia": mod.feature_importances_})
       .query("importancia > 0").sort_values("importancia", ascending=False))
imp["es_nueva"] = imp["variable"].isin(nuevas_cols)
imp.to_csv(C.OUT / "importancia_con_features.csv", index=False)
print("\n" + "=" * 100)
print("IMPORTANCIA DE VARIABLES (top 20; es_nueva = feature construida)\n")
print(imp.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nlas features nuevas se llevan el {imp.loc[imp.es_nueva, 'importancia'].sum():.1%} "
      f"de la importancia total, siendo {len(nuevas_cols)}/{len(cols)} de las columnas")
print("\nok ->", C.OUT)
