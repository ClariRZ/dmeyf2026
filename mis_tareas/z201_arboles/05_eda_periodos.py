"""Tarea 3: EDA sumando los periodos descartados (202103-202108).

Busca tres cosas:
  1. estabilidad del target mes a mes
  2. "meses rotos": variables que en algun periodo se vuelven todo cero o todo nulo
  3. poder predictivo univariado de cada variable sobre BAJA+2
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import duckdb, numpy as np, pandas as pd
import _comun as C

pd.set_option("display.width", 200)

# ---------------------------------------------------------------- 1. target
print("=" * 90)
print("1. EL TARGET MES A MES")
print(duckdb.sql(f"""
    select foto_mes
         , count(*) as clientes
         , sum(clase_ternaria = 'BAJA+1')::int as baja1
         , sum(clase_ternaria = 'BAJA+2')::int as baja2
         , round(100.0 * sum(clase_ternaria = 'BAJA+2') / count(*), 3) as pct_baja2
         , sum(clase_ternaria is null)::int as sin_clase
    from '{C.PARQUET}' group by foto_mes order by foto_mes
""").df().to_string(index=False))
print("\n-> solo 202103-202106 sirven para entrenar: 202107 y 202108 no tienen ventana futura")

# --------------------------------------------------- 2. meses rotos (drifting)
cols = [c for c in duckdb.sql(f"select * from '{C.PARQUET}' limit 0").df().columns
        if c not in ("numero_de_cliente", "foto_mes", "clase_ternaria")]

# por variable y por mes: % de nulos y % de ceros
piezas = []
for c in cols:
    piezas.append(f"""avg(case when "{c}" is null then 1.0 else 0.0 end) as "nul__{c}" """)
    piezas.append(f"""avg(case when "{c}" = 0 then 1.0 else 0.0 end) as "cero__{c}" """)
perfil = duckdb.sql(f"select foto_mes, {', '.join(piezas)} from '{C.PARQUET}' "
                    f"group by foto_mes order by foto_mes").df().set_index("foto_mes")

nulos = perfil[[c for c in perfil if c.startswith("nul__")]].rename(columns=lambda s: s[5:])
ceros = perfil[[c for c in perfil if c.startswith("cero__")]].rename(columns=lambda s: s[6:])
muerta = (nulos + ceros) >= 0.999          # el mes no aporta informacion de esa variable

print("\n" + "=" * 90)
print("2. MESES ROTOS  (variable 100% nula o 100% cero en un mes, pero viva en otro)")
sospechosas = muerta.columns[(muerta.any() & ~muerta.all())]
if len(sospechosas):
    tabla = muerta[sospechosas].T.astype(int)
    tabla.columns = [str(c) for c in tabla.columns]
    tabla["meses_rotos"] = tabla.sum(axis=1)
    print(tabla.sort_values("meses_rotos", ascending=False).to_string())
    print("\n-> 1 = ese mes la variable no informa nada. Usar esos meses para entrenar")
    print("   le ensena al modelo un patron que no existe.")
else:
    print("ninguna")
print("\nvariables muertas en TODOS los meses (candidatas a descartar):",
      list(muerta.columns[muerta.all()]) or "ninguna")

# --------------------------------------- 3. poder predictivo univariado (AUC)
print("\n" + "=" * 90)
print("3. PODER PREDICTIVO UNIVARIADO SOBRE BAJA+2  (202104)")
d = C.cargar(202104)
y = (d["clase_ternaria"] == "BAJA+2").astype(int).to_numpy()

def auc_univariada(x, y):
    """AUC de Mann-Whitney sobre los no nulos; los nulos van aparte."""
    ok = ~pd.isna(x)
    if ok.sum() < 100 or y[ok].sum() < 10 or len(np.unique(x[ok])) < 2:
        return np.nan
    r = pd.Series(x[ok]).rank().to_numpy()
    yy = y[ok]
    n1, n0 = yy.sum(), len(yy) - yy.sum()
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

filas = []
for c in cols:
    x = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
    a = auc_univariada(x, y)
    pct_nul = float(pd.isna(x).mean())
    tasa_nul = float(y[pd.isna(x)].mean()) if pct_nul > 0 else np.nan
    filas.append({"variable": c, "auc": a,
                  "fuerza": abs(a - .5) if a == a else np.nan,
                  "pct_nulos": pct_nul, "unicos": int(d[c].nunique()),
                  "tasa_baja2_si_nulo": tasa_nul})
uni = pd.DataFrame(filas).sort_values("fuerza", ascending=False)
uni.to_csv(C.OUT / "eda_univariado.csv", index=False)
print(uni.head(25).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nauc < 0.5 = la variable BAJA cuando el cliente se va;  > 0.5 = sube")

# los nulos como senal
print("\n" + "=" * 90)
print("4. LOS NULOS COMO SENAL  (tasa base de BAJA+2 = "
      f"{y.mean():.4%})")
nul = uni[uni.pct_nulos.between(.01, .99)].copy()
nul["lift_nulo"] = nul["tasa_baja2_si_nulo"] / y.mean()
print(nul.sort_values("lift_nulo", ascending=False)
      .head(12)[["variable", "pct_nulos", "tasa_baja2_si_nulo", "lift_nulo"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n-> lift > 1 significa que TENER el dato faltante ya predice la baja:")
print("   imputar con la mediana borraria esa senal.")
print("\nok ->", C.OUT / "eda_univariado.csv")
