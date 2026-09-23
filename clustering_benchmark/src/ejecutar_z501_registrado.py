from datetime import datetime
from importlib import metadata, util
from pathlib import Path
import hashlib
import json
import sys
import traceback

OUT = Path(__file__).resolve().parent
CSV = Path(sys.argv[1])
ORIGINAL = OUT / "z501_original.py"
if (OUT / "resultado.json").exists():
    raise SystemExit("Esta carpeta ya tiene resultados; usá una ejecución nueva.")

r = {"estado": "NO_COMPLETADO", "inicio": datetime.now().astimezone().isoformat(),
     "parametros": {"k": 5, "seed": 214363, "n_trees": 300,
                    "min_samples_leaf": 50, "top_n": 3, "banda": "ic95"},
     "pasos": [], "clientes_por_cluster": {}, "versiones": {}}
try:
    for package in ("numpy", "pandas", "scikit-learn", "duckdb", "matplotlib"):
        r["versiones"][package] = metadata.version(package)
    r["python"] = sys.version
    r["csv"] = str(CSV)
    with CSV.open("rb") as f:
        r["csv_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    r["original_sha256"] = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    spec = util.spec_from_file_location("z501", ORIGINAL)
    m = util.module_from_spec(spec)
    spec.loader.exec_module(m)

    log_original = m.log
    def log(text):
        r["pasos"].append(text)
        log_original(text)
    m.log = log

    cargar = m.cargar_muestra
    def cargar_y_guardar(*args, **kwargs):
        df = cargar(*args, **kwargs)
        df[[m.ID_COL, m.MES_COL, m.TARGET_COL, m.GRUPO_COL]].to_csv(
            OUT / "muestra_cliente_mes.csv.gz", index=False)
        features = m.columnas_features(df)
        (OUT / "variables.json").write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
        r["muestra"] = {str(g): {"clientes": int(d[m.ID_COL].nunique()), "filas": len(d)}
                        for g, d in df.groupby(m.GRUPO_COL)}
        return df
    m.cargar_muestra = cargar_y_guardar

    clusterizar = m.clusterizar
    def clusterizar_y_guardar(P, k, seed):
        labels = clusterizar(P, k, seed)
        m.np.savez_compressed(OUT / "representacion_rf.npz", X=P.to_numpy(),
                             ids=P.index.to_numpy(dtype="int64"),
                             variables=m.np.asarray(P.columns, dtype=str))
        labels.to_csv(OUT / "asignaciones.csv", index_label=m.ID_COL)
        counts = labels.value_counts().sort_index()
        counts.rename("n").to_csv(OUT / "tamanos.csv", index_label="cluster")
        r["clientes_por_cluster"] = {str(k): int(v) for k, v in counts.items()}
        r["n_clusterizados"] = len(labels)
        r["estado"] = "CLUSTERIZADO_SALIDAS_INCOMPLETAS"
        return labels
    m.clusterizar = clusterizar_y_guardar

    caracterizar = m.caracterizar_clusters
    def caracterizar_y_guardar(*args, **kwargs):
        table = caracterizar(*args, **kwargs)
        table.to_csv(OUT / "atributos_rf.csv", index=False)
        return table
    m.caracterizar_clusters = caracterizar_y_guardar

    tendencias = m.tendencias_por_cluster
    def tendencias_y_guardar(*args, **kwargs):
        table = tendencias(*args, **kwargs)
        table.to_csv(OUT / "tendencias.csv")
        return table
    m.tendencias_por_cluster = tendencias_y_guardar

    sys.argv = [str(ORIGINAL), "--csv", str(CSV), "--out", str(OUT / "clusters_tendencias.pdf"),
                "--k", "5", "--seed", "214363", "--n-trees", "300",
                "--min-samples-leaf", "50", "--top-n", "3", "--banda", "ic95"]
    (OUT / "configuracion.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    m.main()
    r["estado"] = "EJECUCION_COMPLETADA_SIN_VALIDACION_DE_ESTABILIDAD"
except (Exception, KeyboardInterrupt) as error:
    r["error_tipo"] = type(error).__name__
    traceback.print_exc()
finally:
    r["fin"] = datetime.now().astimezone().isoformat()
    r["lanzador_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "# E000 — Reproducción exploratoria de z501, k=5\n\n"
    report += "## Estado\n" + r["estado"] + "\n\n## Configuración y resultados\n```json\n"
    report += json.dumps(r, ensure_ascii=False, indent=2) + "\n```\n\n"
    report += ("## Alcance e interpretación\nSe conserva el cálculo de z501 y se agregan exportaciones. "
               "No se imputa, corrige ni filtra de nuevo el dataset. k=5 es una referencia, no un óptimo. "
               "La representación está guiada por un RF supervisado, no es agnóstica al target. "
               "Los lifts describen uso de variables en últimos cortes del bosque, no riesgo relativo de baja. "
               "Las medias mensuales reflejan conjuntos de clientes presentes que pueden cambiar; "
               "no demuestran trayectorias individuales. El OOB se calcula por filas, no con separación por cliente. "
               "Esta corrida no mide estabilidad ni demuestra causalidad o utilidad comercial. "
               "Se conserva el PDF de cátedra: su texto y diseño requieren lectura crítica.\n\n"
               "## Archivos\nVer asignaciones.csv, representacion_rf.npz, tamanos.csv, atributos_rf.csv, "
               "variables.json, muestra_cliente_mes.csv.gz, tendencias.csv y run.log. "
               "Si hubo un error, algunas salidas pueden faltar o estar incompletas. "
               "Asignaciones y muestra contienen IDs: conservar localmente.\n")
    (OUT / "informe.md").write_text(report, encoding="utf-8")
    print("\n=== RESULTADO PARA COMPARTIR ===", flush=True)
    print("Estado:", r["estado"])
    print("Clientes por cluster:", r["clientes_por_cluster"])
    if "error_tipo" in r:
        print("Error:", r["error_tipo"], "(detalle en run.log)")
    print("Carpeta:", OUT)
raise SystemExit(0 if r["estado"].startswith("EJECUCION_COMPLETADA") else 1)
