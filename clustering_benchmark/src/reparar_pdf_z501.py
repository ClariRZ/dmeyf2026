"""Regenera el PDF desde exportaciones de z501; no carga datos crudos ni entrena.

Uso: python reparar_pdf_z501.py --corrida /ruta/corrida --salida /ruta/nueva
La salida debe ser una carpeta nueva (puede contener este script y run.log).
No modifica resultados previos. Sólo requiere numpy, pandas y matplotlib.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUTS = ("resultado.json", "variables.json", "asignaciones.csv",
          "representacion_rf.npz", "atributos_rf.csv", "tendencias.csv")


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cargar(run: Path) -> dict:
    """Lee exclusivamente las salidas, alinea por ID y preserva sus valores."""
    meta = json.loads((run / "resultado.json").read_text(encoding="utf-8"))
    features = json.loads((run / "variables.json").read_text(encoding="utf-8"))
    if not features or len(features) != len(set(features)):
        raise ValueError("Lista de variables vacía o repetida")
    labels = pd.read_csv(run / "asignaciones.csv", index_col="numero_de_cliente")["cluster"]
    if labels.index.has_duplicates or labels.isna().any():
        raise ValueError("Asignaciones incompletas o IDs repetidos")
    with np.load(run / "representacion_rf.npz", allow_pickle=False) as z:
        P = pd.DataFrame(z["X"], index=z["ids"], columns=z["variables"].astype(str))
    if P.index.has_duplicates or P.columns.has_duplicates:
        raise ValueError("Ejes repetidos en la matriz RF")
    if len(P) != len(labels) or set(P.index) != set(labels.index):
        raise ValueError("La matriz y las asignaciones no cubren los mismos IDs")
    if not np.isfinite(P.to_numpy()).all():
        raise ValueError("La representación RF contiene valores no finitos")
    labels = labels.reindex(P.index)
    counts = labels.value_counts().sort_index()
    counts_dict = {str(int(k)): int(v) for k, v in counts.items()}
    if counts_dict != meta.get("clientes_por_cluster"):
        raise ValueError("Los tamaños no coinciden con resultado.json")
    # Es el mismo cociente descriptivo de z501, NO una estimación de riesgo.
    global_share = P.mean(axis=0)
    lift = P.groupby(labels).mean().div(global_share.replace(0, np.nan), axis=1)
    attrs = pd.read_csv(run / "atributos_rf.csv", float_precision="round_trip")
    required = {"atributo", "cluster", "rank", "lift"}
    if not required.issubset(attrs.columns):
        raise ValueError("Faltan columnas en atributos_rf.csv")
    first = attrs.sort_values(["cluster", "rank"])["atributo"].unique().tolist()
    if set(first) - set(features):
        raise ValueError("Hay atributos ajenos a variables.json")
    ordered = first + sorted((f for f in features if f not in first), key=str.lower)
    trend = pd.read_csv(run / "tendencias.csv", header=[0, 1], index_col=[0, 1],
                        float_precision="round_trip")
    trend.index = pd.MultiIndex.from_tuples(
        [(int(c), int(m)) for c, m in trend.index], names=["cluster", "foto_mes"])
    if trend.index.has_duplicates or trend.columns.has_duplicates:
        raise ValueError("Ejes repetidos en tendencias.csv")
    if set(trend.index.get_level_values(0)) != set(counts.index):
        raise ValueError("Las tendencias no incluyen los mismos clusters")
    for field in ("centro", "lo", "hi", "n"):
        if field not in trend.columns.get_level_values(0) or set(trend[field].columns) != set(features):
            raise ValueError("Tendencias incompletas: " + field)
    return dict(meta=meta, features=ordered, counts=counts_dict, lift=lift,
                share=global_share, attrs=attrs, trend=trend,
                months=sorted(trend.index.get_level_values(1).unique()),
                clusters=sorted(int(c) for c in counts.index))


def describir(attr: str, data: dict) -> tuple[str, str]:
    """Nunca busca un máximo sobre una serie vacía o completamente NA."""
    if attr not in data["lift"].columns:
        return ("Sin participación en los últimos cortes de la muestra del bosque.", "sin_participacion")
    s = data["lift"][attr]
    valid = s[np.isfinite(s.to_numpy(dtype=float))]
    if valid.empty:
        reason = ("Frecuencia global cero entre BAJA+2." if data["share"].get(attr) == 0
                  else "No hay valores finitos para comparar clusters.")
        return ("Lift no calculable. " + reason, "lift_no_calculable")
    top = data["attrs"].loc[data["attrs"]["atributo"] == attr]
    top = top.loc[np.isfinite(top["lift"].to_numpy(dtype=float))]
    if not top.empty:
        detail = "; ".join(f"C{int(r.cluster)}: {float(r.lift):.2f}" for r in top.itertuples())
        return ("Entre los atributos destacados del bosque. Lift de últimos cortes: " + detail + ".", "top")
    best = valid.idxmax()
    return (f"Fuera del top seleccionado. Mayor lift de últimos cortes: C{int(best)}, {float(valid.loc[best]):.2f}.", "otro")


def render(data: dict, path: Path) -> tuple[int, list[dict]]:
    """Grafica centros y bandas exportados. No recalcula tendencias ni grupos."""
    params = data["meta"].get("parametros", {})
    band = params.get("banda", "no registrada")
    bands = {"ic95": "media; banda normal aproximada del 95%, recortada por z501",
             "desvio": "media; banda de un desvío estándar, recortada por z501",
             "iqr": "mediana; banda Q25-Q75, recortada por z501"}
    explanation = bands.get(band, "centro y banda ya exportados")
    issues = []
    with path.open("xb") as handle, PdfPages(handle) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        try:
            fig.text(.07, .94, "E000 | RF + K-means", fontsize=22, weight="bold", va="top")
            fig.text(.07, .875, "PDF recuperado sin volver a entrenar", fontsize=14, va="top")
            total = sum(data["counts"].values())
            paragraphs = [
                f"{total:,} clientes agrupados. " + "; ".join(f"C{k}: {v:,}" for k, v in data["counts"].items()) + ".",
                f"Parámetros registrados: k={params.get('k')}; seed={params.get('seed')}; "
                f"árboles={params.get('n_trees')}; mínimo por hoja={params.get('min_samples_leaf')}.",
                "Se leen asignaciones, representación RF, atributos y tendencias guardados antes del fallo. "
                "No se leen los datos bancarios originales ni se modifican los clusters.",
                "El lift describe la frecuencia de uso de una variable en los últimos cortes del bosque, "
                "no el riesgo de baja ni el nivel de consumo. Los lifts no calculables se conservan como tales.",
                "Las líneas representan " + explanation + ". Son estadísticas por mes calendario de los "
                "clientes con datos disponibles; la composición puede cambiar entre meses.",
                "La tabla n cuenta observaciones con dato no faltante para cada variable, no necesariamente "
                "todos los integrantes del cluster. Los huecos no se rellenan como actividad cero.",
                "Cinco clusters es una configuración de referencia, no un número óptimo demostrado. "
                "Esta salida no evalúa estabilidad, causalidad ni utilidad comercial. El OOB original es "
                "por filas y no constituye validación con clientes separados.",
            ]
            y = .80
            for text in paragraphs:
                wrapped = textwrap.fill(text, 112)
                fig.text(.07, y, wrapped, fontsize=11, va="top", linespacing=1.4)
                y -= .029 * (wrapped.count("\n") + 1) + .024
            fig.text(.07, .045, "Se conserva el informe del fallo original. Esta recuperación sólo completa la salida gráfica.", fontsize=9)
            pdf.savefig(fig)
        finally:
            plt.close(fig)
        for number, attr in enumerate(data["features"], 2):
            role, status = describir(attr, data)
            if status in ("sin_participacion", "lift_no_calculable"):
                issues.append({"atributo": attr, "estado": status, "detalle": role})
            fig = plt.figure(figsize=(11.69, 8.27))
            try:
                fig.text(.07, .966, f"E000 | RF + K-means | {number}/{len(data['features']) + 1}", fontsize=9, va="top")
                fig.text(.07, .925, textwrap.fill(attr, 86), fontsize=15, weight="bold", va="top")
                fig.text(.07, .85, textwrap.fill(role, 125), fontsize=10, va="top")
                ax = fig.add_axes([.09, .34, .86, .41])
                for c in data["clusters"]:
                    s = data["trend"].xs(c, level=0).reindex(data["months"])
                    center, low, high = (s[field][attr].to_numpy(dtype=float) for field in ("centro", "lo", "hi"))
                    x = np.arange(len(data["months"]))
                    line, = ax.plot(x, center, marker="o", linewidth=1.8, label=f"C{c}")
                    ax.fill_between(x, low, high, alpha=.10, color=line.get_color())
                ax.set_xticks(np.arange(len(data["months"])), [str(m) for m in data["months"]])
                ax.set_ylabel("Valor en unidades originales")
                ax.set_xlabel("Mes calendario")
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(frameon=False, ncols=min(5, len(data["clusters"])), fontsize=9)
                ax.grid(axis="y", alpha=.20)
                if not np.isfinite(data["trend"]["centro"][attr].to_numpy(dtype=float)).any():
                    ax.text(.5, .5, "Sin centro calculable en los datos exportados", ha="center", transform=ax.transAxes)
                n = data["trend"]["n"][attr].unstack("foto_mes").reindex(index=data["clusters"], columns=data["months"])
                cells = [[str(int(v)) if pd.notna(v) else "—" for v in row] for row in n.to_numpy()]
                table_ax = fig.add_axes([.09, .10, .86, .15])
                table_ax.axis("off")
                table = table_ax.table(cellText=cells, rowLabels=[f"C{c}" for c in data["clusters"]],
                                       colLabels=[str(m) for m in data["months"]], loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.1)
                table_ax.set_title("n con dato no faltante para este atributo", loc="left", fontsize=10, pad=6)
                fig.text(.07, .036, "Centro/banda: " + explanation + ". No se imputan faltantes.", fontsize=8.5)
                pdf.savefig(fig)
            finally:
                plt.close(fig)
        return pdf.get_pagecount(), issues


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corrida", type=Path, required=True)
    p.add_argument("--salida", type=Path)
    args = p.parse_args()
    run = args.corrida.expanduser().resolve(strict=True)
    out = args.salida.expanduser().resolve() if args.salida else Path(tempfile.mkdtemp(prefix="pdf_reparado_", dir=run))
    out.mkdir(parents=True, exist_ok=True)
    if out == run or any((out / f).exists() for f in ("reparacion.json", "informe_reparacion.md", "clusters_tendencias.pdf", "clusters_tendencias.parcial.pdf")):
        raise SystemExit("Usá una subcarpeta nueva para no sobrescribir resultados.")
    result = {"estado": "REPARACION_NO_COMPLETADA", "inicio": datetime.now().astimezone().isoformat(),
              "reentrenamiento": False, "version": "z501_pdf_v1", "pasos": [], "python": sys.version}
    try:
        result["versiones"] = {x: version(x) for x in ("numpy", "pandas", "matplotlib")}
        result["script_sha256"] = digest(Path(__file__))
        if Path(__file__).resolve() != (out / "reparar_pdf_z501.py").resolve():
            shutil.copyfile(__file__, out / "reparar_pdf_z501.py")
        before = {name: digest(run / name) for name in INPUTS}
        result["huellas_entrada"] = before
        result["pasos"].append("Leer exportaciones y alinear representación y asignaciones por ID.")
        print("Leyendo resultados guardados; no se carga el CSV bancario.", flush=True)
        data = cargar(run)
        result["clientes_por_cluster"] = data["counts"]
        result["pasos"].append("Regenerar PDF con los centros y bandas existentes; declarar lifts no calculables.")
        partial = out / "clusters_tendencias.parcial.pdf"
        print(f"Generando {len(data['features']) + 1} páginas sin entrenar.", flush=True)
        pages, issues = render(data, partial)
        if pages != len(data["features"]) + 1:
            raise RuntimeError("Número de páginas incompleto")
        if before != {name: digest(run / name) for name in INPUTS}:
            raise RuntimeError("Cambió una entrada durante la generación")
        partial.rename(out / "clusters_tendencias.pdf")
        result.update(estado="PDF_REGENERADO_SIN_REENTRENAR", paginas=pages,
                      atributos_sin_lift=issues, fuentes_sin_cambios=True,
                      pdf_sha256=digest(out / "clusters_tendencias.pdf"))
        result["pasos"].append("Verificar cantidad de páginas y conservación de las entradas por SHA-256.")
    except (Exception, KeyboardInterrupt) as error:
        result["error_tipo"] = type(error).__name__
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        result["fin"] = datetime.now().astimezone().isoformat()
        text = json.dumps(result, ensure_ascii=False, indent=2)
        (out / "reparacion.json").write_text(text, encoding="utf-8")
        report = "# E000 — Recuperación del PDF de z501\n\n"
        report += "No se reentrena ni se cambian asignaciones. El informe y PDF parcial originales se conservan.\n\n"
        report += "Los lifts se reconstruyen desde la matriz exportada; nunca se reemplazan NA por cero para elegir un máximo.\n\n"
        report += "Se aclaran las leyendas: n cuenta datos disponibles; lift no es riesgo de baja. Centro y bandas no se recalculan.\n\n"
        report += "## Pasos y resultados\n```json\n" + text + "\n```\n\n"
        report += "## Límites\nSalida descriptiva, no validación de los clusters. k=5 no es un óptimo demostrado. "
        report += "Los agregados mensuales no prueban trayectorias individuales. La selección RF usa etiquetas. "
        report += "La banda IC95 original es una aproximación normal no validada aquí. "
        report += "Se verifica el conteo generado de páginas; la lectura visual del PDF real sigue pendiente.\n"
        (out / "informe_reparacion.md").write_text(report, encoding="utf-8")
        print("\n=== REPARACION PARA COMPARTIR ===")
        for key in ("estado", "clientes_por_cluster", "paginas", "reentrenamiento", "fuentes_sin_cambios", "error_tipo"):
            if key in result:
                print(f"{key}: {json.dumps(result[key], ensure_ascii=False)}")
        print("Atributos sin lift calculable:", [x["atributo"] for x in result.get("atributos_sin_lift", []) if x["estado"] == "lift_no_calculable"])
        print("Carpeta:", out)
        if result["estado"] == "PDF_REGENERADO_SIN_REENTRENAR":
            print("PDF:", out / "clusters_tendencias.pdf")
    return 0 if result["estado"] == "PDF_REGENERADO_SIN_REENTRENAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
