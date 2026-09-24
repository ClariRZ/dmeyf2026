"""Comparacion de seis metodos sobre la MISMA matriz RF de z501.
No entrena el bosque, no lee datos bancarios crudos y no modifica E000.
Crea una carpeta nueva dentro de --salida, con un informe por configuracion.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, pairwise_distances, silhouette_score)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

LIMITES = """La representacion esta guiada por un bosque SUPERVISADO y retrospectivo.
Se mantienen sus proporciones de ultimos cortes: no son valores bancarios ni riesgos de baja.
No se cambia el RF, no se imputa, escala, selecciona variables ni reduce dimensiones.
La concordancia con E000 es semejanza, NO exactitud contra una verdad externa.
El ARI de todas las filas trata el ruido -1 como una etiqueta comun; se informa por separado
el ARI sobre clientes asignados por ambos metodos y su denominador.
Silhouette, CH y DB usan solo asignados y geometria euclidea; no son un ranking universal,
y las metricas condicionadas de HDBSCAN no tienen el mismo denominador que las de otros metodos.
GMM diagonal es una aproximacion: estas proporciones son composicionales, no Gaussianas independientes.
BIC y AIC solo se comparan entre GMM en la misma matriz, no con otros algoritmos.
La variacion entre semillas mide inicializacion CON RF FIJO, no estabilidad del pipeline completo.
No se mide aun estabilidad por submuestras, mes, otra representacion ni utilidad comercial.
No se adjudican nombres semanticos ni se selecciona un ganador automatico.
"""
FUENTES = [
    'https://scikit-learn.org/stable/modules/clustering.html',
    'https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html',
    'https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html',
]


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path: Path, value: dict | list) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load_input(source: Path):
    with np.load(source / 'representacion_rf.npz', allow_pickle=False) as z:
        X = np.array(z['X'], dtype=np.float64, copy=True)
        ids = z['ids'].copy()
        features = [str(v) for v in z['variables']]
    if X.ndim != 2 or len(X) < 3 or X.shape[1] < 1:
        raise ValueError('Matriz vacia o con dimensiones no validas.')
    if ids.ndim != 1 or len(ids) != len(X) or ids.dtype.kind not in 'iu':
        raise ValueError('IDs no enteros o no alineados con la matriz.')
    if len(np.unique(ids)) != len(ids) or len(features) != X.shape[1]:
        raise ValueError('IDs duplicados o variables no alineadas.')
    if len(set(features)) != len(features):
        raise ValueError('Nombres de variables duplicados.')
    if not np.isfinite(X).all() or (X < 0).any() or not np.allclose(X.sum(1), 1, atol=1e-8, rtol=1e-8):
        raise ValueError('La matriz no contiene proporciones finitas no negativas que sumen uno.')
    a = pd.read_csv(source / 'asignaciones.csv', dtype={'numero_de_cliente': 'int64', 'cluster': 'int64'})
    if a['numero_de_cliente'].duplicated().any() or set(a['numero_de_cliente']) != set(ids.tolist()):
        raise ValueError('Asignaciones e IDs de la matriz no coinciden exactamente.')
    original = a.set_index('numero_de_cliente').loc[ids, 'cluster'].to_numpy()
    if (original < 0).any():
        raise ValueError('E000 contiene asignaciones negativas inesperadas.')
    return X, ids, features, original


def plan(ks: list[int], seeds: list[int], sizes: list[int]) -> list[dict]:
    jobs = []
    for family in ['KMeans', 'Ward', 'Average', 'Complete', 'GMM_diag']:
        for k in ks:
            for seed in (seeds if family in ['KMeans', 'GMM_diag'] else [None]):
                jobs.append({'metodo': family, 'k': k, 'semilla': seed})
    for size in sizes:
        jobs.append({'metodo': 'HDBSCAN', 'min_cluster_size': size, 'min_samples': 20})
    return jobs


def model_for(job: dict):
    name = job['metodo']
    if name == 'KMeans':
        return KMeans(n_clusters=job['k'], n_init=20, random_state=job['semilla'],
                      max_iter=300, tol=1e-4, algorithm='lloyd')
    if name in ['Ward', 'Average', 'Complete']:
        return AgglomerativeClustering(n_clusters=job['k'], linkage=name.lower(), metric='euclidean')
    if name == 'GMM_diag':
        return GaussianMixture(n_components=job['k'], covariance_type='diag', reg_covar=1e-6,
                               n_init=3, max_iter=300, tol=1e-3, random_state=job['semilla'])
    return HDBSCAN(min_cluster_size=job['min_cluster_size'], min_samples=job['min_samples'],
                   metric='euclidean', cluster_selection_method='eom', allow_single_cluster=False,
                   n_jobs=2, copy=True)


def ari_assigned(a: np.ndarray, b: np.ndarray):
    mask = (a >= 0) & (b >= 0)
    n = int(mask.sum())
    if n < 3 or min(len(np.unique(a[mask])), len(np.unique(b[mask]))) < 2:
        return None, n
    return float(adjusted_rand_score(a[mask], b[mask])), n


def evaluate(X: np.ndarray, D: np.ndarray, labels: np.ndarray, original: np.ndarray):
    keep = labels >= 0
    y = labels[keep]
    labs, counts = np.unique(y, return_counts=True)
    assigned_ari, n_ari = ari_assigned(original, labels)
    r = {'n_total': len(labels), 'n_asignados': int(keep.sum()), 'clusters_observados': len(labs),
         'ruido_pct': float(100 * (~keep).mean()), 'tamano_min': int(counts.min()) if len(counts) else 0,
         'tamano_max': int(counts.max()) if len(counts) else 0,
         'ari_E000_todos': float(adjusted_rand_score(original, labels)),
         'ari_E000_asignados': assigned_ari, 'n_ari_asignados': n_ari,
         'silhouette': None, 'calinski_harabasz': None, 'davies_bouldin': None, 'errores_metricas': {}}
    if not 2 <= len(labs) < len(y):
        r['estado_metricas'] = 'NO_DEFINIDAS: se requieren entre 2 y n-1 grupos asignados'
        return r
    subD = D if keep.all() else D[np.ix_(keep, keep)]
    subX = X if keep.all() else X[keep]
    funcs = {'silhouette': lambda: silhouette_score(subD, y, metric='precomputed'),
             'calinski_harabasz': lambda: calinski_harabasz_score(subX, y),
             'davies_bouldin': lambda: davies_bouldin_score(subX, y)}
    for key, fn in funcs.items():
        try:
            val = float(fn())
            if not np.isfinite(val):
                raise ValueError('Resultado no finito')
            r[key] = val
        except Exception as error:
            r['errores_metricas'][key] = f'{type(error).__name__}: {error}'
    r['estado_metricas'] = 'PARCIALES' if r['errores_metricas'] else 'CALCULADAS_EN_ASIGNADOS'
    return r


def export_run(folder: Path, X, ids, features, labels, original):
    pd.DataFrame({'numero_de_cliente': ids, 'cluster': labels}).to_csv(folder / 'asignaciones.csv', index=False)
    labs, counts = np.unique(labels, return_counts=True)
    pd.DataFrame({'cluster': labs, 'n': counts, 'porcentaje_total': counts * 100 / len(ids)}).to_csv(
        folder / 'tamanos.csv', index=False)
    pd.crosstab(pd.Series(original, name='E000'), pd.Series(labels, name='nuevo')).to_csv(folder / 'cruce_E000.csv')
    rows = []
    for lab in labs:
        if lab < 0:
            continue
        mask = labels == lab
        mean = X[mask].mean(0)
        rest = X[~mask].mean(0) if (~mask).any() else np.full(X.shape[1], np.nan)
        lift = np.divide(mean, rest, out=np.full(X.shape[1], np.nan), where=rest > 0)
        rows.append(pd.DataFrame({'cluster': lab, 'variable': features, 'share_medio': mean,
                                  'share_resto': rest, 'lift_vs_resto': lift}))
    columns = ['cluster', 'variable', 'share_medio', 'share_resto', 'lift_vs_resto']
    (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)).to_csv(
        folder / 'perfiles_representacion_RF.csv', index=False)


def report_run(folder: Path, r: dict) -> None:
    save_json(folder / 'resultado.json', r)
    text = '# ' + r['id'] + '\n\n## Pregunta\n'
    text += 'Que cambia al modificar el agrupamiento sobre una representacion RF fija?\n\n'
    text += '## Entrada y decisiones previas\nVer configuracion.json y manifiesto_entrada.json del lote.\n\n'
    text += '## Pasos ejecutados, resultado y advertencias\n```json\n'
    text += json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + '\n```\n\n'
    text += '## Interpretacion permitida\nTamanos, separacion y concordancia condicionadas a esta matriz y configuracion.\n\n'
    text += '## Lo que NO demuestra y limitaciones\n' + LIMITES
    text += '\n## Archivos\nAsignaciones privadas, tamanos, cruce con E000 y perfiles de la representacion. '
    text += 'Pueden faltar exportaciones si el estado registra un error. Los perfiles no son valores bancarios.\n'
    (folder / 'informe.md').write_text(text, encoding='utf-8')


def execute(source: Path, parent: Path, ks: list[int], seeds: list[int], sizes: list[int], max_n: int = 6000):
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='comparacion_RF_', dir=parent))
    out.chmod(0o700)
    (out / 'codigo_ejecutado.py').write_bytes(Path(__file__).read_bytes())
    print('Salida:', out, flush=True)
    try:
        paths = [source / 'representacion_rf.npz', source / 'asignaciones.csv']
        fingerprints = {p.name: sha(p) for p in paths}
        X, ids, features, original = load_input(source)
        if len(X) > max_n:
            raise ValueError('Se supera el limite de matriz completa. No se submuestrea silenciosamente.')
        if any(k < 2 or k >= len(X) for k in ks) or any(s < 2 or s > len(X) for s in sizes) or len(X) < 20:
            raise ValueError('Configuraciones incompatibles con el numero de filas.')
        jobs = plan(ks, seeds, sizes)
        manifest = {'fuente': str(source), 'huellas': fingerprints, 'n': len(X), 'p': X.shape[1],
                    'variables': features, 'columnas_constantes': int((np.ptp(X, axis=0) == 0).sum()),
                    'python': sys.version, 'versiones': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'threadpoolctl']},
                    'script_sha256': sha(Path(__file__)), 'inicio': datetime.now().astimezone().isoformat()}
        save_json(out / 'manifiesto_entrada.json', manifest)
        save_json(out / 'plan_previo.json', {'experimentos': jobs, 'limites': LIMITES,
                                           'semillas_solo_inicializacion': True, 'fuentes_metodos': FUENTES})
        # Distancias completas comunes: no cambian las filas ni el denominador por un muestreo.
        with threadpool_limits(limits=2):
            D = pairwise_distances(X, metric='euclidean', n_jobs=1)
            np.maximum(D, 0, out=D)
            np.fill_diagonal(D, 0)
            label_sets = {'E000_original': original}
            reference = out / 'E000_original'
            reference.mkdir()
            ref = {'id': 'E000_original', 'config': {'metodo': 'KMeans_original'}, 'estado': 'REFERENCIA_CARGADA',
                   'pasos': ['Cargar asignaciones E000 sin reentrenar y evaluar en matriz guardada.'],
                   'metricas': evaluate(X, D, original, original)}
            save_json(reference / 'configuracion.json', ref['config'])
            report_run(reference, ref)
            records = []
            for number, job in enumerate(jobs, 1):
                name = f'RF_{number:03d}_{job["metodo"]}'
                folder = out / name
                folder.mkdir()
                r = {'id': name, 'config': job, 'estado': 'INICIADO', 'pasos': [], 'metricas': {}, 'warnings': []}
                save_json(folder / 'configuracion.json', job)
                print(f'[{number}/{len(jobs)}] {name}: {job}', flush=True)
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        model = model_for(job)
                        save_json(folder / 'parametros_completos.json', model.get_params(deep=False))
                        r['pasos'].append('Ajustar el metodo a todas las filas y columnas de la matriz RF fija.')
                        labels = np.asarray(model.fit_predict(X), dtype=np.int64)
                        if len(labels) != len(ids):
                            raise ValueError('Numero incorrecto de asignaciones.')
                        r['segundos_ajuste'] = time.perf_counter() - start
                        # Guardar antes de evaluar, para no perder un ajuste por un error posterior.
                        pd.DataFrame({'numero_de_cliente': ids, 'cluster': labels}).to_csv(folder / 'asignaciones.csv', index=False)
                        label_sets[name] = labels
                        r['estado'] = 'AJUSTADO_SALIDAS_INCOMPLETAS'
                        r['pasos'].append('Guardar asignaciones; evaluar geometria y concordancia con E000.')
                        r['metricas'] = evaluate(X, D, labels, original)
                        if isinstance(model, GaussianMixture):
                            r['gmm'] = {'convergio': bool(model.converged_), 'iteraciones': int(model.n_iter_)}
                            for key, fn in [('bic', model.bic), ('aic', model.aic)]:
                                v = float(fn(X))
                                r['gmm'][key] = v if np.isfinite(v) else None
                        r['pasos'].append('Exportar tamanos, cruce con E000 y shares de la representacion por cluster.')
                        export_run(folder, X, ids, features, labels, original)
                        r['estado'] = 'AJUSTE_Y_EXPORTACION_COMPLETOS'
                    except Exception as error:
                        r['error'] = f'{type(error).__name__}: {error}'
                        (folder / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
                        if r['estado'] == 'INICIADO':
                            r['estado'] = 'ERROR_AJUSTE_O_CONFIGURACION'
                    r['warnings'] = [f'{w.category.__name__}: {w.message}' for w in caught]
                r['segundos_totales'] = time.perf_counter() - start
                report_run(folder, r)
                records.append(r)
                # Una falla posterior no elimina la tabla parcial ya generada.
                flat = [{**{'id': t['id'], 'estado': t['estado'], **t['config']}, **t['metricas'],
                         'n_warnings': len(t['warnings']), 'gmm_convergio': t.get('gmm', {}).get('convergio')} for t in records]
                pd.DataFrame(flat).to_csv(out / 'resumen_metodos.csv', index=False)
            pairs = []
            for (an, a), (bn, b) in combinations(label_sets.items(), 2):
                val, n = ari_assigned(a, b)
                pairs.append({'a': an, 'b': bn, 'ari_todos_ruido_como_etiqueta': float(adjusted_rand_score(a, b)),
                              'ari_asignados_ambos': val, 'n_asignados_ambos': n})
            pd.DataFrame(pairs).to_csv(out / 'concordancia_entre_corridas.csv', index=False)
        unchanged = all(sha(p) == fingerprints[p.name] for p in paths)
        errors = sum(t['estado'] != 'AJUSTE_Y_EXPORTACION_COMPLETOS' for t in records)
        warnings_n = sum(bool(t['warnings']) or bool(t['metricas'].get('errores_metricas'))
                         or t.get('gmm', {}).get('convergio') is False for t in records)
        state = 'COMPARACION_EJECUTADA_CON_LIMITES'
        if not unchanged or errors:
            state = 'COMPARACION_PARCIAL_REVISAR'
        summary = {'estado': state, 'configuraciones_ejecutadas': len(records), 'errores': errors,
                   'corridas_con_advertencias': warnings_n, 'fuentes_sin_cambios': unchanged,
                   'n_clientes_comunes': len(X), 'fin': datetime.now().astimezone().isoformat(), 'limites': LIMITES}
        save_json(out / 'resultado_lote.json', summary)
        text = '# Comparacion de metodos sobre representacion RF fija\n\n'
        text += '## Pregunta\nQue cambia al modificar algoritmo, k e inicializacion conservando exactamente la entrada?\n\n'
        text += '## Estado\n```json\n' + json.dumps(summary, ensure_ascii=False, indent=2) + '\n```\n\n'
        text += '## Resultados (orden del plan, NO ranking)\n\n| Corrida | Grupos | Ruido % | Silhouette | ARI E000 todos | Estado |\n|---|---:|---:|---:|---:|---|\n'
        for r in records:
            q = r['metricas']
            text += f"| {r['id']} | {q.get('clusters_observados', '')} | {q.get('ruido_pct', '')} | {q.get('silhouette', '')} | {q.get('ari_E000_todos', '')} | {r['estado']} |\n"
        text += '\n## Informes individuales\nCada carpeta RF_* contiene configuracion, pasos, resultados y limites.\n'
        text += '\n## Pendiente\nInterpretar resultados, ampliar validacion por submuestras y representar los valores bancarios originales.\n\n' + LIMITES
        text += '\n## Fuentes metodologicas\n' + '\n'.join(FUENTES) + '\n'
        (out / 'INFORME_COMPARACION_RF.md').write_text(text, encoding='utf-8')
        print('\n=== COMPARACION PARA COMPARTIR ===')
        print(json.dumps({k: v for k, v in summary.items() if k != 'limites'}, ensure_ascii=False))
        columns = ['id', 'k', 'semilla', 'clusters_observados', 'ruido_pct', 'silhouette', 'ari_E000_todos', 'n_warnings', 'estado']
        print(pd.DataFrame(flat).reindex(columns=columns).to_string(index=False))
        print('Informe:', out / 'INFORME_COMPARACION_RF.md')
        return 0 if unchanged and errors == 0 else 1
    except Exception as error:
        (out / 'error_lote.log').write_text(traceback.format_exc(), encoding='utf-8')
        failure = {'estado': 'LOTE_NO_COMPLETADO', 'error': f'{type(error).__name__}: {error}'}
        save_json(out / 'fallo_lote.json', failure)
        print('\n=== COMPARACION PARA COMPARTIR ===\n', json.dumps(failure, ensure_ascii=False))
        print('Salida:', out)
        return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corrida', type=Path, required=True)
    p.add_argument('--salida', type=Path, required=True, help='Carpeta padre; se crea una subcarpeta unica.')
    p.add_argument('--ks', nargs='+', type=int, default=[3, 5, 7])
    p.add_argument('--semillas', nargs='+', type=int, default=[214363, 17, 101])
    p.add_argument('--min-cluster-sizes', nargs='+', type=int, default=[30, 80, 160])
    args = p.parse_args()
    return execute(args.corrida.resolve(), args.salida.resolve(),
                   list(dict.fromkeys(args.ks)), list(dict.fromkeys(args.semillas)),
                   list(dict.fromkeys(args.min_cluster_sizes)))


if __name__ == '__main__':
    raise SystemExit(main())
