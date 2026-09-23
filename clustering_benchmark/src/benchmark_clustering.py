#!/usr/bin/env python3
"""
Benchmark inicial de clustering para DMEyF 2026.

Esta primera versión implementa la pista A (foto transversal numérica).
No decide cuál es el "mejor" clustering: produce una tabla comparable
de métricas, tamaños, ruido y tiempo de cómputo.

Ejemplo:
    python clustering_benchmark/src/benchmark_clustering.py \
        --data competencia_01.parquet \
        --month 202103 \
        --target BAJA+2 \
        --max-n 12000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    HDBSCAN,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


META = {"numero_de_cliente", "foto_mes", "clase_ternaria"}


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path.suffix}")


def prepare_numeric(df: pd.DataFrame, month: int, target: str, max_n: int, seed: int):
    d = df.loc[
        (df["foto_mes"] == month) & (df["clase_ternaria"] == target)
    ].copy()

    numeric = [
        c for c in d.select_dtypes(include=[np.number]).columns
        if c not in META and not c.startswith("target_")
    ]
    X = d[numeric].replace([np.inf, -np.inf], np.nan)

    # Se usa RobustScaler para reducir la dominancia de colas extremas.
    prep = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RobustScaler(),
    )
    Xp = prep.fit_transform(X)

    if len(d) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(d), size=max_n, replace=False)
        Xp = Xp[idx]
        d = d.iloc[idx].copy()

    return Xp, d, numeric


def build_models(k: int, seed: int):
    return {
        "kmeans": KMeans(n_clusters=k, n_init=20, random_state=seed),
        "minibatch_kmeans": MiniBatchKMeans(n_clusters=k, n_init=10, random_state=seed),
        "agg_ward": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "agg_average": AgglomerativeClustering(n_clusters=k, linkage="average"),
        "agg_complete": AgglomerativeClustering(n_clusters=k, linkage="complete"),
        "gmm": GaussianMixture(n_components=k, covariance_type="full", random_state=seed),
        "birch": Birch(n_clusters=k),
        "dbscan": DBSCAN(eps=1.0, min_samples=20),
        "hdbscan": HDBSCAN(min_cluster_size=30),
        "optics": OPTICS(min_samples=20),
        "spectral": SpectralClustering(
            n_clusters=k,
            assign_labels="kmeans",
            random_state=seed,
            affinity="nearest_neighbors",
        ),
        "meanshift": MeanShift(),
        "affinity_propagation": AffinityPropagation(random_state=seed),
    }


def fit_predict(model, X):
    if isinstance(model, GaussianMixture):
        return model.fit_predict(X)
    return model.fit_predict(X)


def safe_metrics(X, labels, metric_sample=4000, seed=17):
    labels = np.asarray(labels)
    keep = labels != -1
    labels_eval = labels[keep]
    X_eval = X[keep]

    unique = np.unique(labels_eval)
    if len(unique) < 2 or len(X_eval) <= len(unique):
        return np.nan, np.nan, np.nan

    if len(X_eval) > metric_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_eval), metric_sample, replace=False)
        Xs, ys = X_eval[idx], labels_eval[idx]
    else:
        Xs, ys = X_eval, labels_eval

    return (
        silhouette_score(Xs, ys),
        calinski_harabasz_score(Xs, ys),
        davies_bouldin_score(Xs, ys),
    )


def run_benchmark(X, k: int, seed: int):
    rows = []
    for name, model in build_models(k, seed).items():
        # Métodos con costo alto reciben una muestra más pequeña.
        cap = 5000 if name in {
            "agg_ward", "agg_average", "agg_complete",
            "spectral", "meanshift", "affinity_propagation"
        } else len(X)
        Xi = X[:cap]

        t0 = time.perf_counter()
        try:
            labels = fit_predict(model, Xi)
            elapsed = time.perf_counter() - t0
            sil, ch, db = safe_metrics(Xi, labels, seed=seed)
            labs, counts = np.unique(labels, return_counts=True)
            n_noise = int(counts[labs == -1][0]) if -1 in labs else 0
            n_clusters = int(np.sum(labs != -1))
            valid_counts = counts[labs != -1]
            imbalance = (
                float(valid_counts.max() / valid_counts.min())
                if len(valid_counts) > 1 and valid_counts.min() > 0
                else np.nan
            )
            rows.append({
                "method": name,
                "n_used": len(Xi),
                "n_clusters": n_clusters,
                "noise_pct": n_noise / len(Xi),
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "size_ratio_max_min": imbalance,
                "seconds": elapsed,
                "status": "ok",
            })
        except Exception as e:
            rows.append({
                "method": name,
                "n_used": len(Xi),
                "n_clusters": np.nan,
                "noise_pct": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "size_ratio_max_min": np.nan,
                "seconds": time.perf_counter() - t0,
                "status": f"ERROR: {type(e).__name__}: {e}",
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--month", type=int, default=202103)
    p.add_argument("--target", default="BAJA+2")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-n", type=int, default=12000)
    p.add_argument("--out", default="clustering_benchmark/resultados")
    args = p.parse_args()

    df = load_data(Path(args.data))
    X, sample, numeric = prepare_numeric(
        df, args.month, args.target, args.max_n, args.seed
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(X, args.k, args.seed)
    results.to_csv(out / f"benchmark_{args.month}_{args.target}.csv", index=False)
    pd.Series(numeric, name="variable").to_csv(out / "variables_numericas.csv", index=False)

    print(results.sort_values("silhouette", ascending=False).to_string(index=False))
    print(f"\nClientes usados: {len(sample):,}")
    print(f"Variables numéricas originales: {len(numeric)}")
    print(f"Salida: {out.resolve()}")


if __name__ == "__main__":
    main()
