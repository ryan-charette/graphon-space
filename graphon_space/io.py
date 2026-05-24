"""Input/output helpers for graphon records and experiment tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .graphon import StepGraphon


def graphon_record(
    graphon: StepGraphon,
    model: str,
    family: str | None = None,
    seed: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    family_name = family or str(graphon.metadata.get("family", "unknown"))
    record = {
        "model_type": model,
        "k": graphon.k,
        "family": family_name,
        "seed": seed,
        "c": json.dumps(graphon.sizes.tolist()),
        "P": json.dumps(graphon.matrix.tolist()),
        "e": graphon.edge_density(),
        "t_triangle": graphon.triangle_density(),
        "t_2star": graphon.two_star_density(),
        "t_tilde": graphon.reduced_two_star_density(),
        "entropy": graphon.entropy(),
    }
    if model in {"triangle", "edge-triangle"}:
        record["t"] = record["t_triangle"]
    elif model in {"two-star", "2star", "edge-2star"}:
        record["t"] = record["t_2star"]
    else:
        raise ValueError(f"unknown model {model!r}")
    record.update(extra)
    return record


def records_to_dataframe(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(list(records))


def graphons_to_dataframe(
    graphons: Iterable[StepGraphon],
    model: str,
    seed: int | None = None,
    **extra: Any,
) -> pd.DataFrame:
    return records_to_dataframe(
        graphon_record(g, model=model, seed=seed, **extra) for g in graphons
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    elif suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError("output path must end in .parquet, .csv, or .json")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError("input path must end in .parquet, .csv, or .json")


def graphon_from_record(record: dict[str, Any]) -> StepGraphon:
    sizes = record.get("c", record.get("sizes"))
    matrix = record.get("P", record.get("matrix"))
    if sizes is None or matrix is None:
        raise KeyError("record must contain either c/P or sizes/matrix fields")
    if isinstance(sizes, str):
        sizes = json.loads(sizes)
    if isinstance(matrix, str):
        matrix = json.loads(matrix)
    return StepGraphon(sizes, matrix, {"family": record.get("family"), "model_type": record.get("model_type")})
