"""Combine and reduce S4 network-measure Feather dataframes.

S5 discovers subject/session S4 outputs, reduces channel-level clustering
coefficients to their channel mean, validates repeated graph-level measures,
and writes one group dataframe for each requested task and frequency band.

Examples
--------
Single-participant S5 processing with selected parameters::

    conda run -n mne python code/study_reliability/S5_combining_conn_dataframe.py --subject ST001 --session 01 --task EyesClosedNoTask --run 01 --band alpha --effective-cycles 12 24 --n-epochs 6 --features modularity clustering_coef global_eff SWP --output-derivative-desc group-connectivity-single-participant-example

Full reproduction using all present defaults, writing to a new output
derivative::

    conda run -n mne python code/study_reliability/S5_combining_conn_dataframe.py --output-derivative-desc group-connectivity-reproduction
"""

from __future__ import annotations

__author__ = "Matthew Ma <khmma@polyu.edu.hk>"
__version__ = "1.1.1"
__date__ = "2025-07-09"
__last_modified__ = "2026-07-21"

import argparse
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from MAIN_CONSTANT import DIR_BIDS_SPEECHTRACKING
from logging_utils import logger, set_log_file
from reliability_constants import (
    RESTING_TASKS,
    SUPPORTED_BAND_PATTERN,
    SUPPORTED_BANDS,
    validate_supported_bands,
)
from reliability_io import (
    atomic_write_feather,
    atomic_write_json,
    normalize_bids_labels,
    sha256_file,
)


DEFAULT_INPUT_DERIVATIVE_DESC = "analysis-network-measures"
DEFAULT_OUTPUT_DERIVATIVE_DESC = "group-connectivity"
DEFAULT_CONNECTIVITY_METHOD = "ciplv"

FEATURES_OF_INTEREST = (
    "modularity",
    "clustering_coef",
    "global_eff",
    "char_path",
    "assortativity",
    "SWP",
)
CHANNEL_LEVEL_FEATURES = ("clustering_coef",)
GRAPH_LEVEL_FEATURES = (
    "modularity",
    "global_eff",
    "char_path",
    "assortativity",
    "SWP",
)
INPUT_METADATA_COLUMNS = (
    "rand_seed",
    "SubjectCode",
    "Session",
    "Task",
    "Band",
    "NEpoch",
    "TotalEpoch",
    "EpochLen",
    "EffCycles",
    "Channel",
    "ThresScheme",
    "ThresValue",
    "NCyclesWavelet",
)
INVARIANT_FILE_COLUMNS = (
    "SubjectCode",
    "Session",
    "Task",
    "Band",
    "NEpoch",
    "TotalEpoch",
    "EpochLen",
    "EffCycles",
    "ThresScheme",
    "NCyclesWavelet",
)
REDUCED_METADATA_COLUMNS = (
    "rand_seed",
    "SubjectCode",
    "Session",
    "Task",
    "Run",
    "Band",
    "NEpoch",
    "TotalEpoch",
    "EpochLen",
    "EffCycles",
    "ThresScheme",
    "ThresValue",
    "NCyclesWavelet",
)

NETWORK_MEASURE_FEATHER_PATTERN = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_"
    r"task-(?P<task>Eyes(?:Open|Closed)NoTask)_run-(?P<run>[^_]+)_"
    rf"desc-filt-(?P<band>{SUPPORTED_BAND_PATTERN})-"
    r"effcycles-(?P<effective_cycles>\d+)-"
    r"epochlen-(?P<epoch_len>\d+)ms-npo(?P<n_epochs>\d+)_"
    r"con\.feather"
)


def _parse_network_measure_path(path: str | Path) -> dict:
    """Parse and validate entities from one canonical S4 Feather path."""
    path = Path(path)
    match = NETWORK_MEASURE_FEATHER_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected S4 network-measure filename: {path.name}")

    labels = match.groupdict()
    labels["effective_cycles"] = int(labels["effective_cycles"])
    labels["epoch_len"] = int(labels["epoch_len"])
    labels["n_epochs"] = int(labels["n_epochs"])
    if path.parent.name != f"ses-{labels['session']}":
        raise ValueError(
            f"Session folder and filename disagree for S4 input: {path}"
        )
    if path.parent.parent.name != f"sub-{labels['subject']}":
        raise ValueError(
            f"Subject folder and filename disagree for S4 input: {path}"
        )
    return labels


def iter_network_measure_files(
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    input_derivative_desc: str = DEFAULT_INPUT_DERIVATIVE_DESC,
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
    bands: Optional[Iterable[str]] = None,
    effective_cycles: Optional[Iterable[int]] = None,
    n_epochs_values: Optional[Iterable[int]] = None,
) -> Iterable[tuple[Path, dict]]:
    """Yield matching S4 Feather files and parsed labels in stable order."""
    input_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / input_derivative_desc
    )
    if not input_root.is_dir():
        raise FileNotFoundError(f"S5 input derivative does not exist: {input_root}")

    subject_filter = normalize_bids_labels(subjects, "sub-")
    session_filter = normalize_bids_labels(sessions, "ses-")
    task_filter = normalize_bids_labels(tasks, "task-")
    run_filter = normalize_bids_labels(runs, "run-")
    band_filter = normalize_bids_labels(bands)
    cycle_filter = (
        {int(value) for value in effective_cycles}
        if effective_cycles is not None
        else None
    )
    n_epochs_filter = (
        {int(value) for value in n_epochs_values}
        if n_epochs_values is not None
        else None
    )

    for path in sorted(input_root.glob("sub-*/ses-*/*_con.feather")):
        if NETWORK_MEASURE_FEATHER_PATTERN.fullmatch(path.name) is None:
            continue
        labels = _parse_network_measure_path(path)
        if subject_filter is not None and labels["subject"] not in subject_filter:
            continue
        if session_filter is not None and labels["session"] not in session_filter:
            continue
        if task_filter is not None and labels["task"] not in task_filter:
            continue
        if run_filter is not None and labels["run"] not in run_filter:
            continue
        if band_filter is not None and labels["band"] not in band_filter:
            continue
        if (
            cycle_filter is not None
            and labels["effective_cycles"] not in cycle_filter
        ):
            continue
        if n_epochs_filter is not None and labels["n_epochs"] not in n_epochs_filter:
            continue
        yield path, labels


def _single_value(dataframe: pd.DataFrame, column: str, context: str):
    """Return one repeated value or fail when a column is inconsistent."""
    values = dataframe[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Expected one repeated {column} value in {context}; "
            f"found {len(values)}"
        )
    return values.iloc[0]


def validate_network_measure_dataframe(
    dataframe: pd.DataFrame,
    features: Iterable[str] = FEATURES_OF_INTEREST,
    labels: Optional[dict] = None,
    source_path: Optional[str | Path] = None,
) -> None:
    """Validate one S4 dataframe before channel reduction."""
    features = tuple(features)
    context = str(source_path) if source_path is not None else "S4 dataframe"
    if dataframe.empty:
        raise ValueError(f"S4 dataframe is empty: {context}")

    required_columns = set(INPUT_METADATA_COLUMNS) | set(features)
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"S4 dataframe is missing required columns in {context}: "
            f"{missing_columns}"
        )
    unknown_features = sorted(set(features) - set(FEATURES_OF_INTEREST))
    if unknown_features:
        raise ValueError(f"Unsupported S5 features: {unknown_features}")

    null_identifier_columns = [
        column
        for column in INPUT_METADATA_COLUMNS
        if dataframe[column].isna().any()
    ]
    if null_identifier_columns:
        raise ValueError(
            f"Null identifier values in {context}: {null_identifier_columns}"
        )

    for feature in features:
        if not pd.api.types.is_numeric_dtype(dataframe[feature]):
            raise TypeError(f"S5 feature {feature} is not numeric in {context}")

    for column in INVARIANT_FILE_COLUMNS:
        _single_value(dataframe, column, context)

    duplicate_channel_rows = dataframe.duplicated(
        subset=["rand_seed", "ThresValue", "Channel"],
        keep=False,
    )
    if duplicate_channel_rows.any():
        raise ValueError(
            f"Duplicate channel rows within a draw-threshold group in {context}"
        )

    selected_graph_features = [
        feature for feature in features if feature in GRAPH_LEVEL_FEATURES
    ]
    if selected_graph_features:
        for (rand_seed, threshold), group in dataframe.groupby(
            ["rand_seed", "ThresValue"],
            sort=False,
            dropna=False,
        ):
            group_context = (
                f"{context}, rand_seed={rand_seed}, ThresValue={threshold}"
            )
            for feature in selected_graph_features:
                _single_value(group, feature, group_context)

    if labels is None:
        return

    expected_text = {
        "SubjectCode": labels["subject"],
        "Session": labels["session"],
        "Task": labels["task"],
        "Band": labels["band"],
    }
    prefixes = {"SubjectCode": "sub-", "Session": "ses-"}
    for column, expected in expected_text.items():
        actual = str(_single_value(dataframe, column, context)).removeprefix(
            prefixes.get(column, "")
        )
        if actual != str(expected):
            raise ValueError(
                f"Filename/dataframe {column} mismatch in {context}: "
                f"expected {expected}, found {actual}"
            )

    expected_numeric = {
        "NEpoch": labels["n_epochs"],
        "EpochLen": labels["epoch_len"],
        "EffCycles": labels["effective_cycles"],
    }
    for column, expected in expected_numeric.items():
        actual = _single_value(dataframe, column, context)
        try:
            matches = int(actual) == int(expected)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"Filename/dataframe {column} mismatch in {context}: "
                f"expected {expected}, found {actual}"
            )


def clean_df_conn(
    df: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    run: Optional[str] = None,
    labels: Optional[dict] = None,
    source_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Reduce one S4 channel-level dataframe to graph-level analysis rows.

    The original one-argument call remains supported. Clustering coefficient
    is averaged over channels. Graph-level features are validated as repeated
    values and retained once. Channel is intentionally omitted because a
    reduced row no longer represents one channel.
    """
    selected_features = tuple(
        dict.fromkeys(FEATURES_OF_INTEREST if features is None else features)
    )
    validate_network_measure_dataframe(
        df,
        features=selected_features,
        labels=labels,
        source_path=source_path,
    )

    working = df[list(INPUT_METADATA_COLUMNS) + list(selected_features)].copy()
    if run is not None:
        working["Run"] = str(run).removeprefix("run-")
        group_columns = list(REDUCED_METADATA_COLUMNS)
    else:
        group_columns = [
            column for column in REDUCED_METADATA_COLUMNS if column != "Run"
        ]

    records = []
    for keys, group in working.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        for feature in selected_features:
            if feature in CHANNEL_LEVEL_FEATURES:
                record[feature] = group[feature].mean()
            else:
                record[feature] = group[feature].iloc[0]
        records.append(record)

    return pd.DataFrame(records, columns=group_columns + list(selected_features))


def build_combination_jobs(
    input_records: Iterable[tuple[Path, dict]],
    bids_root: str | Path,
    output_derivative_desc: str,
    connectivity_method: str,
) -> list[dict]:
    """Group S4 inputs into deterministic task-band S5 output jobs."""
    grouped_inputs: dict[tuple[str, str], list[tuple[Path, dict]]] = {}
    for path, labels in input_records:
        grouped_inputs.setdefault((labels["task"], labels["band"]), []).append(
            (path, labels)
        )

    output_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / output_derivative_desc
        / connectivity_method
    )
    jobs = []
    for (task, band), records in sorted(grouped_inputs.items()):
        jobs.append(
            {
                "task": task,
                "band": band,
                "input_records": sorted(records, key=lambda item: item[0]),
                "output_path": (
                    output_root
                    / f"task-{task}_desc-filt-{band}_con.feather"
                ),
            }
        )
    return jobs


def _measure_quality_summary(
    dataframe: pd.DataFrame,
    features: Iterable[str],
) -> dict:
    """Summarize missing and non-finite selected feature values."""
    summary = {}
    for feature in features:
        values = dataframe[feature].to_numpy(dtype=float, na_value=np.nan)
        summary[feature] = {
            "missing": int(dataframe[feature].isna().sum()),
            "non_finite": int((~np.isfinite(values)).sum()),
        }
    return summary


def write_run_manifest(
    bids_root: str | Path,
    output_derivative_desc: str,
    connectivity_method: str,
    configuration: dict,
    summary: dict,
    missing_task_band_pairs: list[dict],
    job_records: list[dict],
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """Write one compact JSON manifest for a completed S5 run."""
    manifest_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / output_derivative_desc
        / connectivity_method
        / "manifests"
    )
    manifest_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = (
        manifest_root / f"S5_combining_network_measures_{timestamp}.json"
    )
    payload = {
        "pipeline_step": "S5",
        "created_at": datetime.now().astimezone().isoformat(),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "command": [sys.executable, *sys.argv],
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "configuration": configuration,
        "analysis_unit": list(REDUCED_METADATA_COLUMNS),
        "channel_reduction": {
            "clustering_coef": "arithmetic mean over channel rows",
            "graph_level_features": list(GRAPH_LEVEL_FEATURES),
            "channel_column": "removed after channel reduction",
            "run_column": "parsed from validated S4 filename",
        },
        "summary": summary,
        "missing_task_band_pairs": missing_task_band_pairs,
        "jobs": job_records,
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path


def main_combining_conn_dataframe(
    is_overwrite: bool = False,
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    input_derivative_desc: str = DEFAULT_INPUT_DERIVATIVE_DESC,
    output_derivative_desc: str = DEFAULT_OUTPUT_DERIVATIVE_DESC,
    connectivity_method: str = DEFAULT_CONNECTIVITY_METHOD,
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
    bands: Optional[Iterable[str]] = None,
    effective_cycles: Optional[Iterable[int]] = None,
    n_epochs_values: Optional[Iterable[int]] = None,
    features: Optional[Iterable[str]] = None,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Discover, reduce, combine, and save the requested S4 dataframes."""
    started_at = datetime.now().astimezone()
    run_started_at = time.perf_counter()
    if skip_existing and is_overwrite:
        raise ValueError("skip_existing and is_overwrite cannot both be True")

    selected_subjects = tuple(subjects) if subjects is not None else None
    selected_sessions = tuple(sessions) if sessions is not None else None
    selected_runs = tuple(runs) if runs is not None else None
    selected_cycles = (
        tuple(int(value) for value in effective_cycles)
        if effective_cycles is not None
        else None
    )
    selected_n_epochs = (
        tuple(int(value) for value in n_epochs_values)
        if n_epochs_values is not None
        else None
    )
    selected_tasks = tuple(
        dict.fromkeys(RESTING_TASKS if tasks is None else tasks)
    )
    selected_bands = tuple(
        dict.fromkeys(SUPPORTED_BANDS if bands is None else bands)
    )
    selected_features = tuple(
        dict.fromkeys(FEATURES_OF_INTEREST if features is None else features)
    )
    if not selected_tasks:
        raise ValueError("At least one task must be selected")
    if not selected_bands:
        raise ValueError("At least one band must be selected")
    validate_supported_bands(selected_bands)
    if not selected_features:
        raise ValueError("At least one feature must be selected")
    unknown_features = sorted(set(selected_features) - set(FEATURES_OF_INTEREST))
    if unknown_features:
        raise ValueError(f"Unsupported S5 features: {unknown_features}")

    input_records = list(
        iter_network_measure_files(
            bids_root=bids_root,
            input_derivative_desc=input_derivative_desc,
            subjects=selected_subjects,
            sessions=selected_sessions,
            tasks=selected_tasks,
            runs=selected_runs,
            bands=selected_bands,
            effective_cycles=selected_cycles,
            n_epochs_values=selected_n_epochs,
        )
    )
    if not input_records:
        raise FileNotFoundError("No S4 network-measure Feather files matched")

    jobs = build_combination_jobs(
        input_records,
        bids_root=bids_root,
        output_derivative_desc=output_derivative_desc,
        connectivity_method=connectivity_method,
    )
    discovered_pairs = {(job["task"], job["band"]) for job in jobs}
    missing_task_band_pairs = [
        {"task": task, "band": band}
        for task in selected_tasks
        for band in selected_bands
        if (task, band) not in discovered_pairs
    ]
    existing_outputs = [
        Path(job["output_path"])
        for job in jobs
        if Path(job["output_path"]).exists()
    ]
    summary = {
        "input_files": len(input_records),
        "planned_outputs": len(jobs),
        "existing_outputs": len(existing_outputs),
        "completed": 0,
        "skipped": 0,
    }

    logger.info(
        "CONFIG | "
        f"tasks={selected_tasks} bands={selected_bands} "
        f"effective_cycles={selected_cycles} n_epochs={selected_n_epochs} "
        f"features={selected_features}"
    )
    logger.info(
        f"DISCOVERY | input_files={len(input_records)} "
        f"planned_outputs={len(jobs)} existing_outputs={len(existing_outputs)} "
        f"missing_task_band_pairs={len(missing_task_band_pairs)}"
    )

    if dry_run:
        for job in jobs:
            output_path = Path(job["output_path"])
            status = "existing" if output_path.exists() else "planned"
            logger.info(
                f"DRY_RUN | task={job['task']} band={job['band']} "
                f"inputs={len(job['input_records'])} -> {output_path} | "
                f"status={status}"
            )
        for pair in missing_task_band_pairs:
            logger.warning(
                f"DRY_RUN_MISSING | task={pair['task']} band={pair['band']}"
            )
        return {
            **summary,
            "missing_task_band_pairs": missing_task_band_pairs,
        }

    if missing_task_band_pairs:
        raise FileNotFoundError(
            "No S4 inputs were found for requested task-band pairs; first: "
            f"{missing_task_band_pairs[0]}"
        )
    if existing_outputs and not skip_existing and not is_overwrite:
        raise FileExistsError(
            f"{len(existing_outputs)} planned S5 outputs already exist; first: "
            f"{existing_outputs[0]}. Use --skip-existing to preserve them or "
            "--overwrite to replace them."
        )

    pending_input_files = sum(
        len(job["input_records"])
        for job in jobs
        if not (skip_existing and Path(job["output_path"]).exists())
    )
    progress_bar = tqdm(
        total=pending_input_files,
        desc="S5 input files",
        unit="file",
        dynamic_ncols=True,
        disable=pending_input_files == 0,
    )
    job_records = []
    for job in jobs:
        output_path = Path(job["output_path"])
        if skip_existing and output_path.exists():
            summary["skipped"] += 1
            job_records.append(
                {
                    "task": job["task"],
                    "band": job["band"],
                    "output_path": str(output_path),
                    "input_paths": [
                        str(path) for path, _ in job["input_records"]
                    ],
                    "status": "skipped",
                }
            )
            logger.info(f"SKIPPED | {output_path}")
            continue

        job_started_at = time.perf_counter()
        logger.info(
            f"STARTED | task={job['task']} band={job['band']} "
            f"inputs={len(job['input_records'])} -> {output_path}"
        )
        reduced_dataframes = []
        input_details = []
        for input_path, labels in job["input_records"]:
            dataframe = pd.read_feather(input_path)
            reduced = clean_df_conn(
                dataframe,
                features=selected_features,
                run=labels["run"],
                labels=labels,
                source_path=input_path,
            )
            reduced_dataframes.append(reduced)
            file_stat = input_path.stat()
            input_details.append(
                {
                    "path": str(input_path),
                    "labels": labels,
                    "size_bytes": file_stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        file_stat.st_mtime
                    ).astimezone().isoformat(),
                    "input_shape": list(dataframe.shape),
                    "reduced_shape": list(reduced.shape),
                    "input_quality": _measure_quality_summary(
                        dataframe,
                        selected_features,
                    ),
                    "reduced_quality": _measure_quality_summary(
                        reduced,
                        selected_features,
                    ),
                }
            )
            logger.debug(
                f"REDUCED | {input_path} | {dataframe.shape} -> {reduced.shape}"
            )
            progress_bar.update(1)

        combined = pd.concat(reduced_dataframes, ignore_index=True)
        duplicate_rows = combined.duplicated(
            subset=list(REDUCED_METADATA_COLUMNS),
            keep=False,
        )
        if duplicate_rows.any():
            raise ValueError(
                f"Duplicate reduced analysis units for task={job['task']} "
                f"band={job['band']}"
            )
        combined = combined.sort_values(
            list(REDUCED_METADATA_COLUMNS),
            kind="stable",
        ).reset_index(drop=True)
        atomic_write_feather(
            combined,
            output_path,
            overwrite=is_overwrite,
        )

        summary["completed"] += 1
        record = {
            "task": job["task"],
            "band": job["band"],
            "output_path": str(output_path),
            "output_shape": list(combined.shape),
            "output_size_bytes": output_path.stat().st_size,
            "output_quality": _measure_quality_summary(
                combined,
                selected_features,
            ),
            "duplicate_analysis_units": 0,
            "inputs": input_details,
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - job_started_at,
        }
        job_records.append(record)
        logger.info(
            f"COMPLETED | {output_path} | shape={combined.shape} "
            f"elapsed={record['elapsed_seconds']:.3f}s"
        )
    progress_bar.close()

    elapsed_seconds = time.perf_counter() - run_started_at
    configuration = {
        "bids_root": str(Path(bids_root).resolve()),
        "input_derivative_desc": input_derivative_desc,
        "output_derivative_desc": output_derivative_desc,
        "connectivity_method": connectivity_method,
        "subjects": (
            list(selected_subjects) if selected_subjects is not None else None
        ),
        "sessions": (
            list(selected_sessions) if selected_sessions is not None else None
        ),
        "tasks": list(selected_tasks),
        "runs": list(selected_runs) if selected_runs is not None else None,
        "bands": list(selected_bands),
        "effective_cycles": (
            list(selected_cycles) if selected_cycles is not None else None
        ),
        "n_epochs": (
            list(selected_n_epochs) if selected_n_epochs is not None else None
        ),
        "features": list(selected_features),
        "overwrite": is_overwrite,
        "skip_existing": skip_existing,
    }
    manifest_path = write_run_manifest(
        bids_root=bids_root,
        output_derivative_desc=output_derivative_desc,
        connectivity_method=connectivity_method,
        configuration=configuration,
        summary=summary,
        missing_task_band_pairs=missing_task_band_pairs,
        job_records=job_records,
        started_at=started_at,
        elapsed_seconds=elapsed_seconds,
    )
    logger.info(f"MANIFEST | {manifest_path}")
    return {**summary, "manifest_path": manifest_path}


def parse_args() -> argparse.Namespace:
    """Parse S5 discovery, reduction, and output-safety arguments."""
    parser = argparse.ArgumentParser(
        description="Combine S4 network-measure Feather dataframes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Preview one task-band output:
    python S5_combining_conn_dataframe.py --task EyesClosedNoTask --band alpha --dry-run

  Write a selected validation output:
    python S5_combining_conn_dataframe.py --task EyesClosedNoTask --band alpha --subject ST001 ST002 --output-derivative-desc group-connectivity-validation
""",
    )
    parser.add_argument("--bids-root", default=DIR_BIDS_SPEECHTRACKING)
    parser.add_argument(
        "--input-derivative-desc",
        default=DEFAULT_INPUT_DERIVATIVE_DESC,
    )
    parser.add_argument(
        "--output-derivative-desc",
        default=DEFAULT_OUTPUT_DERIVATIVE_DESC,
    )
    parser.add_argument(
        "--connectivity-method",
        default=DEFAULT_CONNECTIVITY_METHOD,
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        help="Subject label(s), with or without sub-.",
    )
    parser.add_argument(
        "--session",
        nargs="+",
        help="Session label(s), with or without ses-.",
    )
    parser.add_argument("--task", nargs="+", choices=RESTING_TASKS)
    parser.add_argument(
        "--run",
        nargs="+",
        help="Run label(s), with or without run-.",
    )
    parser.add_argument("--band", nargs="+", choices=SUPPORTED_BANDS)
    parser.add_argument("--effective-cycles", nargs="+", type=int)
    parser.add_argument("--n-epochs", nargs="+", type=int)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURES_OF_INTEREST,
        help="S4 features to retain. Defaults to the established six features.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan outputs without reading Feather contents or writing outputs.",
    )
    existing_output_group = parser.add_mutually_exclusive_group()
    existing_output_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Preserve and skip task-band outputs that already exist.",
    )
    existing_output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace task-band outputs that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize logging and run the argument-driven S5 workflow."""
    args = parse_args()
    started_at = time.perf_counter()
    log_path = set_log_file(
        SCRIPT_DIR / "log" / "S5_combining_conn_dataframe.log"
    )
    logger.info(f"LOG_FILE | {log_path.resolve()}")
    logger.info("Starting the S5 network-measure dataframe combination workflow.")
    try:
        main_combining_conn_dataframe(
            is_overwrite=args.overwrite,
            bids_root=args.bids_root,
            input_derivative_desc=args.input_derivative_desc,
            output_derivative_desc=args.output_derivative_desc,
            connectivity_method=args.connectivity_method,
            subjects=args.subject,
            sessions=args.session,
            tasks=args.task,
            runs=args.run,
            bands=args.band,
            effective_cycles=args.effective_cycles,
            n_epochs_values=args.n_epochs,
            features=args.features,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
    except Exception:
        logger.exception("S5 network-measure dataframe combination failed.")
        raise
    logger.info(
        "S5 network-measure dataframe combination completed | "
        f"{time.perf_counter() - started_at:.3f} seconds elapsed."
    )


if __name__ == "__main__":
    main()
