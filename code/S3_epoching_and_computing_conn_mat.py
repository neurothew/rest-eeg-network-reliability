"""Epoch filtered resting-state EEG and compute connectivity matrices.

S3 reads the band-filtered FIF files produced by S2, crops the established
240-second analysis segment, creates fixed-length epochs based on effective
oscillatory cycles, and saves one ``NetworkFeat`` connectivity pickle per
input-band-cycle combination. Epochs are computed in memory and are not saved
as FIF files by default.

The scientific workflow is:

``filtered raw -> crop -> calculate epoch duration -> epoch ->``
``compute ciPLV connectivity -> save pickle``

Examples
--------
Run the following commands from the repository root.

Compute selected effective-cycle values for one alpha-band recording::

    conda run -n mne python code/study_reliability/S3_epoching_and_computing_conn_mat.py --bids-root . --subject ST001 --session 01 --task EyesClosedNoTask --run 01 --band alpha --effective-cycles 6 12 18 24 30 --n-jobs 1 --skip-existing

Reproduce the complete default S3 analysis in a new output folder::

    conda run -n mne python code/study_reliability/S3_epoching_and_computing_conn_mat.py --bids-root . --output-derivative-desc analysis-connectivity-reproduction
"""

__author__ = "Matthew Ma <khmma@polyu.edu.hk>"
__version__ = "1.1.0-trimmed"
__date__ = "2025-07-09"
__last_modified__ = "2026-07-21"

import argparse
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import mne
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from compute_feat_utils import NetworkFeat
from helper_utils import join_bids_fname, parse_bids_fname
from logging_utils import logger, set_log_file, use_log_level
from MAIN_CONSTANT import DIR_BIDS_SPEECHTRACKING
from preprocess_utils import epoching_rs
from reliability_constants import (
    DEFAULT_BANDS,
    RESTING_TASKS,
    SUPPORTED_BAND_PATTERN,
    SUPPORTED_BANDS,
    get_band_freq,
    validate_supported_bands,
)
from reliability_io import (
    atomic_write_json,
    format_epoch_len,
    installed_package_version,
    normalize_bids_labels,
    sha256_file,
)


# Scientific settings used by the reported analysis.
ANALYSIS_CROP_START = 2.5
ANALYSIS_CROP_END = 242.5
CONNECTIVITY_METHOD = "ciplv"
WAVELET_N_CYCLES = 3
MIN_EPOCHS_FOR_DOWNSTREAM_ANALYSIS = 8
MAX_EFFECTIVE_CYCLES_BY_BAND = {"delta": 30}

list_effcycles_pre_embc2025 = [
    6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
]
list_effcycles_post_embc2025 = [
    36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
]
list_effcycles_for_reproducibility = [
    6, 12, 18, 24, 30,
    36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
]
list_effcycles_exhaustive = (
    list_effcycles_pre_embc2025 + list_effcycles_post_embc2025
)

list_effcycles = list_effcycles_for_reproducibility
# Historical alternatives:
# list_effcycles = list_effcycles_pre_embc2025
# list_effcycles = list_effcycles_post_embc2025
# list_effcycles = list_effcycles_exhaustive

FILTERED_RAW_PATTERN = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_"
    r"task-(?P<task>Eyes(?:Open|Closed)NoTask)_run-(?P<run>[^_]+)_"
    rf"desc-filt-(?P<band>{SUPPORTED_BAND_PATTERN})_eeg\.fif$"
)


def get_effective_cycles_for_band(
    band: str,
    effective_cycles: Iterable[int],
) -> list[int]:
    """Apply the established band-specific effective-cycle limit."""
    max_cycles = MAX_EFFECTIVE_CYCLES_BY_BAND.get(band)
    if max_cycles is None:
        return list(effective_cycles)
    return [value for value in effective_cycles if value <= max_cycles]


def find_filtered_raw_files(
    bids_root: str | Path,
    input_derivative_desc: str,
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
    bands: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Find matching S2 filtered FIF files in deterministic order."""
    input_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / input_derivative_desc
    )
    if not input_root.is_dir():
        raise FileNotFoundError(f"S3 input derivative does not exist: {input_root}")

    subject_filter = normalize_bids_labels(subjects, "sub-")
    session_filter = normalize_bids_labels(sessions, "ses-")
    task_filter = set(tasks) if tasks is not None else None
    run_filter = normalize_bids_labels(runs, "run-")
    band_filter = set(bands) if bands is not None else None

    input_files = []
    for path in sorted(input_root.glob("sub-*/ses-*/eeg/*_eeg.fif")):
        match = FILTERED_RAW_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        labels = match.groupdict()
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
        input_files.append(path)

    if not input_files:
        raise FileNotFoundError(
            f"No S2 filtered FIF files matched the requested selection in {input_root}"
        )
    return input_files


def build_connectivity_jobs(
    input_files: Iterable[Path],
    effective_cycles: Iterable[int],
    bids_root: str | Path,
    output_derivative_desc: str,
) -> tuple[list[dict], int]:
    """Plan output paths for all valid input-band-cycle combinations."""
    effective_cycles = tuple(effective_cycles)
    jobs = []
    excluded_count = 0

    for input_path in input_files:
        match = FILTERED_RAW_PATTERN.fullmatch(input_path.name)
        if match is None:
            raise ValueError(f"Unexpected S2 filename: {input_path.name}")

        labels = match.groupdict()
        band = labels["band"]
        band_cycles = get_effective_cycles_for_band(band, effective_cycles)
        excluded_count += len(effective_cycles) - len(band_cycles)

        for cycles in band_cycles:
            l_freq, _ = get_band_freq(band)
            epoch_len = float(np.round(cycles / l_freq, 2))
            bids_dict = parse_bids_fname(input_path.name)
            bids_dict["desc"] = bids_dict["desc"] | {
                "effcycles": cycles,
                "epochlen": format_epoch_len(epoch_len),
            }
            output_name = f"{join_bids_fname(bids_dict)}_con.pkl"
            output_path = (
                Path(bids_root)
                / "derivative"
                / "study_reliability"
                / output_derivative_desc
                / CONNECTIVITY_METHOD
                / f"sub-{labels['subject']}"
                / f"ses-{labels['session']}"
                / output_name
            )
            jobs.append(
                {
                    "input_path": input_path,
                    "output_path": output_path,
                    "bids_dict": bids_dict,
                    "band": band,
                    "effective_cycles": cycles,
                }
            )

    return jobs, excluded_count


def _atomic_save_network_feat(
    network_feat: NetworkFeat,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Save a connectivity pickle without exposing a partial final file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        network_feat.save_to_pkl(str(temporary_path))
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"S3 output appeared during processing: {output_path}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def epoch_and_compute_connectivity(job: dict, overwrite: bool = False) -> dict:
    """Crop and epoch one filtered recording, then compute and save ciPLV.

    Epoch duration is the requested number of effective cycles divided by the
    lower frequency of the recording's band.
    """
    input_path = Path(job["input_path"])
    output_path = Path(job["output_path"])
    band = job["band"]
    l_freq, h_freq = get_band_freq(band)
    epoch_len = float(np.round(job["effective_cycles"] / l_freq, 2))
    started_at = time.perf_counter()

    # STEP 1: Load and validate the filtered recording.
    raw = mne.io.read_raw_fif(input_path, preload=True, verbose="ERROR")
    if not np.isclose(raw.info["highpass"], l_freq) or not np.isclose(
        raw.info["lowpass"], h_freq
    ):
        raise ValueError(
            f"Filter metadata mismatch for {input_path}: expected "
            f"{l_freq}-{h_freq} Hz, found "
            f"{raw.info['highpass']}-{raw.info['lowpass']} Hz"
        )
    if raw.times[-1] < ANALYSIS_CROP_END:
        raise ValueError(
            f"Recording ends at {raw.times[-1]:.3f}s and cannot be cropped "
            f"through {ANALYSIS_CROP_END:.3f}s: {input_path}"
        )

    # STEP 2: Crop the analysis segment and create fixed-length epochs.
    cropped_raw = raw.copy().crop(ANALYSIS_CROP_START, ANALYSIS_CROP_END)
    epochs = epoching_rs(cropped_raw, len_epoch=epoch_len)
    if len(epochs) < MIN_EPOCHS_FOR_DOWNSTREAM_ANALYSIS:
        raise ValueError(
            f"{band} with {job['effective_cycles']} effective cycles produced "
            f"{len(epochs)} epochs; at least "
            f"{MIN_EPOCHS_FOR_DOWNSTREAM_ANALYSIS} are required"
        )

    # TODO: Add an opt-in --save-epochs mode if reusable epoched FIF files
    # become necessary. Keep it disabled by default because of disk usage.

    # STEP 3: NetworkFeat computes connectivity during construction.
    with use_log_level("WARNING"):
        network_feat = NetworkFeat(
            epochs,
            job["bids_dict"],
            "epo",
            CONNECTIVITY_METHOD,
            l_freq=l_freq,
            h_freq=h_freq,
            n_cycles=WAVELET_N_CYCLES,
            dir_output=str(output_path.parent),
        )

        # STEP 4: Save the completed connectivity object.
        _atomic_save_network_feat(network_feat, output_path, overwrite=overwrite)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "band": band,
        "effective_cycles": job["effective_cycles"],
        "epoch_len": epoch_len,
        "n_epochs": len(epochs),
        "connectivity_shape": tuple(
            int(value) for value in network_feat.conc_data.shape
        ),
        "output_size_bytes": output_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started_at,
        "status": "completed",
    }


def write_run_manifest(
    bids_root: str | Path,
    input_derivative_desc: str,
    output_derivative_desc: str,
    bands: Iterable[str],
    effective_cycles: Iterable[int],
    excluded_by_band: dict[str, list[int]],
    summary: dict[str, int],
    job_records: list[dict],
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """Write the S3 settings, provenance, summary, and job records to JSON."""
    manifest_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / output_derivative_desc
        / CONNECTIVITY_METHOD
        / "manifests"
    )
    manifest_root.mkdir(parents=True, exist_ok=True)
    finished_at = datetime.now().astimezone()
    timestamp = finished_at.strftime("%Y%m%d_%H%M%S_%f")
    manifest_path = manifest_root / f"S3_connectivity_manifest_{timestamp}.json"
    script_path = Path(__file__).resolve()

    manifest = {
        "stage": "S3_epoching_and_computing_conn_mat",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "command": [sys.executable, *sys.argv],
        "source": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "script_version": __version__,
        },
        "software": {
            "python": platform.python_version(),
            "mne": installed_package_version("mne"),
            "mne_connectivity": installed_package_version("mne-connectivity"),
            "numpy": installed_package_version("numpy"),
            "joblib": installed_package_version("joblib"),
        },
        "paths": {
            "bids_root": str(Path(bids_root).resolve()),
            "input_derivative_desc": input_derivative_desc,
            "output_derivative_desc": output_derivative_desc,
        },
        "settings": {
            "bands_hz": {band: list(get_band_freq(band)) for band in bands},
            "effective_cycles": list(effective_cycles),
            "excluded_effective_cycles_by_band": excluded_by_band,
            "analysis_crop_seconds": [ANALYSIS_CROP_START, ANALYSIS_CROP_END],
            "minimum_epochs_for_downstream_analysis": (
                MIN_EPOCHS_FOR_DOWNSTREAM_ANALYSIS
            ),
            "connectivity_method": CONNECTIVITY_METHOD,
            "wavelet_n_cycles": WAVELET_N_CYCLES,
        },
        "summary": summary,
        "jobs": job_records,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def main_epoching_and_computing_conn_mat(
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    input_derivative_desc: str = "filtered-raw",
    output_derivative_desc: str = "analysis-connectivity",
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
    bands: Optional[Iterable[str]] = None,
    effective_cycles: Optional[Iterable[int]] = None,
    n_jobs: int = -1,
    dry_run: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """Discover inputs, plan jobs, compute connectivity, and write a manifest."""
    started_at = datetime.now().astimezone()
    run_started_at = time.perf_counter()

    # Resolve and validate the requested analysis settings.
    if skip_existing and overwrite:
        raise ValueError("skip_existing and overwrite cannot both be True")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    selected_bands = tuple(DEFAULT_BANDS if bands is None else bands)
    selected_cycles = tuple(
        dict.fromkeys(
            list_effcycles if effective_cycles is None else effective_cycles
        )
    )
    if not selected_bands:
        raise ValueError("At least one frequency band must be selected")
    if not selected_cycles or any(value <= 0 for value in selected_cycles):
        raise ValueError("Effective cycles must contain positive integers")
    validate_supported_bands(selected_bands)

    # Discover inputs and plan one output per valid input-band-cycle combination.
    input_files = find_filtered_raw_files(
        bids_root=bids_root,
        input_derivative_desc=input_derivative_desc,
        subjects=subjects,
        sessions=sessions,
        tasks=tasks,
        runs=runs,
        bands=selected_bands,
    )
    jobs, excluded_job_count = build_connectivity_jobs(
        input_files=input_files,
        effective_cycles=selected_cycles,
        bids_root=bids_root,
        output_derivative_desc=output_derivative_desc,
    )

    recordings = {
        tuple(
            FILTERED_RAW_PATTERN.fullmatch(path.name).group(label)
            for label in ("subject", "session", "task", "run")
        )
        for path in input_files
    }
    excluded_by_band = {
        band: [
            cycle
            for cycle in selected_cycles
            if cycle not in get_effective_cycles_for_band(band, selected_cycles)
        ]
        for band in selected_bands
    }
    excluded_by_band = {
        band: values for band, values in excluded_by_band.items() if values
    }
    existing_jobs = [job for job in jobs if Path(job["output_path"]).exists()]
    summary = {
        "recordings": len(recordings),
        "input_files": len(input_files),
        "planned": len(jobs),
        "completed": 0,
        "skipped": 0,
        "existing": len(existing_jobs),
        "excluded": excluded_job_count,
    }

    logger.info(
        "CONFIG | "
        f"bands={selected_bands} effective_cycles={selected_cycles} "
        f"crop={ANALYSIS_CROP_START}-{ANALYSIS_CROP_END}s "
        f"method={CONNECTIVITY_METHOD} wavelet_n_cycles={WAVELET_N_CYCLES} "
        f"n_jobs={n_jobs}"
    )
    logger.info(
        f"DISCOVERY | recordings={len(recordings)} "
        f"filtered_fif_files={len(input_files)} planned_jobs={len(jobs)} "
        f"existing_outputs={len(existing_jobs)}"
    )
    for band, excluded in excluded_by_band.items():
        logger.info(
            f"EXCLUDED | band={band} effective_cycles={excluded} | "
            f"maximum={MAX_EFFECTIVE_CYCLES_BY_BAND[band]}"
        )

    if dry_run:
        for job in jobs:
            status = "existing" if Path(job["output_path"]).exists() else "planned"
            l_freq, _ = get_band_freq(job["band"])
            epoch_len = float(np.round(job["effective_cycles"] / l_freq, 2))
            logger.info(
                f"DRY_RUN | {job['input_path']} -> {job['output_path']} | "
                f"band={job['band']} effective_cycles={job['effective_cycles']} "
                f"epoch_len={epoch_len}s status={status}"
            )
        return summary

    # Apply the requested existing-output policy before starting expensive work.
    if existing_jobs and not skip_existing and not overwrite:
        first_existing = existing_jobs[0]["output_path"]
        raise FileExistsError(
            f"{len(existing_jobs)} planned S3 outputs already exist; first: "
            f"{first_existing}. Use --skip-existing to preserve them or "
            "--overwrite to replace them."
        )

    jobs_to_run = jobs
    job_records = []
    if skip_existing:
        jobs_to_run = [job for job in jobs if not Path(job["output_path"]).exists()]
        skipped_jobs = [job for job in jobs if Path(job["output_path"]).exists()]
        summary["skipped"] = len(skipped_jobs)
        for job in skipped_jobs:
            l_freq, _ = get_band_freq(job["band"])
            job_records.append(
                {
                    "input_path": str(job["input_path"]),
                    "output_path": str(job["output_path"]),
                    "band": job["band"],
                    "effective_cycles": job["effective_cycles"],
                    "epoch_len": float(
                        np.round(job["effective_cycles"] / l_freq, 2)
                    ),
                    "status": "skipped",
                    "output_size_bytes": Path(job["output_path"]).stat().st_size,
                }
            )

    # Run the planned computations and display one shared progress bar.
    if jobs_to_run:
        tasks = (
            delayed(epoch_and_compute_connectivity)(job, overwrite=overwrite)
            for job in jobs_to_run
        )
        parallel_pool = Parallel(n_jobs=n_jobs, return_as="generator")
        results = list(
            tqdm(
                parallel_pool(tasks),
                total=len(jobs_to_run),
                desc="S3 connectivity",
                unit="job",
            )
        )
        summary["completed"] = len(results)
        job_records.extend(results)

    logger.info(f"SUMMARY | {summary}")
    manifest_path = write_run_manifest(
        bids_root=bids_root,
        input_derivative_desc=input_derivative_desc,
        output_derivative_desc=output_derivative_desc,
        bands=selected_bands,
        effective_cycles=selected_cycles,
        excluded_by_band=excluded_by_band,
        summary=summary,
        job_records=job_records,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - run_started_at,
    )
    logger.info(f"MANIFEST | {manifest_path}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse S3 file-selection and execution options."""
    parser = argparse.ArgumentParser(
        description="Epoch S2 filtered EEG and compute ciPLV connectivity pickles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Preview one recording and the delta cycle exclusions:
    conda run -n mne python code/study_reliability/S3_epoching_and_computing_conn_mat.py --bids-root . --subject ST001 --session 01 --task EyesClosedNoTask --run 01 --band delta --dry-run

  Compute missing reproducibility outputs while preserving existing pickles:
    conda run -n mne python code/study_reliability/S3_epoching_and_computing_conn_mat.py --bids-root . --skip-existing
""",
    )
    parser.add_argument("--bids-root", default=DIR_BIDS_SPEECHTRACKING)
    parser.add_argument("--input-derivative-desc", default="filtered-raw")
    parser.add_argument(
        "--output-derivative-desc",
        default="analysis-connectivity",
        help="Output folder under derivative/study_reliability.",
    )
    parser.add_argument(
        "--subject", nargs="+", help="Subject label(s), with or without sub-."
    )
    parser.add_argument(
        "--session", nargs="+", help="Session label(s), with or without ses-."
    )
    parser.add_argument("--task", nargs="+", choices=RESTING_TASKS)
    parser.add_argument(
        "--run", nargs="+", help="Run label(s), with or without run-."
    )
    parser.add_argument(
        "--band",
        nargs="+",
        choices=SUPPORTED_BANDS,
        help="Band(s) to process. Defaults to all five bands.",
    )
    parser.add_argument(
        "--effective-cycles",
        nargs="+",
        type=int,
        help="Effective-cycle values. Defaults to the reproducibility list.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Joblib worker count. Default: -1 (all available CPUs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned jobs without loading or saving EEG data.",
    )
    existing_output_group = parser.add_mutually_exclusive_group()
    existing_output_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Preserve and skip connectivity pickles that already exist.",
    )
    existing_output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace connectivity pickles that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize logging and run the argument-driven S3 workflow."""
    args = parse_args()
    started_at = time.perf_counter()
    log_path = set_log_file(
        SCRIPT_DIR / "log" / "S3_epoching_and_computing_conn_mat.log"
    )
    logger.info(f"LOG_FILE | {log_path.resolve()}")
    logger.info("Starting trimmed S3 epoching and connectivity computation ...")
    try:
        main_epoching_and_computing_conn_mat(
            bids_root=args.bids_root,
            input_derivative_desc=args.input_derivative_desc,
            output_derivative_desc=args.output_derivative_desc,
            subjects=args.subject,
            sessions=args.session,
            tasks=args.task,
            runs=args.run,
            bands=args.band,
            effective_cycles=args.effective_cycles,
            n_jobs=args.n_jobs,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
        )
    except Exception:
        logger.exception("Trimmed S3 epoching and connectivity computation failed")
        raise

    elapsed = time.perf_counter() - started_at
    logger.info(
        "Trimmed S3 epoching and connectivity computation completed | "
        f"{elapsed:.3f} seconds elapsed"
    )


if __name__ == "__main__":
    main()
