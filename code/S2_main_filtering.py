"""Filter resting-state S1 intermediate objects into EEG frequency bands.

S2 loads each selected ``Preprocessing`` pickle created by S1, filters its raw
EEG data into one or more named bands, and saves one FIF file per input-band
pair. S3 reads these FIF files for epoching and connectivity analysis.

The processing flow is intentionally short:

``discover inputs -> build jobs -> load -> filter -> refresh metadata -> save``

Each input pickle is loaded again for every band so filtering one band cannot
affect another. Existing outputs are preserved unless ``--overwrite`` is
explicitly requested. A successful non-dry run writes one JSON manifest.

Examples
--------
Process one recording and two bands in a new output derivative::

    python code/study_reliability/S2_main_filtering.py \
      --bids-root . \
      --subject ST001 \
      --session 01 \
      --task EyesClosedNoTask \
      --run 01 \
      --band theta alpha \
      --output-derivative-desc filtered-raw-ST001-example

Reproduce the complete current default run in a new output derivative::

    python code/study_reliability/S2_main_filtering.py \
      --bids-root . \
      --input-derivative-desc preproc-intermediate \
      --output-derivative-desc filtered-raw-reproduction \
      --band delta theta alpha beta gamma
"""

__author__ = "Matthew Ma <khmma@polyu.edu.hk>"
__version__ = "1.0.0"
__date__ = "2026-07-21"

import argparse
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import mne
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from helper_utils import load_obj
from logging_utils import logger, set_log_file, use_console_log_level
from MAIN_CONSTANT import DIR_BIDS_SPEECHTRACKING
from reliability_constants import (
    DEFAULT_BANDS,
    RESTING_TASKS,
    SUPPORTED_BANDS,
    get_band_freq,
    validate_supported_bands,
)
from reliability_io import (
    atomic_write_json,
    installed_package_version,
    normalize_bids_labels,
    sha256_file,
)


INTERMEDIATE_PATTERN = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_"
    r"task-(?P<task>Eyes(?:Open|Closed)NoTask)_run-(?P<run>[^_]+)_"
    r"desc-intermediate\.pkl$"
)


@dataclass(frozen=True)
class IntermediateFile:
    """One S1 intermediate pickle and its BIDS entities."""

    path: Path
    subject: str
    session: str
    task: str
    run: str


def parse_args() -> argparse.Namespace:
    """Parse S2 file-selection and execution options."""
    parser = argparse.ArgumentParser(
        description="Filter S1 resting-state intermediate pickles into EEG bands."
    )
    parser.add_argument(
        "--bids-root", default=DIR_BIDS_SPEECHTRACKING, help="BIDS root directory."
    )
    parser.add_argument(
        "--input-derivative-desc",
        default="preproc-intermediate",
        help="Input derivative folder name.",
    )
    parser.add_argument(
        "--output-derivative-desc",
        default="filtered-raw",
        help="Output derivative folder name.",
    )
    parser.add_argument(
        "--subject", nargs="+", help="Subject label(s), with or without sub-."
    )
    parser.add_argument(
        "--session", nargs="+", help="Session label(s), with or without ses-."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=RESTING_TASKS,
        help="Task label(s) to filter.",
    )
    parser.add_argument("--run", nargs="+", help="Run label(s), with or without run-.")
    parser.add_argument(
        "--band",
        nargs="+",
        choices=SUPPORTED_BANDS,
        help="Band(s) to create. Defaults to delta theta alpha beta gamma.",
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
        help="Preserve and skip FIF files that already exist.",
    )
    existing_output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace FIF files that already exist.",
    )
    return parser.parse_args()


def main_filtering(
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    input_derivative_desc: str = "preproc-intermediate",
    output_derivative_desc: str = "filtered-raw",
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
    bands: Optional[Iterable[str]] = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """Discover inputs, run each input-band job, and write a run manifest."""
    if skip_existing and overwrite:
        raise ValueError("skip_existing and overwrite cannot both be True")

    selected_bands = tuple(DEFAULT_BANDS if bands is None else bands)
    if not selected_bands:
        raise ValueError("At least one frequency band must be selected")
    validate_supported_bands(selected_bands)

    intermediate_files = list(
        discover_intermediate_files(
            bids_root=bids_root,
            input_derivative_desc=input_derivative_desc,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            runs=runs,
        )
    )
    jobs = [
        (
            intermediate_file,
            band,
            build_output_path(
                intermediate_file,
                band,
                bids_root=bids_root,
                output_derivative_desc=output_derivative_desc,
            ),
        )
        for intermediate_file in intermediate_files
        for band in selected_bands
    ]

    logger.info(
        "CONFIG | "
        f"bids_root={Path(bids_root)} | "
        f"input={input_derivative_desc} | output={output_derivative_desc} | "
        f"bands={selected_bands} | dry_run={dry_run} | "
        f"skip_existing={skip_existing} | overwrite={overwrite}"
    )
    logger.info(f"DISCOVERY | Found {len(intermediate_files)} intermediate files")
    logger.info(f"DISCOVERY | Planned {len(jobs)} filtering jobs")

    summary = {"planned": len(jobs), "completed": 0, "skipped": 0}
    if dry_run:
        for intermediate_file, band, output_path in jobs:
            l_freq, h_freq = get_band_freq(band)
            logger.info(
                f"DRY_RUN | {intermediate_file.path} -> {output_path} | "
                f"band={band} l_freq={l_freq} h_freq={h_freq}"
            )
        return summary

    existing_outputs = [output_path for _, _, output_path in jobs if output_path.exists()]
    if existing_outputs and not (skip_existing or overwrite):
        preview = "\n".join(f"  {path}" for path in existing_outputs[:5])
        remainder = len(existing_outputs) - 5
        if remainder > 0:
            preview += f"\n  ... and {remainder} more"
        raise FileExistsError(
            f"{len(existing_outputs)} planned S2 output(s) already exist:\n"
            f"{preview}\n"
            "Use --skip-existing, --overwrite, or a new output derivative."
        )

    started_at = datetime.now().astimezone()
    run_started_at = time.perf_counter()
    results: list[dict] = []
    old_mne_log_level = mne.set_log_level("WARNING", return_old_level=True)
    try:
        with use_console_log_level("WARNING"):
            for intermediate_file, band, output_path in tqdm(
                jobs,
                total=len(jobs),
                desc="S2 filtering",
                unit="job",
            ):
                result = filter_one_job(
                    intermediate_file,
                    band,
                    output_path,
                    skip_existing=skip_existing,
                    overwrite=overwrite,
                )
                results.append(result)
                summary[result["status"]] += 1
    finally:
        mne.set_log_level(old_mne_log_level)

    logger.info(f"SUMMARY | {summary}")
    if results:
        manifest_path = write_run_manifest(
            bids_root=bids_root,
            input_derivative_desc=input_derivative_desc,
            output_derivative_desc=output_derivative_desc,
            bands=selected_bands,
            results=results,
            summary=summary,
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - run_started_at,
        )
        logger.info(f"MANIFEST | {manifest_path}")
    return summary


def discover_intermediate_files(
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    input_derivative_desc: str = "preproc-intermediate",
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
) -> Iterable[IntermediateFile]:
    """Yield matching S1 intermediate pickles in stable sorted order."""
    input_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / input_derivative_desc
    )
    if not input_root.is_dir():
        raise FileNotFoundError(f"S2 input derivative does not exist: {input_root}")

    subject_filter = normalize_bids_labels(subjects, "sub-")
    session_filter = normalize_bids_labels(sessions, "ses-")
    task_filter = set(tasks) if tasks is not None else None
    run_filter = normalize_bids_labels(runs, "run-")

    for path in sorted(input_root.glob("sub-*/ses-*/preprocessing-obj/*.pkl")):
        match = INTERMEDIATE_PATTERN.fullmatch(path.name)
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
        yield IntermediateFile(path=path, **labels)


def build_output_path(
    intermediate_file: IntermediateFile,
    band: str,
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    output_derivative_desc: str = "filtered-raw",
) -> Path:
    """Build the S2 FIF path for one input-band pair."""
    get_band_freq(band)
    filename = (
        f"sub-{intermediate_file.subject}_ses-{intermediate_file.session}_"
        f"task-{intermediate_file.task}_run-{intermediate_file.run}_"
        f"desc-filt-{band}_eeg.fif"
    )
    return (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / output_derivative_desc
        / f"sub-{intermediate_file.subject}"
        / f"ses-{intermediate_file.session}"
        / "eeg"
        / filename
    )


def filter_one_job(
    intermediate_file: IntermediateFile,
    band: str,
    output_path: Path,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> dict:
    """Load, filter, refresh metadata, and save one input-band pair."""
    if output_path.exists():
        if skip_existing:
            logger.info(f"SKIPPED | Existing output: {output_path}")
            return {
                "input_path": str(intermediate_file.path),
                "output_path": str(output_path),
                "band": band,
                "status": "skipped",
                "raw_shape": None,
                "sfreq": None,
            }
        if not overwrite:
            raise FileExistsError(f"S2 output already exists: {output_path}")
        logger.warning(f"OVERWRITE | Existing output: {output_path}")

    l_freq, h_freq = get_band_freq(band)
    logger.info(
        f"STARTED | {intermediate_file.path} | "
        f"band={band} l_freq={l_freq} h_freq={h_freq}"
    )

    preprocessing = load_obj(str(intermediate_file.path))
    preprocessing.raw.filter(l_freq=l_freq, h_freq=h_freq)
    preprocessing.set_meta_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing.raw.save(str(output_path), overwrite=overwrite)

    logger.info(f"COMPLETED | {output_path}")
    return {
        "input_path": str(intermediate_file.path),
        "output_path": str(output_path),
        "band": band,
        "status": "completed",
        "raw_shape": [len(preprocessing.raw.ch_names), int(preprocessing.raw.n_times)],
        "sfreq": float(preprocessing.raw.info["sfreq"]),
    }


def write_run_manifest(
    bids_root: str | Path,
    input_derivative_desc: str,
    output_derivative_desc: str,
    bands: Iterable[str],
    results: list[dict],
    summary: dict[str, int],
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """Write one compact JSON manifest for a completed S2 run."""
    script_path = Path(__file__).resolve()
    finished_at = datetime.now().astimezone()
    timestamp = finished_at.strftime("%Y%m%d_%H%M%S_%f")
    manifest_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / output_derivative_desc
        / "manifests"
    )
    manifest_path = manifest_root / (
        f"S2_main_filtering_manifest_{timestamp}.json"
    )
    manifest = {
        "stage": "S2_main_filtering",
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
            "numpy": installed_package_version("numpy"),
        },
        "paths": {
            "bids_root": str(Path(bids_root)),
            "input_derivative_desc": input_derivative_desc,
            "output_derivative_desc": output_derivative_desc,
        },
        "bands_hz": {band: list(get_band_freq(band)) for band in bands},
        "summary": summary,
        "jobs": results,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    """Run the S2 command-line workflow."""
    args = parse_args()
    started_at = time.perf_counter()
    log_path = set_log_file(SCRIPT_DIR / "log" / "S2_main_filtering.log")
    logger.info(f"LOG_FILE | {log_path.resolve()}")
    logger.info("Starting trimmed S2 filtering procedures ...")
    main_filtering(
        bids_root=args.bids_root,
        input_derivative_desc=args.input_derivative_desc,
        output_derivative_desc=args.output_derivative_desc,
        subjects=args.subject,
        sessions=args.session,
        tasks=args.task,
        runs=args.run,
        bands=args.band,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
    )
    elapsed = time.perf_counter() - started_at
    logger.info(f"Trimmed S2 completed | {elapsed:.3f} seconds elapsed")


if __name__ == "__main__":
    main()
