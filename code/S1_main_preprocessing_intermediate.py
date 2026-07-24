"""Create intermediate resting-state preprocessing objects.

The script saves each fitted :class:`preprocess_utils.Preprocessing` object as
a pickle so S2 can create frequency-filtered signals without repeating S1.
Scientific operations are intentionally written together in
``preprocess_one_bdf_file`` so the preprocessing sequence is easy to audit.

Examples
--------
Preview one recording and its planned output without loading EEG data::

    python code/study_reliability/S1_main_preprocessing_intermediate.py \
        --bids-root . --subject ST001 --session 01 \
        --task EyesClosedNoTask --run 01 --dry-run

Reproduce S1 in a separate derivative folder, preserving existing outputs::

    python code/study_reliability/S1_main_preprocessing_intermediate.py \
        --bids-root . \
        --derivative-desc preproc-intermediate-reproduction-YYYYMMDD \
        --skip-existing

Optionally override scientific settings with the existing TOML profile::

    python code/study_reliability/S1_main_preprocessing_intermediate.py \
        --bids-root . \
        --config code/study_reliability/S1_main_preprocessing_intermediate.toml \
        --derivative-desc preproc-intermediate-configured-YYYYMMDD
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import mne
from tqdm import tqdm


__author__ = "Matthew Ma <khmma@polyu.edu.hk>"
__version__ = "1.0.0"
__date__ = "2026-07-21"

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from helper_utils import get_bids_fname_tag, save_obj
from logging_utils import logger, set_log_file, use_console_log_level
from MAIN_CONSTANT import DIR_BIDS_SPEECHTRACKING
from preprocess_utils import Preprocessing
from reliability_constants import RESTING_TASKS
from reliability_io import (
    atomic_write_json,
    installed_package_version,
    normalize_bids_labels,
    sha256_file,
)


RESTING_BDF_PATTERN = re.compile(r".*task-Eyes(Open|Closed)NoTask.*\.bdf$")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Scientific settings used for every selected resting-state recording."""

    sfreq: int = 512
    reference: str = "average"
    remove_eog: bool = True
    remove_emg: bool = False
    notch_frequencies: tuple[float, ...] = (50, 100, 150, 200, 250)
    ica_highpass_hz: float = 1.0
    ica_method: str = "picard"
    ica_ortho: bool = False
    ica_extended: bool = True
    ica_random_state: int = 20240910


DEFAULT_CONFIG = PreprocessingConfig()


@dataclass(frozen=True)
class BdfFile:
    """One selected BDF input and the paths needed to process it."""

    path: Path
    subject: str
    session: str
    task: str
    run: str
    output_dir: Path
    inspection_dir: Path

    @property
    def output_path(self) -> Path:
        name = (
            f"sub-{self.subject}_ses-{self.session}_task-{self.task}"
            f"_run-{self.run}_desc-intermediate.pkl"
        )
        return self.output_dir / name


@dataclass
class PreprocessingResult:
    """Compact manifest entry for one planned BDF file."""

    subject: str
    session: str
    task: str
    run: str
    input_path: str
    output_path: str
    status: str
    elapsed_seconds: float
    raw_shape: Optional[tuple[int, int]] = None
    sfreq: Optional[float] = None
    bad_eog: Optional[list[int]] = None
    bad_emg: Optional[list[int]] = None
    ica_exclude: Optional[list[int]] = None
    error: Optional[str] = None


def preprocess_one_bdf_file(
    bdf_file: BdfFile,
    config: PreprocessingConfig = DEFAULT_CONFIG,
    *,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> PreprocessingResult:
    """Run the canonical S1 operations on one BDF file and save the result."""
    output_path = bdf_file.output_path
    if output_path.exists():
        if skip_existing:
            logger.info(f"SKIPPED | Existing output: {output_path}")
            return _result_for(bdf_file, "skipped", 0.0)
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --skip-existing "
                "to preserve it or --overwrite to replace it."
            )
        logger.warning(f"OVERWRITE | Existing output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    preprocessing = Preprocessing(
        str(bdf_file.path),
        sfreq=config.sfreq,
        ref_method=config.reference,
        rm_eog=config.remove_eog,
        rm_emg=config.remove_emg,
        dir_vs=str(bdf_file.inspection_dir),
    )

    # Canonical S1 preprocessing order.
    preprocessing.load_vs_bad_chs()
    preprocessing.resampling()
    preprocessing.rereferencing()
    preprocessing.make_montage()
    preprocessing.notch_filtering(config.notch_frequencies)
    preprocessing.fit_ICA(
        ica_highpass_hz=config.ica_highpass_hz,
        ica_method=config.ica_method,
        ica_ortho=config.ica_ortho,
        ica_extended=config.ica_extended,
        ica_random_state=config.ica_random_state,
    )
    preprocessing.remove_ica_artifacts()
    preprocessing.interpolate_bad_chs()
    save_obj(preprocessing, str(output_path))

    elapsed = time.perf_counter() - started_at
    artifact_row = preprocessing.df_artifacts.iloc[0]
    result = _result_for(bdf_file, "completed", elapsed)
    result.raw_shape = (
        len(preprocessing.raw.ch_names),
        int(preprocessing.raw.n_times),
    )
    result.sfreq = float(preprocessing.raw.info["sfreq"])
    result.bad_eog = _component_indices(artifact_row["bad_eog"])
    result.bad_emg = _component_indices(artifact_row["bad_emg"])
    result.ica_exclude = _component_indices(preprocessing.ica.exclude)
    logger.info(f"COMPLETED | {output_path} | {elapsed:.3f} seconds")
    return result


def main_preprocessing_intermediate(
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    derivative_desc: str = "preproc-intermediate",
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
    config: PreprocessingConfig = DEFAULT_CONFIG,
    config_path: Optional[str] = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> dict[str, int]:
    """Discover selected BDF files and run the S1 workflow."""
    if skip_existing and overwrite:
        raise ValueError("skip_existing and overwrite cannot both be True")
    validate_config(config)

    started_at = datetime.now().astimezone()
    run_started_at = time.perf_counter()
    bdf_files = list(
        discover_bdf_files(
            bids_root,
            derivative_desc=derivative_desc,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            runs=runs,
        )
    )
    summary = {
        "planned": len(bdf_files),
        "completed": 0,
        "skipped": 0,
        "failed": 0,
    }
    logger.info(f"CONFIG | {asdict(config)}")
    logger.info(f"DISCOVERY | Found {len(bdf_files)} resting-state BDF files")

    if dry_run:
        for bdf_file in bdf_files:
            logger.info(f"DRY_RUN | {bdf_file.path} -> {bdf_file.output_path}")
        return summary

    results: list[PreprocessingResult] = []
    old_mne_log_level = mne.set_log_level("WARNING", return_old_level=True)
    try:
        with use_console_log_level("WARNING"):
            for bdf_file in tqdm(
                bdf_files,
                total=len(bdf_files),
                desc="S1 preprocessing",
                unit="file",
            ):
                file_started_at = time.perf_counter()
                try:
                    result = preprocess_one_bdf_file(
                        bdf_file,
                        config,
                        skip_existing=skip_existing,
                        overwrite=overwrite,
                    )
                except Exception as error:
                    logger.exception(f"FAILED | {bdf_file.path}")
                    result = _result_for(
                        bdf_file,
                        "failed",
                        time.perf_counter() - file_started_at,
                        error=f"{type(error).__name__}: {error}",
                    )
                    results.append(result)
                    summary["failed"] += 1
                    if not continue_on_error:
                        manifest_path = write_run_manifest(
                            bids_root=bids_root,
                            derivative_desc=derivative_desc,
                            config=config,
                            config_path=config_path,
                            selectors=_selectors(subjects, sessions, tasks, runs),
                            results=results,
                            summary=summary,
                            started_at=started_at,
                            elapsed_seconds=time.perf_counter() - run_started_at,
                        )
                        logger.info(f"MANIFEST | {manifest_path}")
                        raise
                else:
                    results.append(result)
                    summary[result.status] += 1
    finally:
        mne.set_log_level(old_mne_log_level)

    manifest_path = write_run_manifest(
        bids_root=bids_root,
        derivative_desc=derivative_desc,
        config=config,
        config_path=config_path,
        selectors=_selectors(subjects, sessions, tasks, runs),
        results=results,
        summary=summary,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - run_started_at,
    )
    logger.info(f"SUMMARY | {summary}")
    logger.info(f"MANIFEST | {manifest_path}")
    return summary


def validate_config(config: PreprocessingConfig) -> None:
    """Reject scientifically invalid or unsupported preprocessing settings."""
    if config.sfreq <= 0:
        raise ValueError("sfreq must be greater than zero")
    if config.reference not in {"average", "mastoid", "REST"}:
        raise ValueError("reference must be average, mastoid, or REST")
    nyquist_hz = config.sfreq / 2
    if not config.notch_frequencies or any(
        frequency <= 0 or frequency >= nyquist_hz
        for frequency in config.notch_frequencies
    ):
        raise ValueError(
            f"notch frequencies must be above zero and below {nyquist_hz:g} Hz"
        )
    if not 0 < config.ica_highpass_hz < nyquist_hz:
        raise ValueError(
            f"ica_highpass_hz must be above zero and below {nyquist_hz:g} Hz"
        )
    if config.ica_method != "picard":
        raise ValueError("Only the validated picard ICA method is supported")
    if not isinstance(config.ica_random_state, int):
        raise ValueError("ica_random_state must be an integer")


def load_config(config_path: Optional[str] = None) -> PreprocessingConfig:
    """Load optional TOML overrides on top of the visible built-in defaults."""
    if config_path is None:
        validate_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    with Path(config_path).open("rb") as config_file:
        overrides = tomllib.load(config_file)
    allowed_names = {field.name for field in fields(PreprocessingConfig)}
    unknown_names = set(overrides) - allowed_names
    if unknown_names:
        raise ValueError(f"Unknown preprocessing setting(s): {sorted(unknown_names)}")
    if "notch_frequencies" in overrides:
        overrides["notch_frequencies"] = tuple(overrides["notch_frequencies"])

    config = replace(DEFAULT_CONFIG, **overrides)
    validate_config(config)
    return config


def discover_bdf_files(
    bids_root: str | Path,
    *,
    derivative_desc: str,
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
) -> Iterable[BdfFile]:
    """Yield matching resting-state BDF files in deterministic path order."""
    root = Path(bids_root)
    filters = _selectors(subjects, sessions, tasks, runs)

    for input_path in sorted(root.glob("sub-*/ses-*/*/*.bdf")):
        if not RESTING_BDF_PATTERN.match(input_path.name):
            continue
        labels = {
            "subject": get_bids_fname_tag(str(input_path), "sub"),
            "session": get_bids_fname_tag(str(input_path), "ses"),
            "task": get_bids_fname_tag(str(input_path), "task"),
            "run": get_bids_fname_tag(str(input_path), "run"),
        }
        if labels["task"] not in RESTING_TASKS:
            continue
        if any(filters[name] is not None and labels[name] not in filters[name] for name in filters):
            continue

        output_dir = (
            root
            / "derivative"
            / "study_reliability"
            / derivative_desc
            / f"sub-{labels['subject']}"
            / f"ses-{labels['session']}"
            / "preprocessing-obj"
        )
        inspection_dir = (
            root
            / "derivative"
            / "study_reliability"
            / "visual-inspection"
            / f"sub-{labels['subject']}"
            / f"ses-{labels['session']}"
            / "eeg"
        )
        yield BdfFile(
            path=input_path,
            output_dir=output_dir,
            inspection_dir=inspection_dir,
            **labels,
        )


def write_run_manifest(
    *,
    bids_root: str | Path,
    derivative_desc: str,
    config: PreprocessingConfig,
    config_path: Optional[str],
    selectors: dict[str, Optional[set[str]]],
    results: list[PreprocessingResult],
    summary: dict[str, int],
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """Write one compact manifest for this invocation of the trimmed S1."""
    finished_at = datetime.now().astimezone()
    manifest_dir = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / derivative_desc
        / "manifests"
    )
    timestamp = finished_at.strftime("%Y%m%d_%H%M%S_%f")
    manifest_path = manifest_dir / f"S1_preprocessing_manifest_{timestamp}.json"
    script_path = Path(__file__).resolve()
    payload = {
        "stage": "S1_main_preprocessing_intermediate",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "command": [sys.executable, *sys.argv],
        "source": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "script_version": __version__,
        },
        "software": {
            "python": platform.python_version(),
            "mne": installed_package_version("mne"),
            "python_picard": installed_package_version("python-picard"),
        },
        "paths": {
            "bids_root": str(Path(bids_root).resolve()),
            "derivative_desc": derivative_desc,
            "config_path": (
                None if config_path is None else str(Path(config_path).resolve())
            ),
        },
        "selectors": {
            name: None if values is None else sorted(values)
            for name, values in selectors.items()
        },
        "configuration": asdict(config),
        "summary": summary,
        "files": [asdict(result) for result in results],
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path


def _selectors(
    subjects: Optional[Iterable[str]],
    sessions: Optional[Iterable[str]],
    tasks: Optional[Iterable[str]],
    runs: Optional[Iterable[str]],
) -> dict[str, Optional[set[str]]]:
    """Normalize optional BIDS selectors once for discovery and provenance."""
    return {
        "subject": normalize_bids_labels(subjects, "sub-"),
        "session": normalize_bids_labels(sessions, "ses-"),
        "task": normalize_bids_labels(tasks, "task-"),
        "run": normalize_bids_labels(runs, "run-"),
    }


def _result_for(
    bdf_file: BdfFile,
    status: str,
    elapsed_seconds: float,
    error: Optional[str] = None,
) -> PreprocessingResult:
    """Build the common portion of one manifest file result."""
    return PreprocessingResult(
        subject=bdf_file.subject,
        session=bdf_file.session,
        task=bdf_file.task,
        run=bdf_file.run,
        input_path=str(bdf_file.path.resolve()),
        output_path=str(bdf_file.output_path.resolve()),
        status=status,
        elapsed_seconds=round(elapsed_seconds, 6),
        error=error,
    )


def _component_indices(value) -> list[int]:
    """Convert scalar or array-like ICA indices to JSON integers."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return [int(index) for index in value]


def parse_args() -> argparse.Namespace:
    """Parse file selection, configuration, and output-safety options."""
    parser = argparse.ArgumentParser(
        description="Create intermediate resting-state preprocessing objects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Preview one recording:
    python code/study_reliability/S1_main_preprocessing_intermediate.py --bids-root . --subject ST001 --session 01 --task EyesClosedNoTask --run 01 --dry-run

  Reproduce S1 in a separate derivative folder:
    python code/study_reliability/S1_main_preprocessing_intermediate.py --bids-root . --derivative-desc preproc-intermediate-reproduction-YYYYMMDD --skip-existing
""",
    )
    parser.add_argument("--bids-root", default=DIR_BIDS_SPEECHTRACKING, help="BIDS root directory.")
    parser.add_argument("--derivative-desc", default="preproc-intermediate", help="Output derivative folder name.")
    parser.add_argument("--subject", nargs="+", help="Subject label(s), with or without sub-.")
    parser.add_argument("--session", nargs="+", help="Session label(s), with or without ses-.")
    parser.add_argument("--task", nargs="+", choices=RESTING_TASKS, help="Resting task label(s).")
    parser.add_argument("--run", nargs="+", help="Run label(s), with or without run-.")
    parser.add_argument("--config", metavar="PATH", help="Optional TOML setting overrides.")
    parser.add_argument("--dry-run", action="store_true", help="List planned files without preprocessing.")
    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument("--skip-existing", action="store_true", help="Preserve existing outputs.")
    existing_group.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a file fails.")
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    log_path = set_log_file(
        SCRIPT_DIR / "log" / "S1_main_preprocessing_intermediate.log"
    )
    logger.info(f"LOG_FILE | {Path(log_path).resolve()}")
    config = load_config(args.config)
    main_preprocessing_intermediate(
        bids_root=args.bids_root,
        derivative_desc=args.derivative_desc,
        subjects=args.subject,
        sessions=args.session,
        tasks=args.task,
        runs=args.run,
        config=config,
        config_path=args.config,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
