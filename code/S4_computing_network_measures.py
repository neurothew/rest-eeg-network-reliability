"""
This script aims to compute network measures from the connectivity matrices computed in the previous step.

Refer to the figure to see the whole analysis pipeline.

Examples
--------
Single-participant S4 processing with selected parameters:
    conda run -n mne python code/study_reliability/S4_computing_network_measures.py --subject ST001 --session 01 --task EyesClosedNoTask --run 01 --band alpha theta --effective-cycles 12 24 --n-epochs 6 10 --thresholds 0.10 0.50 1.00 --n-jobs 1 --output-derivative-desc analysis-network-measures-single-participant-example

Full reproduction using all present defaults, writing to a new output derivative:
    conda run -n mne python code/study_reliability/S4_computing_network_measures.py --output-derivative-desc analysis-network-measures-reproduction

"""
__author__ = "Matthew Ma <khmma@polyu.edu.hk>"
__version__ = "1.2.1"
__date__ = "2025-07-09"
__last_modified__ = "2026-07-21"

try:
    import shutup
except ModuleNotFoundError:
    shutup = None
else:
    shutup.please()

import argparse
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from helper_utils import list_files, load_obj
# from tictocpy import tic, toc # This is no longer needed.
from MAIN_CONSTANT import DIR_BIDS_SPEECHTRACKING
import pandas as pd
import numpy as np
from logging_utils import (
    logger,
    set_log_file,
    set_log_level,
    use_console_log_level,
    verbose,
)
import compute_network_feat_utils

# S3 pickles created before the utility rename reference this module name.
sys.modules.setdefault("compute_feat_utils", compute_network_feat_utils)

from compute_network_feat_utils import (
    compute_network_measures,
    get_conn_mat,
    thresholding_conn_mat,
)
from joblib import Parallel, delayed
from tqdm import tqdm
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


CONNECTIVITY_PICKLE_PATTERN = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_"
    r"task-(?P<task>Eyes(?:Open|Closed)NoTask)_run-(?P<run>[^_]+)_"
    rf"desc-filt-(?P<band>{SUPPORTED_BAND_PATTERN})-"
    r"effcycles-(?P<effective_cycles>\d+)-"
    r"epochlen-(?P<epoch_len>\d+)ms_con\.pkl"
)

list_effcycles_for_reproducibility = [
    6, 12, 18, 24, 30, 36, 42, 48, 54, 60,
    66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
]
list_effcycles = list_effcycles_for_reproducibility
list_band = ["delta", "theta", "alpha", "beta", "gamma"]
list_n_epochs = [6, 10, 14, 18, 22]
list_thres_value = list(np.round(np.arange(0.1, 1.05, 0.05), 2))

DEFAULT_INPUT_DERIVATIVE_DESC = "analysis-connectivity"
DEFAULT_OUTPUT_DERIVATIVE_DESC = "analysis-network-measures"
DEFAULT_CONNECTIVITY_METHOD = "ciplv"
THRESHOLD_SCHEME = "weight"
MAX_SAMPLED_EPOCH_FRACTION = 0.75
EPOCH_SAMPLING_SCHEME = "requested-n-unique-v1"


def sample_single_network_measures(this_network_feat, this_band, thres_scheme, thres_value, rand_seed, n_epochs=None):
    """

    Compute network measurements from a selected number of random epochs with provided random seed
    """
    n_epochs_total = this_network_feat.conc_data.shape[0]
    node_names = this_network_feat.node_names
    if n_epochs == None:
        n_epochs = n_epochs_total

    rng = np.random.default_rng(rand_seed)
    random_indices = rng.choice(n_epochs_total, n_epochs, replace=False)

    # np.random.seed(rand_seed)
    # random_indices = np.random.choice(n_epochs_total, size=n_epochs, replace=False)
    conn_mat = get_conn_mat(this_network_feat.conc_data[random_indices, :, :, :], this_network_feat.conc_freqs, this_band, is_inclusive=True)
    conn_mat = thresholding_conn_mat(conn_mat, thres_scheme, thres_value)
    # pdb.set_trace()

    df_network_meas = compute_network_measures(conn_mat)
    df_network_meas.insert(0, "NCyclesWavelet", this_network_feat.n_cycles)
    df_network_meas.insert(0, "ThresValue", thres_value)
    df_network_meas.insert(0, "ThresScheme", thres_scheme)
    df_network_meas.insert(0, "Channel", node_names)
    df_network_meas.insert(0, "EffCycles", this_network_feat.bids_dict["desc"]["effcycles"])
    df_network_meas.insert(0, "EpochLen", int(this_network_feat.bids_dict["desc"]["epochlen"].split("ms")[0]))
    df_network_meas.insert(0, "TotalEpoch", n_epochs_total)
    df_network_meas.insert(0, "NEpoch", n_epochs)
    df_network_meas.insert(0, "Band", this_band)
    df_network_meas.insert(0 ,"Task", this_network_feat.bids_dict["task"])
    df_network_meas.insert(0 ,"Session", this_network_feat.bids_dict["ses"])
    df_network_meas.insert(0 ,"SubjectCode", this_network_feat.bids_dict["sub"])
    df_network_meas.insert(0 ,"rand_seed", rand_seed)
    # print(f"Finish computing {this_network_feat.sbjcode} seed {rand_seed} | {this_band} | EffCycles {bids_dict['effcycles']} | ThresValue {thres_value}")
    return(df_network_meas)

def sample_multiple_network_measures(this_network_feat, this_band, thres_scheme, list_thres_value, n_epochs=6):
    """
    Call the sample_single_network_measures function multiple times with different random seeds.

    User can specify a list of threshold values to compute from

    Parameters
    ----------
    this_newtwork_feat:
        dummy
    this_band:
        dummy
    thres_scheme:
        dummy
    list_thres_value:
        dummy
    n_epochs: int
        number of epochs to sample from
    """
    # bids_dict = parse_bids_fname(this_network_feat.fname)
    n_epochs_total = this_network_feat.conc_data.shape[0]
    from math import comb
    if comb(n_epochs_total, 6) <= 30:
        print("Less than 30")
        n_iter = n_epochs_total
    else:
        n_iter = 30
    
    if not type(list_thres_value) == list:
        list_thres_value = [list_thres_value]

    # initialize list of random seeds
    rng = np.random.default_rng(20250204)
    list_random_seed = rng.choice(1000, n_iter, replace=False)

    list_df = []
    for k, rand_seed in enumerate(list_random_seed):
        for thres_value in list_thres_value:
            df = sample_single_network_measures(this_network_feat, this_band, thres_scheme, thres_value, rand_seed=rand_seed, n_epochs=n_epochs)
            list_df.append(df)
    # print(f"Finish computing {this_network_feat.sbjcode} | {this_band} | EffCycles {bids_dict['effcycles']}")
    return(list_df)


def draw_unique_combos_with_seed_list(n_epochs_total, n_epochs, n_draws=30, seed=20250504, n_seed_pool=1000):
    """
    Draw unique combinations of epochs using a list of random seeds.

    The caller must pass the epoch count that will actually be sampled later.
    Reusing a returned seed with the same total and requested epoch counts then
    reproduces the combination whose uniqueness was checked here.
    
    Example
    -------
    >>> combos, seed_list = draw_unique_combos_with_seed_list(n_epochs_total=8, n_epochs=6)
    >>> print(len(combos), seed_list) 
    """
    max_unique = math.comb(n_epochs_total, n_epochs)
    target = min(n_draws, max_unique)

    rng = np.random.default_rng(seed)
    seed_list = rng.choice(1000, n_seed_pool, replace=False)

    combo_to_seed = {}
    for s in seed_list:
        rng2 = np.random.default_rng(int(s))
        combo = tuple(sorted(rng2.choice(n_epochs_total, n_epochs, replace=False)))
        if combo not in combo_to_seed:
            combo_to_seed[combo] = int(s)
        if len(combo_to_seed) >= target:
            break

    if len(combo_to_seed) < target:
        raise RuntimeError(f"Only got {len(combo_to_seed)} unique combos from seed pool of {n_seed_pool}.")

    # Return combos and their seeds in parallel lists
    combos = [np.array(c) for c in combo_to_seed.keys()]
    seeds = list(combo_to_seed.values())
    return combos, seeds


def sample_multiple_network_measures_v2(this_network_feat, this_band, thres_scheme, list_thres_value, n_epochs=6):
    """
    Call the sample_single_network_measures function multiple times with different random seeds.

    User can specify a list of threshold values to compute from

    Parameters
    ----------
    this_newtwork_feat:
        dummy
    this_band:
        dummy
    thres_scheme:
        dummy
    list_thres_value:
        dummy
    n_epochs: int
        number of epochs to sample from
    """
    # bids_dict = parse_bids_fname(this_network_feat.fname)
    n_total_epochs = this_network_feat.conc_data.shape[0]

    max_overlapping_ratio = 0.75
    list_max_n_epochs_to_sample = np.floor(n_total_epochs * max_overlapping_ratio)
    min_n_epochs = n_epochs
    # min_n_epochs = 6
    # Keep worker details at DEBUG so they do not interrupt the tqdm display.
    logger.debug(f"Total epochs: {n_total_epochs}, Max epochs to sample with {max_overlapping_ratio*100}% overlap: {list_max_n_epochs_to_sample}, Min epochs to sample: {min_n_epochs}.")
    # print(n_total_epochs, list_max_n_epochs_to_sample)
    for n_total_epoch, max_n_epochs_to_sample in zip([n_total_epochs], [list_max_n_epochs_to_sample]):
        logger.debug(f"Band: {this_band}, Total Epochs: {n_total_epoch}, Max Epochs to Sample: {max_n_epochs_to_sample}, Number of epochs sampled: {n_epochs}.")
        # print(n_total_epoch, max_n_epochs_to_sample)
        if max_n_epochs_to_sample < min_n_epochs:
            logger.debug(f"Band: {this_band}, Max Epochs to Sample: {max_n_epochs_to_sample} is less than Min Epochs: {min_n_epochs}.")
            return [pd.DataFrame()]
            break
        # DEPRECATED historical method (retained for provenance; do not reactivate):
        # combos, list_random_seeds = draw_unique_combos_with_seed_list(
        #     n_epochs_total=int(n_total_epoch),
        #     n_epochs=int(max_n_epochs_to_sample),
        # )
        # It checked uniqueness at the 75% eligibility ceiling, then reused the
        # seeds at the requested n_epochs. Different seeds can be unique at the
        # ceiling yet select the same requested-size epoch set. The direct method
        # below checks and later samples with the same n_epochs value.
        combos, list_random_seeds = draw_unique_combos_with_seed_list(
            n_epochs_total=int(n_total_epoch),
            n_epochs=int(n_epochs),
        )
        logger.debug(f"Number of combos: {len(combos)}, seeds: {list_random_seeds}")
        # print(len(combos), list_random_seeds)

    # Create a list of random seeds
    # from math import comb
    # if comb(n_epochs_total, 6) <= 30:
    #     print("Less than 30")
    #     n_iter = n_epochs_total
    # else:
    #     n_iter = 30
    # # initialize list of random seeds
    # rng = np.random.default_rng(20250204)
    # list_random_seed = rng.choice(1000, n_iter, replace=False)    

    if not type(list_thres_value) == list:
        list_thres_value = [list_thres_value]

    list_df = []
    for k, rand_seed in enumerate(list_random_seeds):
        for thres_value in list_thres_value:
            df = sample_single_network_measures(this_network_feat, this_band, thres_scheme, thres_value, rand_seed=rand_seed, n_epochs=n_epochs)
            list_df.append(df)
    # print(f"Finish computing {this_network_feat.sbjcode} | {this_band} | EffCycles {bids_dict['effcycles']}")
    return(list_df)

# def main_compute_network_measures(is_overwrite=False):
#     dir_root_bids = DIR_BIDS_SPEECHTRACKING

#     list_effcycles_pre_embc2025 = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
#     list_effcycles_post_embc2025 = [36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
#     list_effcycles = list_effcycles_pre_embc2025
#     # list_band = ["delta", "theta", "alpha", "beta", "gamma"]
#     list_band = ["theta"]

#     list_thres_value = list(np.round(np.arange(0.1, 1.05, 0.05), 2))
#     list_dir_sub_conn = list_files(os.path.join(dir_root_bids, "derivative", "study_reliability", "analysis-connectivity", "ciplv"), "sub-.*")
#     for dir_sub in list_dir_sub_conn:
#         list_dir_sub_ses = list_files(dir_sub, "ses-.*")
#         for dir_sub_ses in list_dir_sub_ses:
#             for this_band in list_band:
#                 for this_effcycles in list_effcycles:
#                     tic()
#                     # Defining the output directory
#                     flist = list_files(dir_sub_ses, f"sub-.*task-Eyes(Open|Closed)NoTask.*desc-filt-{this_band}-effcycles-{this_effcycles}.*_con.pkl")
#                     # pdb.set_trace()
#                     for fname in flist:
#                         this_network_feat = load_obj(fname)
#                         fname_output_df = os.path.basename(fname).split(".pkl")[0] + ".feather"
#                         if not is_overwrite:
#                             if os.path.exists(os.path.join(dir_sub_ses, fname_output_df)):
#                                 logger.info(f"File {fname_output_df} already exists, skipping...")
#                                 continue
#                         # fname_output
#                         list_df_all = sample_multiple_network_measures_v2(this_network_feat, this_band, "weight", list_thres_value, n_epochs=6)
#                         df_all = pd.concat(list_df_all).reset_index(drop=True)
#                         # pdb.set_trace()
#                         df_all.to_feather(os.path.join(dir_sub_ses, fname_output_df))
#                         logger.info(f"Computed network measures for {this_network_feat.bids_dict['sub']} | {this_network_feat.bids_dict['task']} | {this_band} | EffCycles {this_effcycles} | {len(list_df_all)} iterations | {toc()} seconds elapsed.")


# def main_compute_network_measures_parallel(is_parallel=True, is_overwrite=False):
#     dir_root_bids = DIR_BIDS_SPEECHTRACKING
#     # list_effcycles_pre_embc2025 = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
#     # list_effcycles_post_embc2025 = [36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
#     list_effcycles_for_reproducibility = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

#     list_effcycles = list_effcycles_for_reproducibility
#     # list_band = ["delta", "theta", "alpha", "beta", "gamma"]
#     list_band = ["alpha"]

#     list_thres_value = list(np.round(np.arange(0.1, 1.05, 0.05), 2))
#     list_dir_sub_conn = list_files(os.path.join(dir_root_bids, "derivative", "study_reliability", "analysis-connectivity", "ciplv"), "sub-.*")
#     for dir_sub in list_dir_sub_conn:
#         list_dir_sub_ses = list_files(dir_sub, "ses-.*")
#         for dir_sub_ses in list_dir_sub_ses:
#             for this_band in list_band:
#                 if is_parallel:
#                     tic()
#                     # Parallel(n_jobs=-1)(delayed(parallel_effcycles_worker)(this_band, this_effcycle, dir_features, n_epochs) for this_effcycle in list_effcycles)
#                     # dir_sub_ses, this_band, this_effcycles, list_thres_value, n_epochs, logger=None, is_overwrite=False
#                     n_epochs = 6
#                     tasks = (delayed(parallel_effcycles_worker)(dir_sub_ses, this_band, this_effcycles, list_thres_value, n_epochs, logger=logger, is_overwrite=True)
#                             for this_effcycles in list_effcycles)
#                     # Execute the tasks in parallel
#                     results = Parallel(n_jobs=-1, verbose=10)(tasks)
#                     logger.info(f"Computed network measures in parallel for sub {os.path.basename(dir_sub)} | {this_band} | {toc()} seconds elapsed.")
#                 else:
#                     for this_effcycles in list_effcycles:
#                         tic()
#                         # Defining the output directory
#                         flist = list_files(dir_sub_ses, f"sub-.*task-Eyes(Open|Closed)NoTask.*desc-filt-{this_band}-effcycles-{this_effcycles}.*_con.pkl")
#                         for fname in flist:
#                             this_network_feat = load_obj(fname)
#                             fname_output_df = os.path.basename(fname).split(".pkl")[0] + ".feather"
#                             if not is_overwrite:
#                                 if os.path.exists(os.path.join(dir_sub_ses, fname_output_df)):
#                                     logger.info(f"File {fname_output_df} already exists, skipping...")
#                                     continue
#                             # fname_output
#                             list_df_all = sample_multiple_network_measures(this_network_feat, this_band, "weight", list_thres_value, n_epochs=6)
#                             df_all = pd.concat(list_df_all).reset_index(drop=True)
#                             pdb.set_trace()
#                             # df_all.to_feather(os.path.join(dir_sub_ses, fname_output_df))
#                             logger.info(f"Computed network measures for {this_network_feat.bids_dict["sub"]} | {this_network_feat.bids_dict["task"]} | {this_band} | EffCycles {this_effcycles} | {len(list_df_all)} iterations | {toc()} seconds elapsed.")


def iter_connectivity_files(
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    input_derivative_desc: str = DEFAULT_INPUT_DERIVATIVE_DESC,
    connectivity_method: str = DEFAULT_CONNECTIVITY_METHOD,
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    runs: Optional[Iterable[str]] = None,
    bands: Optional[Iterable[str]] = None,
    effective_cycles: Optional[Iterable[int]] = None,
) -> Iterable[Path]:
    """Yield matching S3 connectivity pickles in deterministic order."""
    input_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / input_derivative_desc
        / connectivity_method
    )
    if not input_root.is_dir():
        raise FileNotFoundError(f"S4 input derivative does not exist: {input_root}")

    subject_filter = normalize_bids_labels(subjects, "sub-")
    session_filter = normalize_bids_labels(sessions, "ses-")
    task_filter = set(tasks) if tasks is not None else None
    run_filter = normalize_bids_labels(runs, "run-")
    band_filter = set(bands) if bands is not None else None
    cycle_filter = (
        {int(value) for value in effective_cycles}
        if effective_cycles is not None
        else None
    )

    for path in sorted(input_root.glob("sub-*/ses-*/*_con.pkl")):
        match = CONNECTIVITY_PICKLE_PATTERN.fullmatch(path.name)
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
        if (
            cycle_filter is not None
            and int(labels["effective_cycles"]) not in cycle_filter
        ):
            continue
        yield path


def build_network_measure_job(
    input_path: str | Path,
    n_epochs_values: Iterable[int],
    bids_root: str | Path = DIR_BIDS_SPEECHTRACKING,
    output_derivative_desc: str = DEFAULT_OUTPUT_DERIVATIVE_DESC,
) -> dict:
    """Resolve metadata and all sampled-epoch outputs for one S3 pickle."""
    input_path = Path(input_path)
    match = CONNECTIVITY_PICKLE_PATTERN.fullmatch(input_path.name)
    if match is None:
        raise ValueError(f"Unexpected S3 connectivity filename: {input_path.name}")

    labels = match.groupdict()
    output_dir = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / output_derivative_desc
        / f"sub-{labels['subject']}"
        / f"ses-{labels['session']}"
    )
    input_stem = input_path.name.removesuffix("_con.pkl")
    output_paths = {
        int(value): output_dir / f"{input_stem}-npo{int(value)}_con.feather"
        for value in n_epochs_values
    }
    return {
        "input_path": input_path,
        "subject": labels["subject"],
        "session": labels["session"],
        "task": labels["task"],
        "run": labels["run"],
        "band": labels["band"],
        "effective_cycles": int(labels["effective_cycles"]),
        "epoch_len_ms": int(labels["epoch_len"]),
        "output_paths": output_paths,
    }


def job_record_sort_key(record: dict) -> tuple:
    """Return a deterministic scientific-entity order for manifest records."""
    return (
        str(record.get("subject", "")),
        str(record.get("session", "")),
        str(record.get("task", "")),
        str(record.get("run", "")),
        str(record.get("band", "")),
        int(record.get("effective_cycles", -1)),
        int(record.get("n_epochs", -1)),
        str(record.get("output_path", "")),
        str(record.get("status", "")),
    )


def write_run_manifest(
    bids_root: str | Path,
    output_derivative_desc: str,
    connectivity_method: str,
    selected_bands: Iterable[str],
    selected_cycles: Iterable[int],
    selected_n_epochs: Iterable[int],
    selected_thresholds: Iterable[float],
    n_jobs: int,
    summary: dict,
    missing_parameter_pairs: list[dict],
    job_records: list[dict],
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """Write one compact JSON artifact describing an S4 execution."""
    manifest_root = (
        Path(bids_root)
        / "derivative"
        / "study_reliability"
        / output_derivative_desc
        / "manifests"
    )
    manifest_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_root / f"S4_network_measures_{timestamp}.json"
    normalized_records = []
    for record in job_records:
        normalized_records.append(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in record.items()
            }
        )

    payload = {
        "pipeline_step": "S4",
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
        "configuration": {
            "bids_root": str(Path(bids_root).resolve()),
            "output_derivative_desc": output_derivative_desc,
            "connectivity_method": connectivity_method,
            "bands": list(selected_bands),
            "effective_cycles": list(selected_cycles),
            "n_epochs": list(selected_n_epochs),
            "threshold_scheme": THRESHOLD_SCHEME,
            "thresholds": list(selected_thresholds),
            "max_sampled_epoch_fraction": MAX_SAMPLED_EPOCH_FRACTION,
            "epoch_sampling_scheme": EPOCH_SAMPLING_SCHEME,
            "n_jobs": n_jobs,
        },
        "summary": summary,
        "missing_parameter_pairs": missing_parameter_pairs,
        "jobs": normalized_records,
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path


def _run_resolved_network_measure_job(
    job: dict,
    threshold_values: Iterable[float],
    overwrite: bool,
) -> list[dict]:
    """Load one S3 pickle once and compute its requested epoch-count outputs."""
    started_at = time.perf_counter()
    input_path = Path(job["input_path"])
    network_feat = load_obj(str(input_path))
    total_epochs = int(network_feat.conc_data.shape[0])
    max_epochs = int(np.floor(total_epochs * MAX_SAMPLED_EPOCH_FRACTION))
    records = []
    identity = {
        "subject": job["subject"],
        "session": job["session"],
        "task": job["task"],
        "run": job["run"],
    }

    for n_epochs, output_path in job["output_paths"].items():
        output_path = Path(output_path)
        output_started_at = time.perf_counter()
        if n_epochs > max_epochs:
            record = {
                "input_path": input_path,
                "output_path": output_path,
                **identity,
                "band": job["band"],
                "effective_cycles": job["effective_cycles"],
                "n_epochs": n_epochs,
                "total_epochs": total_epochs,
                "max_eligible_epochs": max_epochs,
                "status": "ineligible",
                "elapsed_seconds": time.perf_counter() - output_started_at,
            }
            logger.debug(
                f"INELIGIBLE | {output_path} | requested_epochs={n_epochs} "
                f"total_epochs={total_epochs} max_eligible={max_epochs}"
            )
            records.append(record)
            continue

        logger.debug(
            f"STARTED | {input_path} -> {output_path} | band={job['band']} "
            f"effective_cycles={job['effective_cycles']} n_epochs={n_epochs}"
        )
        list_df_all = sample_multiple_network_measures_v2(
            network_feat,
            job["band"],
            THRESHOLD_SCHEME,
            list(threshold_values),
            n_epochs=n_epochs,
        )
        df_all = pd.concat(list_df_all, ignore_index=True)
        if df_all.empty:
            raise ValueError(
                f"S4 produced an empty dataframe for eligible output: {output_path}"
            )
        atomic_write_feather(df_all, output_path, overwrite=overwrite)
        record = {
            "input_path": input_path,
            "output_path": output_path,
            **identity,
            "band": job["band"],
            "effective_cycles": job["effective_cycles"],
            "n_epochs": n_epochs,
            "total_epochs": total_epochs,
            "max_eligible_epochs": max_epochs,
            "threshold_count": len(tuple(threshold_values)),
            "_random_seeds": [
                int(value) for value in df_all["rand_seed"].drop_duplicates()
            ],
            "dataframe_shape": [int(value) for value in df_all.shape],
            "output_size_bytes": output_path.stat().st_size,
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - output_started_at,
        }
        logger.debug(
            f"COMPLETED | {output_path} | shape={df_all.shape} "
            f"elapsed={record['elapsed_seconds']:.3f}s"
        )
        records.append(record)

    input_elapsed_seconds = time.perf_counter() - started_at
    for record in records:
        record["input_elapsed_seconds"] = input_elapsed_seconds
    logger.debug(
        f"INPUT_COMPLETED | {input_path} | outputs={len(records)} "
        f"elapsed={input_elapsed_seconds:.3f}s"
    )
    return records


def main_compute_network_measures_parallel_updated(
    is_parallel=True,
    is_overwrite=False,
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
    threshold_values: Optional[Iterable[float]] = None,
    n_jobs: int = -1,
    dry_run: bool = False,
    skip_existing: bool = False,
):
    """
    Discover S3 inputs and execute the complete requested S4 hyperparameter grid.

    Existing historical S4 functions remain available. This function is the
    canonical active entry point.
    """
    started_at = datetime.now().astimezone()
    run_started_at = time.perf_counter()
    if skip_existing and is_overwrite:
        raise ValueError("skip_existing and is_overwrite cannot both be True")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if not is_parallel:
        n_jobs = 1

    selected_bands = tuple(dict.fromkeys(list_band if bands is None else bands))
    selected_cycles = tuple(
        dict.fromkeys(
            list_effcycles if effective_cycles is None else effective_cycles
        )
    )
    selected_n_epochs = tuple(
        dict.fromkeys(
            list_n_epochs if n_epochs_values is None else n_epochs_values
        )
    )
    selected_thresholds = tuple(
        float(value)
        for value in dict.fromkeys(
            list_thres_value if threshold_values is None else threshold_values
        )
    )

    if not selected_bands:
        raise ValueError("At least one band must be selected")
    if not selected_cycles or any(int(value) <= 0 for value in selected_cycles):
        raise ValueError("Effective cycles must contain positive integers")
    if (
        not selected_n_epochs
        or any(int(value) <= 0 for value in selected_n_epochs)
    ):
        raise ValueError("n_epochs must contain positive integers")
    if (
        not selected_thresholds
        or any(value <= 0 or value > 1 for value in selected_thresholds)
    ):
        raise ValueError("Thresholds must be greater than 0 and at most 1")
    validate_supported_bands(selected_bands)

    input_files = list(
        iter_connectivity_files(
            bids_root=bids_root,
            input_derivative_desc=input_derivative_desc,
            connectivity_method=connectivity_method,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            runs=runs,
            bands=selected_bands,
            effective_cycles=selected_cycles,
        )
    )
    if not input_files:
        raise FileNotFoundError("No S3 connectivity pickles matched the selection")

    jobs = [
        build_network_measure_job(
            input_path,
            selected_n_epochs,
            bids_root=bids_root,
            output_derivative_desc=output_derivative_desc,
        )
        for input_path in input_files
    ]
    discovered_pairs = {
        (job["band"], job["effective_cycles"])
        for job in jobs
    }
    missing_parameter_pairs = [
        {"band": band, "effective_cycles": int(cycle)}
        for band in selected_bands
        for cycle in selected_cycles
        if (band, int(cycle)) not in discovered_pairs
    ]

    all_outputs = [
        (job, n_epochs, Path(output_path))
        for job in jobs
        for n_epochs, output_path in job["output_paths"].items()
    ]
    existing_outputs = [
        output_path
        for _, _, output_path in all_outputs
        if output_path.exists()
    ]
    summary = {
        "input_files": len(input_files),
        "input_jobs": len(jobs),
        "planned_outputs": len(all_outputs),
        "existing_outputs": len(existing_outputs),
        "completed": 0,
        "skipped": 0,
        "ineligible": 0,
    }

    logger.info(
        "CONFIG | "
        f"bids_root={Path(bids_root).resolve()} "
        f"input_derivative={input_derivative_desc}/{connectivity_method} "
        f"output_derivative={output_derivative_desc} "
        f"bands={selected_bands} effective_cycles={selected_cycles} "
        f"n_epochs={selected_n_epochs} thresholds={selected_thresholds} "
        f"n_jobs={n_jobs}"
    )
    logger.info(
        f"DISCOVERY | input_files={len(input_files)} "
        f"planned_outputs={len(all_outputs)} "
        f"existing_outputs={len(existing_outputs)} "
        f"missing_parameter_pairs={len(missing_parameter_pairs)}"
    )

    if dry_run:
        for job, n_epochs, output_path in all_outputs:
            status = "existing" if output_path.exists() else "planned"
            logger.info(
                f"DRY_RUN | {job['input_path']} -> {output_path} | "
                f"band={job['band']} "
                f"effective_cycles={job['effective_cycles']} "
                f"n_epochs={n_epochs} status={status}"
            )
        return summary

    if existing_outputs and not skip_existing and not is_overwrite:
        raise FileExistsError(
            f"{len(existing_outputs)} planned S4 outputs already exist; first: "
            f"{existing_outputs[0]}. Use --skip-existing to preserve them or "
            "--overwrite to replace them."
        )

    job_records = []
    jobs_to_run = []
    for job in jobs:
        pending_outputs = {}
        for n_epochs, output_path in job["output_paths"].items():
            output_path = Path(output_path)
            if skip_existing and output_path.exists():
                job_records.append(
                    {
                        "input_path": job["input_path"],
                        "output_path": output_path,
                        "subject": job["subject"],
                        "session": job["session"],
                        "task": job["task"],
                        "run": job["run"],
                        "band": job["band"],
                        "effective_cycles": job["effective_cycles"],
                        "n_epochs": n_epochs,
                        "status": "skipped",
                    }
                )
                logger.debug(f"SKIPPED | {output_path}")
            else:
                pending_outputs[n_epochs] = output_path
        if pending_outputs:
            pending_job = dict(job)
            pending_job["output_paths"] = pending_outputs
            jobs_to_run.append(pending_job)

    pending_output_count = sum(
        len(job["output_paths"])
        for job in jobs_to_run
    )
    participant_planned = Counter(
        job["subject"]
        for job, _, _ in all_outputs
    )
    participant_status = defaultdict(Counter)
    cumulative_status = Counter()
    logged_participants = set()

    def register_records(records: Iterable[dict]) -> list[str]:
        """Update run counters and return participants newly completed."""
        touched_subjects = set()
        for record in records:
            status = record["status"]
            subject = record["subject"]
            cumulative_status[status] += 1
            participant_status[subject][status] += 1
            touched_subjects.add(subject)

            random_seeds = record.pop("_random_seeds", None)
            logger.debug(
                "OUTPUT_RESOLVED | "
                f"sub={subject} ses={record['session']} task={record['task']} "
                f"run={record['run']} band={record['band']} "
                f"effective_cycles={record['effective_cycles']} "
                f"n_epochs={record['n_epochs']} status={status} "
                f"total_epochs={record.get('total_epochs')} "
                f"max_eligible={record.get('max_eligible_epochs')} "
                f"unique_combinations={len(random_seeds) if random_seeds else 0} "
                f"random_seeds={random_seeds} output={record['output_path']}"
            )

        newly_completed = []
        for subject in sorted(touched_subjects):
            resolved = sum(participant_status[subject].values())
            if (
                resolved == participant_planned[subject]
                and subject not in logged_participants
            ):
                logged_participants.add(subject)
                newly_completed.append(subject)
        return newly_completed

    def log_completed_participants(subjects_completed: Iterable[str]) -> None:
        """Write one durable INFO summary after each participant resolves."""
        for subject in subjects_completed:
            counts = participant_status[subject]
            logger.info(
                "PARTICIPANT_COMPLETED | "
                f"sub={subject} planned={participant_planned[subject]} "
                f"written={counts['completed']} "
                f"ineligible={counts['ineligible']} skipped={counts['skipped']} "
                f"elapsed={time.perf_counter() - run_started_at:.3f}s"
            )

    log_completed_participants(register_records(job_records))

    if jobs_to_run:
        tasks = (
            delayed(parallel_effcycles_worker)(
                job,
                list_thres_value=selected_thresholds,
                is_overwrite=is_overwrite,
                skip_existing=skip_existing,
            )
            for job in jobs_to_run
        )
        try:
            with use_console_log_level("WARNING"):
                with tqdm(
                    total=pending_output_count,
                    desc="S4 requests",
                    unit="request",
                    dynamic_ncols=True,
                    mininterval=0.5,
                ) as progress:
                    progress.set_postfix(
                        written=cumulative_status["completed"],
                        ineligible=cumulative_status["ineligible"],
                        skipped=cumulative_status["skipped"],
                    )

                    if n_jobs == 1:
                        result_stream = (
                            parallel_effcycles_worker(
                                job,
                                list_thres_value=selected_thresholds,
                                is_overwrite=is_overwrite,
                                skip_existing=skip_existing,
                            )
                            for job in jobs_to_run
                        )
                    else:
                        parallel_pool = Parallel(
                            n_jobs=n_jobs,
                            return_as="generator_unordered",
                        )
                        result_stream = parallel_pool(tasks)

                    for input_records in result_stream:
                        newly_completed = register_records(input_records)
                        input_counts = Counter(
                            record["status"] for record in input_records
                        )
                        first_record = input_records[0]
                        logger.info(
                            "INPUT_COMPLETED | "
                            f"sub={first_record['subject']} "
                            f"ses={first_record['session']} "
                            f"task={first_record['task']} "
                            f"run={first_record['run']} "
                            f"band={first_record['band']} "
                            f"effective_cycles={first_record['effective_cycles']} "
                            f"written={input_counts['completed']} "
                            f"ineligible={input_counts['ineligible']} "
                            f"cumulative_written={cumulative_status['completed']} "
                            f"cumulative_ineligible={cumulative_status['ineligible']} "
                            f"elapsed={first_record['input_elapsed_seconds']:.3f}s"
                        )
                        log_completed_participants(newly_completed)
                        job_records.extend(input_records)
                        progress.update(len(input_records))
                        progress.set_postfix(
                            written=cumulative_status["completed"],
                            ineligible=cumulative_status["ineligible"],
                            skipped=cumulative_status["skipped"],
                        )
        except Exception:
            resolved_pending = (
                cumulative_status["completed"]
                + cumulative_status["ineligible"]
            )
            logger.error(
                "RUN_FAILED | "
                f"resolved={resolved_pending} "
                f"written={cumulative_status['completed']} "
                f"ineligible={cumulative_status['ineligible']} "
                f"skipped={cumulative_status['skipped']} "
                f"remaining={pending_output_count - resolved_pending}"
            )
            raise

    for status in ("completed", "skipped", "ineligible"):
        summary[status] = sum(
            record["status"] == status for record in job_records
        )
    resolved_pending = summary["completed"] + summary["ineligible"]
    if resolved_pending != pending_output_count:
        raise RuntimeError(
            "S4 pending-output reconciliation failed: "
            f"pending={pending_output_count}, resolved={resolved_pending}"
        )
    resolved_planned = resolved_pending + summary["skipped"]
    if resolved_planned != summary["planned_outputs"]:
        raise RuntimeError(
            "S4 planned-output reconciliation failed: "
            f"planned={summary['planned_outputs']}, resolved={resolved_planned}"
        )
    job_records.sort(key=job_record_sort_key)
    logger.info(f"SUMMARY | {summary}")

    manifest_path = write_run_manifest(
        bids_root=bids_root,
        output_derivative_desc=output_derivative_desc,
        connectivity_method=connectivity_method,
        selected_bands=selected_bands,
        selected_cycles=selected_cycles,
        selected_n_epochs=selected_n_epochs,
        selected_thresholds=selected_thresholds,
        n_jobs=n_jobs,
        summary=summary,
        missing_parameter_pairs=missing_parameter_pairs,
        job_records=job_records,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - run_started_at,
    )
    logger.info(f"MANIFEST | {manifest_path}")
    return summary


def parallel_effcycles_worker(
    dir_sub_ses,
    this_band=None,
    this_effcycles=None,
    list_thres_value=None,
    n_epochs=None,
    logger=None,
    is_overwrite=False,
    skip_existing=False,
):
    """
    Compute S4 outputs for one effective-cycle input.

    The active path accepts a resolved job dictionary and processes every
    requested sampled-epoch count after loading the S3 pickle once. The
    historical directory-based call signature remains supported.
    """
    if isinstance(dir_sub_ses, dict):
        return _run_resolved_network_measure_job(
            dir_sub_ses,
            threshold_values=list_thres_value,
            overwrite=is_overwrite,
        )

    # Historical directory-based path retained for compatibility.
    flist = list_files(
        dir_sub_ses,
        f"sub-.*task-Eyes(Open|Closed)NoTask.*desc-filt-"
        f"{this_band}-effcycles-{this_effcycles}.*_con.pkl",
    )
    records = []
    for fname in flist:
        this_network_feat = load_obj(fname)
        fname_output_df = (
            os.path.basename(fname).split("_con.pkl")[0]
            + f"-npo{n_epochs}_con.feather"
        )
        dir_output = os.path.join(
            DIR_BIDS_SPEECHTRACKING,
            "derivative",
            "study_reliability",
            DEFAULT_OUTPUT_DERIVATIVE_DESC,
            *dir_sub_ses.split(os.sep)[-2:],
        )
        output_path = Path(dir_output) / fname_output_df
        if output_path.exists() and not is_overwrite:
            if logger is not None:
                logger.info(f"File {fname_output_df} already exists, skipping...")
            continue
        list_df_all = sample_multiple_network_measures_v2(
            this_network_feat,
            this_band,
            THRESHOLD_SCHEME,
            list_thres_value,
            n_epochs=n_epochs,
        )
        df_all = pd.concat(list_df_all, ignore_index=True)
        atomic_write_feather(df_all, output_path, overwrite=is_overwrite)
        records.append(
            {
                "input_path": fname,
                "output_path": output_path,
                "band": this_band,
                "effective_cycles": this_effcycles,
                "n_epochs": n_epochs,
                "status": "completed",
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    """Parse S4 file-selection, hyperparameter, and execution options."""
    parser = argparse.ArgumentParser(
        description="Compute network measures from S3 connectivity pickles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Preview the full configured grid for one recording:
    python S4_computing_network_measures.py --bids-root ../.. --subject ST001 --session 01 --task EyesClosedNoTask --run 01 --dry-run

  Compute missing alpha outputs for selected settings:
    python S4_computing_network_measures.py --band alpha --effective-cycles 6 12 18 --n-epochs 6 10 --skip-existing
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
    parser.add_argument(
        "--band",
        nargs="+",
        choices=SUPPORTED_BANDS,
        help=f"Band(s) to process. Default: {list_band}.",
    )
    parser.add_argument(
        "--effective-cycles",
        nargs="+",
        type=int,
        help="Effective-cycle values. Defaults to list_effcycles.",
    )
    parser.add_argument(
        "--n-epochs",
        nargs="+",
        type=int,
        help=f"Sampled epoch counts. Default: {list_n_epochs}.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Proportional thresholds. Defaults to list_thres_value.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Joblib worker count. Default: -1 (all available CPUs).",
    )
    parser.add_argument(
        "--log-level",
        choices=("INFO", "DEBUG"),
        default="INFO",
        help="Main-process log detail. Default: INFO.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned outputs without loading connectivity pickles.",
    )
    existing_output_group = parser.add_mutually_exclusive_group()
    existing_output_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Preserve and skip S4 Feather files that already exist.",
    )
    existing_output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace S4 Feather files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize logging and run the argument-driven S4 workflow."""
    args = parse_args()
    started_at = time.perf_counter()
    log_path = set_log_file(
        SCRIPT_DIR / "log" / "S4_computing_network_measures.log"
    )
    set_log_level(args.log_level)
    logger.info(f"LOG_FILE | {log_path.resolve()}")
    logger.info("Starting the S4 network-measures workflow.")
    try:
        main_compute_network_measures_parallel_updated(
            is_parallel=args.n_jobs != 1,
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
            threshold_values=args.thresholds,
            n_jobs=args.n_jobs,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
    except Exception:
        logger.exception("S4 network-measures workflow failed.")
        raise
    logger.info(
        f"S4 network-measures workflow completed | "
        f"{time.perf_counter() - started_at:.3f} seconds elapsed."
    )


if __name__ == "__main__":
    main()
