"""Connectivity and network-measure utilities used by reliability S3 and S4.
"""

import pickle

import bct
import mne
import mne_connectivity
import numpy as np
import pandas as pd
import small_world_propensity as swp

from logging_utils import logger


__all__ = [
    "NetworkFeat",
    "get_conn_mat",
    "thresholding_conn_mat",
    "compute_network_measures",
]


def _get_band_indices(freqs, band, is_inclusive=True):
    """Return the frequency indices belonging to an EEG band."""
    band_limits = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
        "full": (1, 45),
    }
    if band not in band_limits:
        raise ValueError(f"Unsupported frequency band: {band}")

    lower, upper = band_limits[band]
    freqs = np.asarray(freqs)
    if is_inclusive:
        return (freqs >= lower) & (freqs <= upper)
    return (freqs >= lower) & (freqs < upper)


def _compute_conn_mat(data, conn_method, sfreq, l_freq=1, h_freq=45, n_cycles=7, n_jobs=1):
    """Compute time-resolved spectral connectivity for epoched EEG data."""
    logger.info("Computing connectivity with %s method.", conn_method)
    results = mne_connectivity.spectral_connectivity_time(
        data,
        freqs=np.arange(l_freq, h_freq + 1),
        sfreq=sfreq,
        faverage=False,
        method=conn_method,
        n_cycles=n_cycles,
        n_jobs=n_jobs,
    )
    return results.get_data(output="dense"), np.asarray(results.freqs)


def get_conn_mat(conc_data, conc_freqs, band="full", is_inclusive=False):
    """Average connectivity over epochs and the requested frequency band."""
    band_indices = _get_band_indices(conc_freqs, band, is_inclusive=is_inclusive)
    if not np.any(band_indices):
        raise ValueError(f"No frequencies found for band {band!r}")

    if conc_data.shape[0] != 1:
        conn_mat = conc_data[:, :, :, band_indices].mean(axis=(0, 3))
    else:
        conn_mat = np.squeeze(conc_data[:, :, :, band_indices].mean(axis=3))
    return conn_mat + conn_mat.T


def thresholding_conn_mat(conn_mat, thres_scheme, thres_value):
    """Apply S4's proportional-weight connectivity threshold."""
    if thres_scheme != "weight":
        raise ValueError(
            "The reliability pipeline supports only thres_scheme='weight'; "
            f"received {thres_scheme!r}."
        )
    return bct.threshold_proportional(conn_mat, thres_value, copy=True)


def _safe_call(func, *args, **kwargs):
    """Return ``np.nan`` when a network-measure calculation fails."""
    try:
        return func(*args, **kwargs)
    except Exception as error:
        logger.warning("%s failed: %s", func.__name__, error)
        return np.nan


def compute_network_measures(conn_mat):
    """Compute the node-level network measures used by S4."""
    distance_result = _safe_call(bct.distance_wei, conn_mat)
    dist_mat = distance_result[0] if isinstance(distance_result, tuple) else np.nan

    modularity_result = _safe_call(bct.modularity_und, conn_mat)
    modularity = modularity_result[1] if isinstance(modularity_result, tuple) else np.nan
    clustering_coef = _safe_call(bct.clustering_coef_wu, conn_mat)
    eig_centrality = _safe_call(bct.eigenvector_centrality_und, conn_mat)

    betweenness_centrality = _safe_call(bct.betweenness_wei, dist_mat)
    if not isinstance(betweenness_centrality, float) or not np.isnan(betweenness_centrality):
        n_nodes = conn_mat.shape[0]
        betweenness_centrality /= (n_nodes - 1) * (n_nodes - 2)

    char_path_result = _safe_call(bct.charpath, dist_mat)
    if isinstance(char_path_result, tuple):
        char_path, global_eff, _, _, _ = char_path_result
    else:
        char_path = np.nan
        global_eff = np.nan

    assortativity = _safe_call(bct.assortativity_wei, conn_mat)
    df_swp = _safe_call(swp.small_world_propensity, conn_mat)
    if isinstance(df_swp, pd.DataFrame):
        swp_value = df_swp["SWP"].values[0]
    else:
        swp_value = np.nan
        df_swp = pd.DataFrame(
            {"SWP": [np.nan], "Clustering_Lp": [np.nan], "PathLength_Lp": [np.nan]}
        )

    measures = pd.DataFrame(
        {
            "modularity": modularity,
            "clustering_coef": clustering_coef,
            "eig_centrality": eig_centrality,
            "betweenness_centrality": betweenness_centrality,
            "char_path": char_path,
            "global_eff": global_eff,
            "assortativity": assortativity,
            "SWP": swp_value,
        }
    )
    return measures.merge(df_swp, on="SWP")


class NetworkFeat:
    """Compute and serialize S3 connectivity outputs from an MNE instance."""

    def __init__(
        self,
        inst,
        bids_dict,
        datatype,
        conn_method,
        l_freq=1,
        h_freq=45,
        n_cycles=7,
        n_jobs=1,
        dir_output=None,
    ):
        if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError("inst must be an MNE Raw or Epochs instance.")

        self.raw = inst
        self.data = inst.get_data(picks=["eeg"])
        self.bids_dict = bids_dict
        self.datatype = datatype
        self.conn_method = conn_method
        self._dir_output = dir_output
        self.n_jobs = n_jobs
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.n_cycles = n_cycles
        self.node_names = None
        self.fit()

    def fit(self):
        """Calculate connectivity and retain only the serializable results."""
        logger.info("Fitting NetworkFeat from in-memory %s.", type(self.raw).__name__)
        data = self.data[np.newaxis, :, :] if self.datatype == "raw" else self.data
        self.conc_data, self.conc_freqs = _compute_conn_mat(
            data,
            self.conn_method,
            sfreq=self.raw.info["sfreq"],
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            n_cycles=self.n_cycles,
            n_jobs=self.n_jobs,
        )
        self.node_names = self.raw.copy().pick("eeg").info["ch_names"]
        self.data = None
        self.raw = None

    def save_to_pkl(self, fname_out):
        """Serialize the S3 connectivity result."""
        logger.info("Saving to %s", fname_out)
        with open(fname_out, "wb") as file:
            pickle.dump(self, file)
