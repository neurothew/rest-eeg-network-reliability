import numpy as np
import mne
from datetime import datetime
import logging
import os
from helper_utils import list_files, get_bids_fname_tag, save_obj, load_obj
import pandas as pd
from logging_utils import logger, verbose, set_log_file
import pickle
import pdb
from typing import Sequence

def format_epoch_len(len_epoch):
    len_epoch_ms = int(np.round(len_epoch * 1000))
    return f"{len_epoch_ms}ms"

def find_list_timestamp(this_events, sfreq):
    list_timestamp = []
    for this_id in [100]:
        list_start_idx = np.where(this_events[:, 2] == this_id)[0]
        for this_start_idx in list_start_idx:
            this_event_timestamp = this_events[this_start_idx:this_start_idx+4, 0]
            this_event_code = this_events[this_start_idx:this_start_idx+4, 2]
            if any([True if this_code in [97, 98, 99] else False for this_code in this_event_code]):
                continue
            else:
                # this_start_t = this_event_timestamp[0]/sfreq # when stim = 100
                # this_end_t = this_event_timestamp[-1]/sfreq # when stim = 101
                this_start_t = this_event_timestamp[1]/sfreq # when stim = stimulicode
                this_end_t = this_event_timestamp[2]/sfreq # when stim = 50
                list_timestamp.append([this_start_t, this_end_t])
    return(list_timestamp)

def epoching_rs(thisraw, len_epoch):
    """Epoching for resting-state data
    """
    thisevents = mne.make_fixed_length_events(thisraw, duration=len_epoch)
    thisepochs = mne.Epochs(thisraw, thisevents, tmin=0, tmax=len_epoch, baseline=None, preload=True)
    return(thisepochs)

def tidy_and_reref(this_epoch, this_ref="average"):
    """
    A wrapper of the .set_eeg_reference, handling specifically for the Biosemi data collected in our lab.

    Our data sometimes consist of 32/64 channels, and were with 6 external electrodes.
    
    Example
    -------
    >>> # Tidy and average to averaged mastoids
    >>> this_epoch = tidy_and_reref(this_epoch, "mastoid")
    """
    # ----------------------------------------------------
    # Define mastoid and EOG channels 
    # EXG3, EXG4: Left and right
    # EXG5, EXG6: Top and bottom
    raweog_list = ['EXG3', 'EXG4', 'EXG5', 'EXG6']
    mastoid_list = ['EXG1', 'EXG2']
    unused_list = ['EXG7', 'EXG8']
    # Drop the mastoid in case if no mastoid referencing is needed
    # this_epoch.add_reference_channels(mastoid_list)

    # ----------------------------------------------------
    # Re-reference to averaged mastoids
    match this_ref:
        case "mastoid":
            this_epoch.set_eeg_reference(ref_channels=mastoid_list)
        case "average":
            this_epoch.set_eeg_reference("average")
        case "REST":
            this_epoch.set_eeg_reference("REST")

    this_epoch.drop_channels(mastoid_list)
    this_epoch.drop_channels(unused_list)
    return(this_epoch)

    # # ----------------------------------------------------
    # # Create HEOG and VEOG from original EOG channels and add it to the data
    # # Also define the montage according to channel number
    # heogdata_raw = this_epoch.get_data(mne.pick_channels(this_epoch.info['ch_names'], ['EXG3', 'EXG4']))
    # veogdata_raw = this_epoch.get_data(mne.pick_channels(this_epoch.info['ch_names'], ['EXG5', 'EXG6']))

    # heogdata = heogdata_raw[0, :] - heogdata_raw[1, :]
    # heogdata = heogdata[np.newaxis, :]
    # veogdata = veogdata_raw[0, :] - veogdata_raw[1, :]
    # veogdata = veogdata[np.newaxis, :]
    # eogdata = np.concatenate((heogdata, veogdata), axis=0)
    # eog_info = mne.create_info(['HEOG', 'VEOG'], this_epoch.info['sfreq'], 'eog')
    # # eog_info['custom_ref_applied'] = True

    # # Deprecated in MNE 1.3, shouldn't be changed
    # # eog_info['lowpass'] = this_epoch.info['lowpass']
    # eograw = mne.io.RawArray(eogdata, eog_info)
    # this_epoch.add_channels([eograw], force_update_info=True)
    # this_epoch.drop_channels(raweog_list)

    # if (this_epoch.info['nchan'] >= 32) and (this_epoch.info['nchan'] < 64):
    #     eeg_nchan = 32
    #     this_epoch.set_montage(mne.channels.make_standard_montage('biosemi32'))
    # elif (this_epoch.info['nchan'] >= 64):
    #     eeg_nchan = 64
    #     this_epoch.set_montage(mne.channels.make_standard_montage('biosemi64'))
    # return(this_epoch)

# def get_logger(proc_name, dir_log=None):
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     if not dir_log == None:
#         fname_log = f'{dir_log}{os.sep}{proc_name}_time-{current_time}.log'
#     else:
#         fname_log = f'{proc_name}_time-{current_time}.log'
#     logging.basicConfig(level=logging.INFO,
#                         format = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
#                         handlers=[logging.FileHandler(fname_log), logging.StreamHandler()])
#     logger = logging.getLogger(proc_name)
#     return(logger)

class Preprocessing():
    def __init__(self, fname: str, sfreq: int, ref_method: str = "average", rm_eog: bool = True, rm_emg: bool = False, l_freq: float = 1, h_freq: float = 45, dir_vs: str=None):
        self.fname = fname
        self.sfreq = sfreq
        self.ref_method = ref_method
        self.rm_eog = rm_eog
        self.rm_emg = rm_emg
        self.dir_vs = dir_vs # directory of visual inspection
        self.l_freq = l_freq
        self.h_freq = h_freq
        self._ica = None
        self.set_meta_data()
        self.load_bdf()

    @property
    def ica(self):
        """Return the computed ICA.
        """
        if self._ica is None:
            raise ValueError("Run fit_ICA() first.")
        return self._ica

    def load_bdf(self) -> None:
        logger.info(f"LOAD_DATA | Loading BDF file: {os.path.basename(self.fname)}")
        self.raw = mne.io.read_raw_bdf(self.fname, preload=True)
        logger.info(f"LOAD_DATA | ✓ Loaded {self.raw.info['nchan']} channels, {self.raw.times[-1]:.1f}s duration")

    # def set_meta_data(self) -> None:
    #     """Initialize meta data, most importantly the output filename

    #     Also checking whether it already exist.
    #     """
    #     self.sbjcode = os.path.basename(self.fname).split("-")[1]
    #     self.task = os.path.basename(self.fname).split("-")[0]
    #     self.ses = os.path.basename(self.fname).split("-")[2].split('S')[1]
    #     # fname_output_temp = f"sub-{self.sbjcode}_ses-{self.ses}_task-{self.task}_eeg.fif"
    #     self.fname_output = f"sub-{self.sbjcode}_ses-{self.ses}_task-{self.task}_eeg.fif"
    #     self.dir_output = os.path.join(self.dir_root_bids, f"sub-{self.sbjcode}")
    #     os.makedirs(self.dir_output, exist_ok=True)

    #     fname_output_full = os.path.join(self.dir_output, self.fname_output)
    #     if os.path.isfile(fname_output_full):
    #         logger.info(f"Already exists! subject {self.sbjcode}, session {self.ses}, task {self.task}")

    #     # if self.dir_root_bids == None:
    #     #     logger.info("The BIDS root directory is not set, setting the present directory as the root directory.")
    #     #     self.dir_root_bids = os.getcwd()
    #     # else:
    #     #     self.dir_root_bids = os.path.join(self.dir_root_bids, f"sub-{self.sbjcode}")
    #     #     os.makedirs(self.dir_root_bids, exist_ok=True)
    #     #     self.fname_output = os.path.join(self.dir_root_bids, fname_output_temp)
    #     #     if os.path.isfile(self.fname_output):
    #     #         logger.info(f"Already exists! subject {self.sbjcode}, session {self.ses}, task {self.task}")

    def set_meta_data(self) -> None:
        """Initialize meta data, most importantly the output filename

        Also checking whether it already exist.

        """
        logger.info("METADATA | Extracting subject information from filename")
        self.sbjcode = get_bids_fname_tag(self.fname, "sub")
        self.task = get_bids_fname_tag(self.fname, "task")
        self.ses = get_bids_fname_tag(self.fname, "ses")
        self.run = get_bids_fname_tag(self.fname, "run")
        logger.info(f"METADATA | ✓ Subject: {self.sbjcode}, Task: {self.task}, Session: {self.ses}, Run: {self.run}")

        # fname_output_temp = f"sub-{self.sbjcode}_ses-{self.ses}_task-{self.task}_eeg.fif"
        # self.fname_output = f"sub-{self.sbjcode}_ses-{self.ses}_task-{self.task}_eeg.fif"
        # self.dir_output = os.path.join(self.dir_root_bids, f"sub-{self.sbjcode}")
        # os.makedirs(self.dir_output, exist_ok=True)

        # fname_output_full = os.path.join(self.dir_output, self.fname_output)
        # if os.path.isfile(fname_output_full):
            # logger.info(f"Already exists! subject {self.sbjcode}, session {self.ses}, task {self.task}")

        # if self.dir_root_bids == None:
        #     logger.info("The BIDS root directory is not set, setting the present directory as the root directory.")
        #     self.dir_root_bids = os.getcwd()
        # else:
        #     self.dir_root_bids = os.path.join(self.dir_root_bids, f"sub-{self.sbjcode}")
        #     os.makedirs(self.dir_root_bids, exist_ok=True)
        #     self.fname_output = os.path.join(self.dir_root_bids, fname_output_temp)
        #     if os.path.isfile(self.fname_output):
        #         logger.info(f"Already exists! subject {self.sbjcode}, session {self.ses}, task {self.task}")



    def load_vs_bad_chs(self):
        """Load visually inspected bad electrodes

        """
        try:
            if not self.dir_vs == None:
                logger.info("BAD_CHANNELS | Loading visually inspected bad channels")
                dir_inspection = self.dir_vs
                # dir_inspection = os.path.join(self.dir_root_bids, "derivative", "visual-inspection", f"sub-{self.sbjcode}", f"ses-{self.ses}", "eeg")    
                fname_tsv = list_files(dir_inspection, ".*.tsv")[0]
                df_channel_tsv = pd.read_csv(fname_tsv, sep='\t')
                list_bad_elec = list(df_channel_tsv.loc[df_channel_tsv['status'] == 'bad', 'name'].values)

                # dir_inspection = f"{os.getcwd()}{os.sep}log_inspection_exp-CleanTracking"
                # this_inspection_log = pd.read_csv(list_files(dir_inspection, f"CleanTracking-{self.sbjcode}.*.csv")[0])
                # logger.info("Loading inspection log")
                # thisrow = this_inspection_log.iloc[0]
                # list_bad_elec = list(thisrow[thisrow==1].index)
                self.raw.info['bads'] = list_bad_elec
                logger.info(f"BAD_CHANNELS | ✓ Marked {len(list_bad_elec)} bad channels: {list_bad_elec}")
                # return(list_bad_elec)
            else:
                logger.info("BAD_CHANNELS | ⚠ No visual inspection directory provided - skipping")
                # self.raw.info['bads'] = list_bad_elec
        except Exception as e:
            raise RuntimeError(f"Some errors happened while loading bad electrodes from visual inspection: {str(e)}")

    @staticmethod
    def create_eog_chs(input_raw) -> mne.io.Raw:
        """Create HEOG and VEOG channels from four external electrodes

        By default, the recorded data from our lab consist of 4 EOG channels, we need to combine them in order to form
        HEOG and VEOG channels.
        """
        thisraw = input_raw.copy()
        
        # ----------------------------------------------------
        # Define mastoid and EOG channels 
        # EXG3, EXG4: Left and right
        # EXG5, EXG6: Top and bottom
        raweog_list = ['EXG3', 'EXG4', 'EXG5', 'EXG6']

        # ----------------------------------------------------
        # Create HEOG and VEOG from original EOG channels and add it to the data
        # Also define the montage according to channel number
        heogdata_raw = thisraw.get_data(mne.pick_channels(thisraw.info['ch_names'], ['EXG3', 'EXG4']))
        veogdata_raw = thisraw.get_data(mne.pick_channels(thisraw.info['ch_names'], ['EXG5', 'EXG6']))

        heogdata = heogdata_raw[0, :] - heogdata_raw[1, :]
        heogdata = heogdata[np.newaxis, :]
        veogdata = veogdata_raw[0, :] - veogdata_raw[1, :]
        veogdata = veogdata[np.newaxis, :]
        eogdata = np.concatenate((heogdata, veogdata), axis=0)
        eog_info = mne.create_info(['HEOG', 'VEOG'], thisraw.info['sfreq'], 'eog')
        # eog_info['custom_ref_applied'] = True

        # Deprecated in MNE 1.3, shouldn't be changed
        # eog_info['lowpass'] = thisraw.info['lowpass']
        eograw = mne.io.RawArray(eogdata, eog_info)
        thisraw.add_channels([eograw], force_update_info=True)
        thisraw.drop_channels(raweog_list)

        if (thisraw.info['nchan'] >= 32) and (thisraw.info['nchan'] < 64):
            thisraw.set_montage(mne.channels.make_standard_montage('biosemi32'))
        elif (thisraw.info['nchan'] >= 64):
            thisraw.set_montage(mne.channels.make_standard_montage('biosemi64'))
        return(thisraw)

    def make_montage(self) -> None:
        logger.info("MONTAGE | Creating EOG channels and setting electrode positions")
        self.raw = self.create_eog_chs(self.raw)
        logger.info("MONTAGE | ✓ EOG channels created and montage applied")

    def fit_ICA(
        self,
        ica_highpass_hz: float = 1,
        ica_method: str = "picard",
        ica_ortho: bool = False,
        ica_extended: bool = True,
        ica_random_state: int = 20240910,
    ) -> mne.preprocessing.ICA:
        """
        Fit ICA and identify candidate EOG and muscle components.

        ICA is fitted to a high-pass-filtered copy of ``self.raw``. The original
        ``self.raw`` is not filtered or corrected by this method. The fitted ICA
        object is stored in ``self._ica``, while the detected component indices
        are stored in the one-row ``self.df_artifacts`` table.

        Parameters
        ----------
        ica_highpass_hz : float
            High-pass frequency, in Hz, applied only to the ICA fitting copy.
        ica_method : str
            ICA implementation passed to :class:`mne.preprocessing.ICA`.
            The current pipeline uses and validates ``"picard"``.
        ica_ortho : bool
            Picard ``ortho`` fitting option.
        ica_extended : bool
            Picard ``extended`` fitting option.
        ica_random_state : int
            Random seed used when initializing ICA.

        Returns
        -------
        mne.preprocessing.ICA
            The fitted ICA object, also available afterward through
            ``self.ica``.

        Expected outcome
        ----------------
        ``self.ica`` contains the fitted decomposition and
        ``self.df_artifacts`` contains ``bad_eog`` and ``bad_emg`` component
        lists. Artifact components are identified here but are not removed
        until :meth:`remove_ica_artifacts` is called.
        """
        logger.info(
            "ICA_FIT | Fitting ICA for artifact detection "
            f"({ica_highpass_hz}Hz high-pass filter)"
        )
        # Create a copy for running ICA
        thisraw_filt_forica = self.raw.copy().filter(l_freq=ica_highpass_hz, h_freq=None)
        thisica = mne.preprocessing.ICA(
            method=ica_method,
            fit_params=dict(ortho=ica_ortho, extended=ica_extended),
            random_state=ica_random_state,
        ).fit(thisraw_filt_forica)
        self._ica = thisica

        bad_eog_idx, _ = self.ica.find_bads_eog(thisraw_filt_forica)
        bad_muscle_idx, _ = self.ica.find_bads_muscle(thisraw_filt_forica)
        # logger.info(f"Bad ICA components: {bad_eog_idx}")
        logger.info(f"ICA_FIT | ✓ ICA completed - EOG components: {bad_eog_idx}, EMG components: {bad_muscle_idx}")
        thisdict = {"SubjectCode": self.sbjcode, "Task": self.task, "bad_eog": bad_eog_idx, "bad_emg": bad_muscle_idx}
        self.df_artifacts = pd.DataFrame([thisdict])
        return(thisica)
        # df_artifacts.to_csv(f"{self.dir_root_bids}{os.sep}df_bad_ics_task-{self.task}.csv")
        # df_artifacts.to_feather(f"{self.dir_root_bids}{os.sep}df_bad_ics_task-{self.task}")
        # return(thisica)
        # thisica.apply(thisraw)

    def remove_ica_artifacts(self) -> None:
        """Remove noisy ICA components and reconstruct the data
        """
        logger.info("ICA_APPLY | Removing ICA artifacts from data")
        bad_muscle_idx = self.df_artifacts["bad_emg"][0]
        bad_eog_idx = self.df_artifacts["bad_eog"][0]
        bad_eog_idx = [int(x) for x in bad_eog_idx] if isinstance(bad_eog_idx, list) else [bad_eog_idx]
        # Supplementary? ICA muscles artifcat removal
        if self.rm_emg:
            self.ica.exclude.extend(bad_muscle_idx)

        if self.rm_eog:
            self.ica.exclude.extend(bad_eog_idx)

        self.ica.apply(self.raw)

        bad_components_idx = list(set(self.ica.exclude))
        # pdb.set_trace()
        logger.info(f"ICA_APPLY | ✓ Artifacts removed - {bad_components_idx if bad_components_idx else 'No components excluded'}")

    def resampling(self) -> None:
        sfreq_original = self.raw.info['sfreq']
        logger.info(f"RESAMPLING | Resampling from {sfreq_original}Hz to {self.sfreq}Hz")
        # logger.info(f"Resampling to {self.sfreq}")
        self.raw.resample(self.sfreq)
    
    def filtering(self) -> None:
        logger.info(f"FILTERING | Applying default FIR bandpass filter: {self.l_freq}-{self.h_freq}Hz")
        # logger.info(f"Applying default FIR filter from {self.l_freq} to {self.h_freq}.")
        self.raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
        logger.info(f"FILTERING | ✓ Completed")

    def interpolate_bad_chs(self) -> None:
        if not len(self.raw.info['bads']) == 0:
            logger.info(f"INTERPOLATION | Interpolating {len(self.raw.info['bads'])} bad channels: {self.raw.info['bads']}")
            self.raw.interpolate_bads(reset_bads=True)
            logger.info("INTERPOLATION | ✓ Bad channels interpolated")
        else:
            logger.info("INTERPOLATION | ✓ No bad channels to interpolate")
            # logger.info("No bad channels to be interpolated.")

    def notch_filtering(
        self,
        notch_freqs: Sequence[float] = (50, 100, 150, 200, 250),
    ) -> None:
        """
        Remove line-noise frequencies from ``self.raw`` in place.

        Parameters
        ----------
        notch_freqs : sequence of float
            Frequencies, in Hz, passed to ``Raw.notch_filter``.

        Returns
        -------
        None
            The method updates ``self.raw`` directly.

        Expected outcome
        ----------------
        ``self.raw`` retains its channels and duration, with the requested
        narrow-band frequencies attenuated.
        """
        logger.info(
            f"NOTCH_FILTER | Applying notch filters at {tuple(notch_freqs)} Hz"
        )
        # logger.info("Notch filtering at 50 Hz to remove line noises.")
        self.raw.notch_filter(notch_freqs)
        logger.info("NOTCH_FILTER | ✓ Completed")

    def rereferencing(self) -> None:
        """
        A wrapper of the .set_eeg_reference, handling specifically for the Biosemi data collected in our lab.

        Our data sometimes consist of 32/64 channels, and were with 6 external electrodes.
        
        Example
        -------
        >>> # Tidy and average to averaged mastoids
        >>> this_epoch = tidy_and_reref(this_epoch, "mastoid")
        """
        logger.info(f"REREFERENCING | Re-referencing to {self.ref_method} reference")
        # logger.info(f"Referencing with {self.ref_method}")
        # ----------------------------------------------------
        # Define mastoid and EOG channels 
        # EXG3, EXG4: Left and right
        # EXG5, EXG6: Top and bottom
        # raweog_list = ['EXG3', 'EXG4', 'EXG5', 'EXG6']
        mastoid_list = ['EXG1', 'EXG2']
        unused_list = ['EXG7', 'EXG8']
        # Drop the mastoid in case if no mastoid referencing is needed
        # this_epoch.add_reference_channels(mastoid_list)

        # ----------------------------------------------------
        # Re-reference to averaged mastoids
        match self.ref_method:
            case "mastoid":
                self.raw.set_eeg_reference(ref_channels=mastoid_list)
            case "average":
                self.raw.set_eeg_reference("average")
            case "REST":
                self.raw.set_eeg_reference("REST")

        self.raw.drop_channels(mastoid_list)
        self.raw.drop_channels(unused_list)
        logger.info(f"REREFERENCING | ✓ Completed, mastoid and unused channels dropped")

    def cropping(self, crop_start, crop_end) -> None:
        """Cropping of the raw data
        
        Wrapper of Raw.crop()
        """
        logger.info(f"CROPPING | Cropping from {crop_start}s to {crop_end}s")
        # logging.info(f"Cropping the recording from {crop_start}s to {crop_end}s")
        self.raw.crop(crop_start, crop_end)
        logger.info(f"CROPPING | ✓ Completed")

    def save_pkl(self, fname):
        """
        Save the Preprocessing object as a pickle file.

        Parameters
        ----------
        fname : str
            The filename to save the Preprocessing object.
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_pkl(cls, fname):
        """
        Load a Preprocessing object from a pickle file.
        Parameters
        ----------
        fname : str
            The filename to load the Preprocessing object from.
        """
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    # def save_class(self) -> None:
    #     """Save the present class
    #     """
    #     fname_output_full = os.path.join(self.dir_output, self.fname_output)
    #     logger.info(f"Saving the .raw file to {fname_output_full}.")
    #     self.raw.save(fname_output_full)

    def pipeline_intermediate():
        pass
    # def fit(self):
    #     self.fname_output = self.set_fname_output()
    #     self.raw = mne.io.read_raw_bdf(self.fname, preload=True)

    #     # Resampling
    #     self.raw = self.raw.resample()

    #     # Load list of bad electrodes from visual inspection
    #     self.raw = self.load_vs_bad_chs()

    #     # Referencing to averaged mastoids
    #     self.raw = tidy_and_reref(self.raw, "average")

    #     # Notch filtering to remove line noise in Hong Kong (50 Hz)
    #     if self.is_notch == True:
    #         self.raw = self.notch_filtering()

    #     # Fit ICA and apply ICA on the original data
    #     thisica = self.fit_ICA()
    #     thisica.apply(self.raw)

    #     # Interpolate bad channels, if necessary
    #     self.raw = self.interpolate_bad_chs()

    #     # Save intermediate files
    #     # self.raw.save(self.fname_output)

    #     # Filtering
    #     self.raw = self.filtering()

    #     # input meta data
    #     # self.raw.info['description'] = f"bad_eog_idx: {bad_eog_idx}"
    #     self.raw.info['subject_info']['his_id'] = self.sbjcode
        
    #     if self.fname_output == None:
    #         return(self.raw)
    #     else:
    #         self.raw.save(self.fname_output)
    #     logger.info(f"Finished preprocessing for subject {self.sbjcode}, session {self.ses}, task {self.task}")

        
        








    


