# For storing functions
import sys, os, re, warnings
import numpy as np

# Plot related library
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
from mne.viz import plot_topomap

# import pandas as pd
import logging
logger = logging.getLogger(__name__)

# For file storage
import pickle

LINE_FREQ_HK = 50

# Optional prep usage
# from pyprep.prep_pipeline import PrepPipeline

def create_bids_compatible_derivative(dir_bids, derivative_desc, modality_desc):
    """
    Create a BIDS-compatible derivative directory structure.
    
    Parameters
    ----------
    dir_bids: str
        Path to the BIDS root directory.
    derivative_name: str
        Name of the derivative dataset.
    modality_name: str
        Should be bids-compatible, or some custom class (e.g., 'preprocessing-obj').
    
    Examples
    --------
    >>> create_bids_compatible_derivative(dir_bids, "preproc-intermediate", "preprocessing-obj")
    The folder will then be created as: "dir_bids/derivative/preproc-intermediate/sub-<sub_label>/ses-<ses_label>/preprocessing-obj"
    where the `sub_label` and `ses_label` are extracted from the BIDS folder structure.
    """
    list_dir_sub = list_files(dir_bids, "sub-.*")
    for dir_sub in list_dir_sub:
        list_dir_sub_ses = list_files(dir_sub, "ses-.*")
        for dir_sub_ses in list_dir_sub_ses:
            list_dir_sub_mod = list_files(dir_sub_ses, "")
            for dir_sub_ses_mod in list_dir_sub_mod:
                sub_label = get_bids_fname_tag(dir_sub_ses_mod, "sub")
                ses_label = get_bids_fname_tag(dir_sub_ses_mod, "ses")
                # dir_to_create = dir_sub_ses_mod.split('eeg_bids_speech-tracking/')[-1].split('/eeg')[0]
                dir_to_create = os.path.join(dir_bids, "derivative", derivative_desc, f"sub-{sub_label}", f"ses-{ses_label}", modality_desc)
                # os.removedirs(dir_to_create)
                if not os.path.exists(dir_to_create):
                    os.makedirs(dir_to_create)

def parse_bids_fname(filename: str) -> dict:
    """
    Parse a BIDS filename into key-value pairs.
    
    Args:
        filename (str): BIDS filename
        
    Returns:
        dict: Key-value pairs from filename
        
    Example:
        >>> parse_bids_filename("sub-01_task-rest_run-1_eeg.set")
        {'sub': '01', 'task': 'rest', 'run': '1', 'extension': '.set'}
    """
    # Get basename and extension
    parts = filename.split('.')
    basename = parts[0]
    extension = '.' + '.'.join(parts[1:]) if len(parts) > 1 else ''
    
    # Parse key-value pairs
    result = {}
    for entity in basename.split('_'):
        if '-' in entity:
            key, value = entity.split('-', 1)
            result[key] = value
            
    if extension:
        result['extension'] = extension

    if "desc" in result.keys():
        dict_desc = parse_desc(result['desc'])
        result["desc"] = dict_desc

    return result

def join_bids_fname(bids_dict: dict) -> str:
    """
    Create a BIDS-compatible filename from a dictionary of key-value pairs.
    
    Args:
        bids_dict (dict): Dictionary of BIDS entities
        
    Returns:
        str: BIDS-compatible filename without extension
        
    Example:
        >>> join_bids_filename({'sub': '01', 'task': 'rest', 'run': '1'})
        'sub-01_task-rest_run-1'
    """
    # Remove extension if present
    bids_dict = {k: v for k, v in bids_dict.items() if k != 'extension'}
    
    list_to_join = []
    for k, v in bids_dict.items():
        if isinstance(v, dict):
            # If value is a dictionary, join its items
            list_to_join.append(f"desc-{'-'.join(f'{sub_k}-{sub_v}' for sub_k, sub_v in v.items())}")
        else:
            # Otherwise, just use the key-value pair
            list_to_join.append(f"{k}-{v}")

    return '_'.join(list_to_join)

def parse_desc(this_desc):
    list_this_desc = this_desc.split('-')

    desc_dict = {}
    for i in range(0, len(list_this_desc), 2):
        key = list_this_desc[i]
        value = list_this_desc[i + 1]
        desc_dict[key] = value

        # if i + 1 < len(list_this_desc):
        #     key = list_this_desc[i]
        #     value = list_this_desc[i + 1]
        #     desc_dict[key] = value
        # else:
        #     desc_dict[list_this_desc[i]] = None
    return desc_dict

def list_files(datadir, pattern):
    """ List files with desired pattern using regular expression
    Parameters
    --------
    datadir: str
        Data directory
    pattern: str 
        Pattern in regular expression
    
    Returns
    --------
    filelist: list
        list of files with desired pattern.
    """
    filelist = [f for f in os.listdir(datadir) if re.match(pattern, f)]
    filelist = [os.path.join(datadir, filename) for filename in filelist]
    return (filelist)

get_funcname = lambda n=0: sys._getframe(n + 1).f_code.co_name

def gen_subplot_grid(num_row, num_col):
    gs = gridspec.GridSpec(num_row, num_col)
    

def get_lab_fname_meta(fname):
    """
    This function aims to extract the eyes condition, subject code, session and block number
    from files named according to HKPolyU's NLCLAB rule.
    
    e.g. EyesClosedNoTask-S281-S1-B1.bdf
    
    Update 20230213:
    Wrap the fname with os.path.normpath, replace the delimiter by os.sep to adapt the function to both windows and linux.
    
    Parameters
    ----------------
    fname: str
        filename of the .bdf file
    """
    eyescond, sbjcode, snum, bnum = os.path.normpath(fname).split(os.sep)[-1].split('.')[0].split('-')[0:4]
    only_fname = fname.split(os.sep)[-1].split('.')[0]
    return(eyescond, sbjcode, int(snum[1:]), int(bnum[1:]), only_fname)

def get_bids_fname_meta(fname):
    """
    Return
    --------
    eyescond, sbjcode, setting, snum, bnum, only_fname
    
    """
    sbjcode = re.search(r'sub-[a-zA-Z0-9]*', fname).group().split('-')[-1]
    eyescond = re.search(r'task-[a-zA-Z0-9]*', fname).group().split('-')[-1]
    setting = re.search(r'setting-[a-zA-Z0-9]*', fname).group().split('-')[-1]
    snum = re.search(r'ses-[a-zA-Z0-9]*', fname).group().split('-')[-1]
    bnum = re.search(r'blk-[a-zA-Z0-9]*', fname).group().split('-')[-1]
    only_fname = fname.split(os.sep)[-1].split('.')[0]
    return eyescond, sbjcode, setting, int(snum), int(bnum), only_fname

def get_bids_fname_tag(fname, tag):
    """
    Return the value of a particular BIDS tag.

    Return
    ------
    >>> fname = "df_task-rest"
    >>> get_bids_fname_tag(fname, "task")
    >>> "rest"
    """
    re_pattern = f"{tag}-[a-zA-Z0-9\-]*"
    thisre = re.search(re_pattern, fname).group()
    if len(re.findall('-', thisre)) == 1:
        thisattr = thisre.split('-')[-1]
    elif len(re.findall('-', thisre)) > 1: #preserve all later parts
        thisattr = "-".join(thisre.split('-')[1:len(thisre)])
    return thisattr

def prep_to_df(this_prep):
    """
    Given a fitted PREP object, output a pandas dataframe
    """
    thisdf = pd.DataFrame(columns=['SubjectCode', 'Channel', 'is_bad_before_robust_ref', 'is_bad_after_robust_ref', 'is_bad_after_interp'])
    thisdf['Channel'] = pd.Series(this_prep.raw.info['ch_names'])
    thisdf['SubjectCode'] = this_prep.raw.info['subject_info']['his_id']
    thisdf['is_bad_before_robust_ref'] = 0
    thisdf['is_bad_after_robust_ref'] = 0
    thisdf['is_bad_after_interp'] = 0

    for ch_name in this_prep.noisy_channels_original["bad_all"]:
        thisdf.loc[(thisdf.Channel==ch_name), 'is_bad_before_robust_ref'] = 1

    for ch_name in this_prep.interpolated_channels:
        thisdf.loc[(thisdf.Channel==ch_name), 'is_bad_after_robust_ref'] = 1

    for ch_name in this_prep.still_noisy_channels:
        thisdf.loc[(thisdf.Channel==ch_name), 'is_bad_after_interp'] = 1
    
    return(thisdf)

import configparser

def save_config(filename, section, **params):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Add a section to the config
    config[section] = {}

    # Loop through the parameters and add them to the section
    for key, value in params.items():
        config[section][key] = str(value)

    # Write the configuration to a file
    with open(filename, 'a+') as configfile:
        config.write(configfile)

def load_config(filename, section):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(filename)

    # Get the parameters from the specified section
    if config.has_section(section):
        return {key: config.get(section, key) for key in config.options(section)}
    else:
        raise Exception(f"Section {section} not found in the configuration file.")

def load_preprocessing_config(filename, section):
    config = configparser.ConfigParser()
    config.read(filename)

    if config.has_section(section):
        dict_options = {
            "sfreq": config.getint(section, 'sfreq'),
            "rm_eog": config.get(section, 'rm_eog'),
            "rm_emg": config.getboolean(section, 'rm_emg'),
            "filtertype": config.get(section, 'filtertype'),
            "avgref": config.getboolean(section, 'avgref'),
            "is_prep": config.getboolean(section, 'is_prep')
        }
    return dict_options

def preprocess_bdf_compute_ica(fname, dir_output=None, sfreq=512, filtertype='highpass', is_prep=False):
    """
    Computing ICA solution, suitable for 32 and 64 channels.
        
    Forked from preprocess_bdf_prep
    Ver 20240229

    Updates
    20240229 - The PREP pipeline and the ICA used a fixed random seed to guarantee reproducibility
    20240306 - Change is_prep default to False, and make it optional to remove emg (rm_emg)

    Parameters
    ------------------
    filename: str
        filename of the input .bdf.
    outdir: str
        output directory. set to None if dont want to save the preprocessed file.
        
        
    Return
    --------------------
    Return 0 if file already exists, else return 1.
    """
    # sfreq = 512
    # rm_eog = 'hardz'
    # filtertype = 'bandpass'
    # avgref = True
    # outdir = None
    # path_raw_data = '../data/_RAW/_Longitudinal'
    # filename = list_files(path_raw_data, f'EyesClosedNoTask-P01-S182-B1.bdf')[0]
    # print(filename)
        
    # ----------------------------------------------------
    # Step 0: Initialize
    eyescond, sbjcode, snum, bnum, only_fname = get_lab_fname_meta(fname)
    
    # Create a .txt file storing the parameters
    # 
    # sfreq
    # rm_eog
    # rm_emg
    # filtertype
    # avgref
    # is_prep


    fname_output = f"{dir_output}/sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_mne-ica.fif"
    
    if os.path.isfile(fname_output):
        logger.info(f"Skipped | sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_setting-preprocessed-raw")
        return (0, 0, 0)
    
    # ----------------------------------------------------
    # Step 1: Read bdf file with specified montage file
    # logger.info(f"Beginning preprocessing of sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_setting-preprocessed-raw")
    
    thisraw = mne.io.read_raw_bdf(fname, verbose=True)
    thisraw.load_data(verbose=True)
    
    # Check the number of channels
    if (thisraw.info['nchan'] >= 32) and (thisraw.info['nchan'] < 64):
        eeg_nchan = 32
    elif (thisraw.info['nchan'] >= 64):
        eeg_nchan = 64

    # Add necessary info
    thisraw.info['subject_info'] = {'his_id': sbjcode}

    # ----------------------------------------------------
    # Step 2: resample to 512 Hz
    if not sfreq == None:
        thisraw.resample(sfreq)

    # ----------------------------------------------------
    # Step 3: define mastoid and EOG channels 
    # EXG3, EXG4: Left and right
    # EXG5, EXG6: Top and bottom
    raweog_list = ['EXG3', 'EXG4', 'EXG5', 'EXG6']
    mastoid_list = ['EXG1', 'EXG2']
    unused_list = ['EXG7', 'EXG8', 'Status']
    # Drop the mastoid in case if no mastoid referencing is needed
    thisraw.drop_channels(mastoid_list)
    thisraw.drop_channels(unused_list)

    # ----------------------------------------------------
    # Step 4: Create HEOG and VEOG from original EOG channels and add it to the data
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
        eeg_nchan = 32
        thisraw.set_montage(mne.channels.make_standard_montage('biosemi32'))
    elif (thisraw.info['nchan'] >= 64):
        eeg_nchan = 64
        thisraw.set_montage(mne.channels.make_standard_montage('biosemi64'))

    # ----------------------------------------------------
    # Step 5: PREP pipeline
    if is_prep:
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(LINE_FREQ_HK, thisraw.info['sfreq'] / 2, LINE_FREQ_HK),
        }
        from pyprep.prep_pipeline import PrepPipeline
        prep = PrepPipeline(thisraw.copy(), prep_params, thisraw.get_montage(), ransac=False, random_state=20240229)
        prep.fit()

        thisraw = prep.raw.copy()
        df_prep = prep_to_df(prep)
        if not dir_output == None:
            dir_prep_out = f"{dir_output}{os.sep}log-PREP"
            if not os.path.isdir(dir_prep_out):
                os.mkdir(dir_prep_out)
            df_prep.to_feather(f"{dir_prep_out}{os.sep}df_log-PREP_sub-{sbjcode}")
            df_prep.to_excel(f"{dir_prep_out}{os.sep}df_log-PREP_sub-{sbjcode}.xlsx", index=False)

        # After running the PREP pipeline, there is a chance that some channels are being labeled as bad
        # after interpolation, we need to manually de-select them in order to continue artifact rejection.
        thisraw.info['bads'] = []

    # ----------------------------------------------------
    # Step 6: Filtering
    # 1. Non-causal filtering affect the ERP onset, it might causes it to appear earlier than it seems.
    # 2. Actually, lower cut-off is desirable. 0.01-0.05 is preferred.
    # 3. We are using FIR filter here.

    if (filtertype == 'highpass'):
        thisraw.filter(l_freq=1, h_freq=None, method='fir')
    elif (filtertype == 'bandpass'):
        thisraw.filter(l_freq=1, h_freq=45, method='fir')
    elif (filtertype == 'bandpass-trans'):
        thisraw.filter(l_freq=1, h_freq=45, l_trans_bandwidth=0.1, method='fir')

    # ----------------------------------------------------
    # Step 7: ICA eyes artifcat removal
    # thisica = mne.preprocessing.ICA(method='picard', fit_params=dict(extended=True), random_state = 20240229, n_components=len(mne.pick_types(thisraw.info, eeg=True))).fit(thisraw)
    thisica = mne.preprocessing.ICA(method='picard', fit_params=dict(extended=True), random_state = 20240229).fit(thisraw)
    thisica.save(fname_output)
    


def preprocess_bdf_prep(fname, dir_output=None, sfreq=512, rm_eog='hardz', rm_emg=False, filtertype='highpass', avgref=True, is_prep=False, dir_input_ica=None, is_notch=True):
    """
    Preprocessing of bdf files, suitable for 32 and 64 channels.
    
    Former preprocess_longit_bdf
    
    Ver 20240430

    Updates
    20240229 - The PREP pipeline and the ICA used a fixed random seed to guarantee reproducibility
    20240306 - Change is_prep default to False, and make it optional to remove emg (rm_emg)
    20240430 - Update such that we can now better control on whether we should output a file

    Parameters
    ------------------
    fname: str
        filename of the input .bdf.
    dir_output: the desired output directory
        default is None. 
    sfreq: sampling frequency
    rm_eog: different ways of removing eyes artifacts
    rm_emg: whether to automatically detect and remove EMG artifacts using MNE's `find_bads_emg` (check the algorithm before you enable it)
    filtertype: filters to be applied
    avgref: whether to conduct average re-referencing
    is_prep: whether to run the PREP pipeline
    dir_input_ica: str
        Default is None. if not `None`, save the ICA decomposition results in to the input folder (this is to ensure reproducibility since ICA is not deterministic)
    is_notch: bool
        Whether to apply a notch filter at 50 Hz and its harmonics upto 250Hz. Default and recommend to be True.
        
    
    0. Initialize
    1. Read bdf file with specified montage file
    2. Resample to 512 Hz
    3. Define mastoid and EOG channels
    4. Create HEOG and VEOG from original EOG channels and add it to the data, identify the correct montage
    5. Run the PREP pipeline
    6. Filtering
    7. ICA eyes/muscles artifacts removal (removing muscle artifacts is optional)
    8. ICA muscles artifacts removal (optional)
    9. Re-referencing
    10. Save the preprocessed file

    
    Return
    --------------------
    Return 0 if file already exists, else return 1.
    """        
    # ----------------------------------------------------
    # Step 0: Initialize
    eyescond, sbjcode, snum, bnum, _ = get_lab_fname_meta(fname)
    
    # Create a .txt file storing the parameters
    # 
    # sfreq
    # rm_eog
    # rm_emg
    # filtertype
    # avgref
    # is_prep

    # Define output filename if dir_output is provided
    if not dir_output == None:
        fname_output = f"{dir_output}{os.sep}sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_mne-raw.fif"
        dir_ica = f"{dir_output}{os.sep}ica"
        if not os.path.isdir(dir_ica):
            os.mkdir(dir_ica)
        fname_output_ica = f"{dir_ica}{os.sep}sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_mne-ica.fif"
        
        # If the file exists in the dir_output, skip the preprocessing
        if os.path.isfile(fname_output):
            logger.info(f"Skipped | sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_setting-preprocessed-raw")
            return (0, 0, 0)
    
    # ----------------------------------------------------
    # Step 1: Read bdf file with specified montage file
    # logger.info(f"Beginning preprocessing of sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_setting-preprocessed-raw")
    
    thisraw = mne.io.read_raw_bdf(fname, verbose=True)
    thisraw.load_data(verbose=True)
    
    # Check the number of channels
    if (thisraw.info['nchan'] >= 32) and (thisraw.info['nchan'] < 64):
        eeg_nchan = 32
    elif (thisraw.info['nchan'] >= 64):
        eeg_nchan = 64

    # Add necessary info
    thisraw.info['subject_info'] = {'his_id': sbjcode}

    # ----------------------------------------------------
    # Step 2: resample to 512 Hz
    if not sfreq == None:
        thisraw.resample(sfreq)

    # ----------------------------------------------------
    # Step 3: define mastoid and EOG channels 
    # EXG3, EXG4: Left and right
    # EXG5, EXG6: Top and bottom
    raweog_list = ['EXG3', 'EXG4', 'EXG5', 'EXG6']
    mastoid_list = ['EXG1', 'EXG2']
    unused_list = ['EXG7', 'EXG8', 'Status']
    # Drop the mastoid in case if no mastoid referencing is needed
    thisraw.drop_channels(mastoid_list)
    thisraw.drop_channels(unused_list)

    # ----------------------------------------------------
    # Step 4: Create HEOG and VEOG from original EOG channels and add it to the data
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
        eeg_nchan = 32
        thisraw.set_montage(mne.channels.make_standard_montage('biosemi32'))
    elif (thisraw.info['nchan'] >= 64):
        eeg_nchan = 64
        thisraw.set_montage(mne.channels.make_standard_montage('biosemi64'))

    # ----------------------------------------------------
    # Step 5: PREP pipeline
    if is_prep:
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(LINE_FREQ_HK, thisraw.info['sfreq'] / 2, LINE_FREQ_HK),
        }
        from pyprep.prep_pipeline import PrepPipeline
        prep = PrepPipeline(thisraw.copy(), prep_params, thisraw.get_montage(), ransac=False, random_state=20240229)
        prep.fit()

        thisraw = prep.raw.copy()
        df_prep = prep_to_df(prep)
        if not dir_output == None:
            dir_prep_out = f"{dir_output}{os.sep}log-PREP"
            if not os.path.isdir(dir_prep_out):
                os.mkdir(dir_prep_out)
            df_prep.to_feather(f"{dir_prep_out}{os.sep}df_log-PREP_sub-{sbjcode}")
            df_prep.to_excel(f"{dir_prep_out}{os.sep}df_log-PREP_sub-{sbjcode}.xlsx", index=False)

        # After running the PREP pipeline, there is a chance that some channels are being labeled as bad
        # after interpolation, we need to manually de-select them in order to continue artifact rejection.
        thisraw.info['bads'] = []

    # ----------------------------------------------------
    # Step 6: Filtering
    # 1. Non-causal filtering affect the ERP onset, it might causes it to appear earlier than it seems.
    # 2. Actually, lower cut-off is desirable. 0.01-0.05 is preferred.
    # 3. We are using FIR filter here.

    if is_notch == True:
        thisraw.notch_filter(np.arange(50, 251, 50))

    if (filtertype == 'highpass'):
        thisraw.filter(l_freq=1, h_freq=None, method='fir', verbose=True)
    elif (filtertype == 'bandpass'):
        thisraw.filter(l_freq=1, h_freq=45, method='fir')
    elif (filtertype == 'bandpass-trans'):
        thisraw.filter(l_freq=1, h_freq=45, l_trans_bandwidth=0.1, h_trans_bandwidth=1, method='fir', verbose=True)

    # ----------------------------------------------------
    # Step 7: ICA eyes artifcat removal
    # thisica = mne.preprocessing.ICA(method='picard', fit_params=dict(extended=True), random_state = 20240229, n_components=len(mne.pick_types(thisraw.info, eeg=True))).fit(thisraw)

    # If dir_ica is provided, check if ICA file exists in the directory and load it.
    if not dir_input_ica == None:
        fname_ica = list_files(dir_input_ica, f"sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_mne-ica.fif")[0]
        # pdb.set_trace()
        thisica = mne.preprocessing.read_ica(fname_ica)
    else:
        # different seeds
        thisica = mne.preprocessing.ICA(method='picard', fit_params=dict(extended=True), random_state = 20240229).fit(thisraw)
        # Save ICA component when dir_output is provided
        if not dir_output == None:
            thisica.save(fname_output_ica)

    if rm_eog == 'default':
        # ----------------------------------------------------
        # Default: adaptive z-score
        logger.info('-----------ICA eyes artifacts removal by adaptive z-score---------------')
        bad_eog_idx, _ = thisica.find_bads_eog(thisraw)
        # thisica.plot_scores(scores_eog, exclude=eog_inds)
        # thisica.plot_sources(thisraw, exclude=eog_inds)
        # Register the index of artifacts
        # Remove the above registered artifacts
    elif rm_eog == 'hardz':
        # ----------------------------------------------------
        # Hard z-score thresholding
        from scipy.stats import spearmanr
        # logger.info('-----------ICA eyes artifacts removal by hard z thresholding-----------------')
        hardthres = 0.3 # to match the previous matlab preprocessing procedure
        def unmix_to_ICA(thisraw, thisica):
            testica = thisica.copy()
            thisdata = thisraw.get_data()[0:eeg_nchan,:]
            processdata = np.divide(thisdata[0:eeg_nchan,:], testica.pre_whitener_)
            processdata -= testica.pca_mean_[:, None]
            unmix_mat = np.dot(testica.unmixing_matrix_, testica.pca_components_[0:testica.n_components_, :])
        #     print("unmix mat: {}".format(np.shape(unmix_mat)))
            proj_data = np.dot(unmix_mat, processdata)
            return(unmix_mat, proj_data)
        #     print("proj data: {}".format(np.shape(proj_data)))

        def get_filtered_eog(thisraw):
            thisraw2 = thisraw.copy()
            thisraw2.filter(1, 45, l_trans_bandwidth=0.1, h_trans_bandwidth=10, method='fir', picks='eog',
                            verbose=False)
            # eog = thisraw2.pick_types(eog=True).get_data()
            eog = thisraw2.pick(['eog']).get_data()
            return(eog)
        # Filter and get eyes data (default did not filter eog channel)
        eog = get_filtered_eog(thisraw)

        _, proj_data = unmix_to_ICA(thisraw, thisica)
        from scipy.stats import zscore
        v_r_vec = []
        h_r_vec = []
        for i in range(np.shape(proj_data)[0]):
            v_r, v_pval = spearmanr(proj_data[i,:], eog[0,:])
            h_r, h_pval = spearmanr(proj_data[i,:], eog[1,:])
            v_r_vec.append(v_r)
            h_r_vec.append(h_r)
        def hard_zscore(v_r_vec, h_r_vec, threshold=3):
            v_r_zscore = np.abs(zscore(v_r_vec))
            h_r_zscore = np.abs(zscore(h_r_vec))
            print(v_r_zscore)
            print(h_r_zscore)
            v_r_idx = np.where(v_r_zscore >= threshold)
            h_r_idx = np.where(h_r_zscore >= threshold)
            hard_eog_arr = np.unique(np.append(v_r_idx, h_r_idx))
            return(hard_eog_arr.tolist())

        bad_eog_idx = hard_zscore(v_r_vec, h_r_vec)
    elif rm_eog == 'hardthres':
        # ----------------------------------------------------
        # Hard correlation thresholding
        from scipy.stats import spearmanr
        logger.info('-----------ICA eyes artifacts removal by hard thresholding at rho > 0.3-----------------')
        hardthres = 0.3 # to match the previous matlab preprocessing procedure
        def unmix_to_ICA(thisraw, thisica):
            testica = thisica.copy()
            thisdata = thisraw.get_data()[0:eeg_nchan,:]
            processdata = np.divide(thisdata[0:eeg_nchan,:], testica.pre_whitener_)
            processdata -= testica.pca_mean_[:, None]
            unmix_mat = np.dot(testica.unmixing_matrix_, testica.pca_components_[0:testica.n_components, :])
        #     print("unmix mat: {}".format(np.shape(unmix_mat)))
            proj_data = np.dot(unmix_mat, processdata)
            return(unmix_mat, proj_data)
        #     print("proj data: {}".format(np.shape(proj_data)))

        def get_filtered_eog(thisraw):
            thisraw2 = thisraw.copy()
            thisraw2.filter(1, 45, l_trans_bandwidth=0.1, h_trans_bandwidth=10, method='fir', picks='eog',
                            verbose=False)
            # eog = thisraw2.pick_types(eog=True).get_data()
            eog = thisraw2.pick(['eog']).get_data()
            return(eog)
        # Filter and get eyes data (default did not filter eog channel)
        eog = get_filtered_eog(thisraw)

        unmix_mat, proj_data = unmix_to_ICA(thisraw, thisica)
        bad_eog_idx= []
        for i in range(np.shape(proj_data)[0]):
            v_r, v_pval = spearmanr(proj_data[i,:], eog[0,:])
            h_r, h_pval = spearmanr(proj_data[i,:], eog[1,:])
            if (np.abs(v_r) >= 0.3) | (np.abs(h_r) >= 0.3):
                bad_eog_idx.append(i)
    #                     print("Suspected channel: {}".format(i))


    # ----------------------------------------------------
    # Find ICA components corresponding to muscle artifacts
    bad_muscle_idx, _ = thisica.find_bads_muscle(thisraw)

    # pdb.set_trace()
    # Remove ICA components of eyes and muscle artifacts
    thisica.exclude.extend(bad_eog_idx)

    # Step 8: ICA muscles artifcat removal
    if rm_emg:
        thisica.exclude.extend(bad_muscle_idx)
    
    # Transform back to data with "clean" ICA components
    thisica.apply(thisraw)

    # ----------------------------------------------------
    # Step 9: Re-referencing
    if avgref == True:
        thisraw.set_eeg_reference('average')

    # ----------------------------------------------------
    # Step 10: Save the file
    # pdb.set_trace()
    if not dir_output == None:
        # thisraw.save(os.path.join(outdir, raw_filename), overwrite=True)
        thisraw.save(fname_output)
        # thisraw.save(f"{dir_output}/sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}_setting-preprocessed-raw.fif", overwrite=True)
        logger.info(f"Success | sub-{sbjcode}_ses-{str(snum).zfill(2)}_blk-{str(bnum).zfill(2)}_task-{eyescond}")
    
    if is_prep:
        return 1, thisraw, df_prep
    else:
        return 1, thisraw, 0


def plot_multiwave(imfs, fs, subplot=True, plot_splines=False, time_offset=0, subplot_ylabel=None, subplot_style=None):
    """
    Plot utility method for plotting mulitple timeseries and their envelope splines with
    ``pylab``.

    Parameters
    ----------
    imfs : ndarray
        intrinsic mode functions

    fs: Sampling rates.

    subplot : bool, optional
        Whether to plot the IMFs in separate figures.

    plot_splines : bool, optional
        Whether to plot the envelope spline curves as well.

    subplot_ylabel: dict, optional
        keys are 0-indexed channel number
        values are string channel name
    """
    if subplot:
        fig, axs = plt.subplots(imfs.shape[0], 1, sharex=True)
        fig.suptitle("Intrinsic Mode Functions")

        # When the input is segmented waves of size (channel, epochs, samples)
        # The following condition will concatenate all the epochs
        if not imfs.ndim == 2:
            # concatenate along the second axis
            imfs = np.reshape(imfs, (imfs.shape[0], imfs.shape[1] * imfs.shape[2]))

        for i in range(imfs.shape[0]):
            # label = "IMF #%d" % (i+1) if (i+1) < imfs.shape[0] else "Residual"
            # print("Plotting", label)
            imf = imfs[i, :]
            times = time_offset + np.arange(0,np.shape(imf)[0]/fs, 1/fs)
            axs[i].plot(times, imf)
            if (subplot_style=='imfs'):
                axs[i].set_ylabel(subplot_ylabel[i], rotation=0)
                axs[i].set_xlabel('Time')
            # times = np.arange(0, np.shape(imf)[0]/fs, 1/fs)
            # axs[i].set_title(label)

            if plot_splines:
                print("plot_multiwave: plot_splines deprecated")
                # maxx, maxy, minx, miny = emd_find_extrema(imf)
                # maxs = emd_evaluate_spline(maxx, maxy)
                # mins = emd_evaluate_spline(minx, miny)
                # means = (maxs+mins)/2
                # axs[i].plot(maxs, "g--")
                # axs[i].plot(mins, "r--")
                # axs[i].plot(minx, miny, "rv")
                # axs[i].plot(maxx, maxy, "g^")
                # axs[i].plot(means, "b:")

    return(fig, axs)


def plot_raw_multiwave(data, fs, title='', subplot=True, time_offset=0, subplot_ylabel=None, subplot_style=None):
    """
    Plot utility method for plotting mulitple timeseries and their envelope splines with
    ``pylab``.

    Parameters
    ----------
    imfs : ndarray
        intrinsic mode functions

    fs: Sampling rates.

    subplot : bool, optional
        Whether to plot the IMFs in separate figures.

    plot_splines : bool, optional
        Whether to plot the envelope spline curves as well.

    subplot_ylabel: dict, optional
        keys are 0-indexed channel number
        values are string channel name
    """
    if subplot:
        fig, axes = plt.subplots(data.shape[0], 1, sharex=True, figsize=(20, 40))
        fig.suptitle(title, y=0.89)

        # When the input is segmented waves of size (channel, epochs, samples)
        # The following condition will concatenate all the epochs
        if not data.ndim == 2:
            # concatenate along the second axis
            data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

        for i in range(data.shape[0]):
            # label = "IMF #%d" % (i+1) if (i+1) < imfs.shape[0] else "Residual"
            # print("Plotting", label)
            ch_data = data[i, :]
            times = time_offset + np.arange(0,np.shape(ch_data)[0]/fs, 1/fs)
            axes[i].plot(times, ch_data)
            if not subplot_ylabel==None:
                axes[i].set_ylabel(subplot_ylabel[i], rotation=0)
                axes[i].yaxis.set_label_coords(-0.04,0.35)
                if i == data.shape[0] - 1:
                    axes[i].set_xlabel('Time (s)')
                
            # times = np.arange(0, np.shape(imf)[0]/fs, 1/fs)
            # axs[i].set_title(label)
    return(fig, axes)


def sliding_window(siglen, winlength, stride):
    """
    Create array of windows with size: (number of windows, window length)
    
    Parameters
    --------
    siglen: int
        length of your target time series
    winlength: int
        window length in terms of time samples
    stride: int
        step in terms of time samples
        
    Returns
    --------
    np.ndarray
        2d array of size (number of windows, window length)
    """
    overlap = stride/winlength
    # If overlap is same as stride (non-overlapping)
    if (overlap==1):
        num_window = siglen/winlength
    else:
        num_window = (siglen/winlength)/overlap-1
    num_of_windows = 3*(1/3)-1
    winarray = np.arange(winlength)[None,:] + stride*np.arange(num_window)[:,None]
    # testimfs[testwin[0,:]]
    # Errors occur when winlength or overlap isnt integer
    while winarray[-1,-1] >= siglen:
        warnings.warn("sliding_window: shrinking sliding window as window length/overlapping might not be integer")
        winarray = winarray[0:-1,:]

    return(winarray.astype(int))

def save_obj(obj, name):
    """
    Save file using pickle
    
    Parameters
    --------
    obj: any type
        object to be saved
    name: str
        filename, with/without extension
    """
    if name.endswith('.pkl'):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Load file with ''.pkl'
    
    Parameters
    --------
    name: str
        filename, with/without extension
    
    Returns
    --------
    The loaded object
    """
    if name.endswith('.pkl'):
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


def plot_maps_biosemi32(maps, title='', cmap='default', show_names=False, fontsize=18, is_save=False, outdir=None, vmin=None, vmax=None, is_show=False):
    """Plot clustered microstate maps
    Note that this function is specialized for biosemi32, for the channel location
    and names.
    
    User can adapt this function to other system by changing montage and providing
    channel names
    
    Parameters
    ----------
    maps : ndarray (n_channels, n_states)
        The maps resulted from _modkmeans
    cmap : colormap
        Default is RdBu_r
    show_names: bool
        Whether to show channel names, default is False
    fontsize: int
        Font size if show channel names, default = 18
    is_save: bool
        Save the figure as {title}.pdf
    outdir: str
        saving directory
    """
    n_states = np.shape(maps)[1]

    if cmap=='default':
        cmap = 'RdBu_r'

    plt.rcParams['font.size'] = fontsize
    fig, axes = plt.subplots(1, n_states, figsize=(n_states*4,4), squeeze=False)
    
    # The following line get 2d electrode position from mne v1.17
#     this_montage_pos = mne.channels.read_montage('biosemi32').get_pos2d()[0:32,0:2]
    
    # in mne v2.20, channels.make_standard_montage() is recommended
    # _pol_to_cart and _cart_to_sph is imported from mne.channels.layout to recreate the function
    # mne.channels.read_montage('biosemi32').get_pos2d() in mne v1.17 in order to get the correct
    # 2d positions
    m = mne.channels.make_standard_montage('biosemi32')
    this_montage_pos_3d = np.zeros([32, 3])
    for i, e in enumerate(list(m._get_ch_pos().values())):
        this_montage_pos_3d[i] = e[0:3]
    from mne.channels.layout import _pol_to_cart, _cart_to_sph
    this_montage_pos = _pol_to_cart(_cart_to_sph(this_montage_pos_3d)[:, 1:][:, ::-1])

    misc_dir = 'misc'
    try:
        chandict = load_obj(os.path.join(misc_dir, 'chandict.pkl'))
    except:
        chandict = load_obj('../misc/chandict.pkl')
    for k in range(0, n_states):
        im, cn = plot_topomap(maps[:,k], this_montage_pos, names=chandict.values(), 
                             show_names=show_names, cmap=cmap, axes=axes[0, k], show=False, vmin=vmin, vmax=vmax)
    if not title=='':
        plt.suptitle(title)
    plt.tight_layout()
    if is_show == True:
        plt.show()
    
    if is_save:
        if outdir==None:
            outdir = os.path.join(os.getcwd(), 'figure')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        plt.savefig(os.path.join(outdir, f'{title}.pdf'), bbox_inches='tight', format='pdf')
    return(fig, axes)


def plot_topomap_custom(chandata, size=6):
    """
    Plot topographical map given data of all channels at a single time instant. Forked from mne.viz.plot_topomap.
    
    Parameters
    ----------
    chandata: np.array (n_channels, 1)
        A n_channels x 1 vector which stores the values of all channels.
    size: int
        Size of the figure, in inches.
    """
    if (np.shape(chandata)[0] >= 32) and (np.shape(chandata)[0] < 64):
        eeg_nchan = 32
        m = mne.channels.make_standard_montage('biosemi32')
    elif (np.shape(chandata)[0] >= 64):
        eeg_nchan = 64
        m = mne.channels.make_standard_montage('biosemi64')

    this_montage_pos_3d = np.zeros([eeg_nchan, 3])
    for i, e in enumerate(list(m._get_ch_pos().values())):
        this_montage_pos_3d[i] = e[0:3]
    from mne.channels.layout import _pol_to_cart, _cart_to_sph
    this_montage_pos = _pol_to_cart(_cart_to_sph(this_montage_pos_3d)[:, 1:][:, ::-1])
    fig, ax = plt.subplots(figsize=(size, size))
    # vals = np.zeros(64)
    im, cn = plot_topomap(chandata, this_montage_pos, sphere = (0, 0, 0, 2.2), names=m.ch_names, sensors=" ", axes=ax, show=False, contours=False)
    return(fig, ax)