# %% Preprocess data #######################################################################################################
# The goal of this script is to preprocess the raw data, extract FAA, and prepare for data analysis
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Cell 1: Preprocessing, Continuous Cleaning, and Quality Reporting
import os
import time
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import autoreject
from autoreject import Ransac

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"
REFERENCE_METHOD = 'virtual_Cz'      
VIRTUAL_REF_CHANNELS = ['Cz', 'FCz', 'CPz', 'C1', 'C2'] 

PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")

SOURCE_DATA = PROJECT_FOLDER / 'bids' / 'sourcedata'
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids' / 'derivatives' / 'processed' / BATCH_NAME
CHANNEL_MAP_PATH = PROJECT_FOLDER / 'UvA_DiverseConfig_ChTypes.csv'

N_JOBS = -1

# ==============================================================================
# Event Dictionaries 
# ==============================================================================
EVENTS_JUSTFACES = {
    'PVV/Agema': 201, 'PVV/Wilders': 202, 'GL/Klaver': 211, 'GL/Timmermans': 212, 'Foil/Foil': 222,
}

EVENTS_STATEMENTSFACES = {
    'GL/Klaver/left/negative/Climate': 5, 'GL/Klaver/left/negative/EU': 6, 'GL/Klaver/left/negative/Gender equality': 7,
    'GL/Klaver/left/negative/Immigration': 8, 'GL/Klaver/left/negative/Integration': 9, 'GL/Klaver/left/negative/Vegetarianism': 10,
    'GL/Klaver/right/negative/Climate': 11, 'GL/Klaver/right/negative/EU': 12, 'GL/Klaver/right/negative/Gender equality': 13,
    'GL/Klaver/right/negative/Immigration': 14, 'GL/Klaver/right/negative/Integration': 15, 'GL/Klaver/right/negative/Vegetarianism': 16,
    'GL/Klaver/left/neutral/Climate': 17, 'GL/Klaver/left/neutral/EU': 18, 'GL/Klaver/left/neutral/Gender equality': 19,
    'GL/Klaver/left/neutral/Immigration': 20, 'GL/Klaver/left/neutral/Integration': 21, 'GL/Klaver/left/neutral/Vegetarianism': 22,
    'GL/Klaver/right/neutral/Climate': 23, 'GL/Klaver/right/neutral/EU': 24, 'GL/Klaver/right/neutral/Gender equality': 25,
    'GL/Klaver/right/neutral/Immigration': 26, 'GL/Klaver/right/neutral/Integration': 27, 'GL/Klaver/right/neutral/Vegetarianism': 28,
    'GL/Klaver/left/positive/Climate': 29, 'GL/Klaver/left/positive/EU': 30, 'GL/Klaver/left/positive/Gender equality': 31,
    'GL/Klaver/left/positive/Immigration': 32, 'GL/Klaver/left/positive/Integration': 33, 'GL/Klaver/left/positive/Vegetarianism': 34,
    'GL/Klaver/right/positive/Climate': 35, 'GL/Klaver/right/positive/EU': 36, 'GL/Klaver/right/positive/Gender equality': 37,
    'GL/Klaver/right/positive/Immigration': 38, 'GL/Klaver/right/positive/Integration': 39, 'GL/Klaver/right/positive/Vegetarianism': 40,
    'GL/Timmermans/left/negative/Climate': 41, 'GL/Timmermans/left/negative/EU': 42, 'GL/Timmermans/left/negative/Gender equality': 43,
    'GL/Timmermans/left/negative/Immigration': 44, 'GL/Timmermans/left/negative/Integration': 45, 'GL/Timmermans/left/negative/Vegetarianism': 46,
    'GL/Timmermans/right/negative/Climate': 47, 'GL/Timmermans/right/negative/EU': 48, 'GL/Timmermans/right/negative/Gender equality': 49,
    'GL/Timmermans/right/negative/Immigration': 50, 'GL/Timmermans/right/negative/Integration': 51, 'GL/Timmermans/right/negative/Vegetarianism': 52,
    'GL/Timmermans/left/neutral/Climate': 53, 'GL/Timmermans/left/neutral/EU': 54, 'GL/Timmermans/left/neutral/Gender equality': 55,
    'GL/Timmermans/left/neutral/Immigration': 56, 'GL/Timmermans/left/neutral/Integration': 57, 'GL/Timmermans/left/neutral/Vegetarianism': 58,
    'GL/Timmermans/right/neutral/Climate': 59, 'GL/Timmermans/right/neutral/EU': 60, 'GL/Timmermans/right/neutral/Gender equality': 61,
    'GL/Timmermans/right/neutral/Immigration': 62, 'GL/Timmermans/right/neutral/Integration': 63, 'GL/Timmermans/right/neutral/Vegetarianism': 64,
    'GL/Timmermans/left/positive/Climate': 65, 'GL/Timmermans/left/positive/EU': 66, 'GL/Timmermans/left/positive/Gender equality': 67,
    'GL/Timmermans/left/positive/Immigration': 68, 'GL/Timmermans/left/positive/Integration': 69, 'GL/Timmermans/left/positive/Vegetarianism': 70,
    'GL/Timmermans/right/positive/Climate': 71, 'GL/Timmermans/right/positive/EU': 72, 'GL/Timmermans/right/positive/Gender equality': 73,
    'GL/Timmermans/right/positive/Immigration': 74, 'GL/Timmermans/right/positive/Integration': 75, 'GL/Timmermans/right/positive/Vegetarianism': 76,
    'PVV/Agema/left/negative/Climate': 77, 'PVV/Agema/left/negative/EU': 78, 'PVV/Agema/left/negative/Gender equality': 79,
    'PVV/Agema/left/negative/Immigration': 80, 'PVV/Agema/left/negative/Integration': 81, 'PVV/Agema/left/negative/Vegetarianism': 82,
    'PVV/Agema/right/negative/Climate': 83, 'PVV/Agema/right/negative/EU': 84, 'PVV/Agema/right/negative/Gender equality': 85,
    'PVV/Agema/right/negative/Immigration': 86, 'PVV/Agema/right/negative/Integration': 87, 'PVV/Agema/right/negative/Vegetarianism': 88,
    'PVV/Agema/left/neutral/Climate': 89, 'PVV/Agema/left/neutral/EU': 90, 'PVV/Agema/left/neutral/Gender equality': 91,
    'PVV/Agema/left/neutral/Immigration': 92, 'PVV/Agema/left/neutral/Integration': 93, 'PVV/Agema/left/neutral/Vegetarianism': 94,
    'PVV/Agema/right/neutral/Climate': 95, 'PVV/Agema/right/neutral/EU': 96, 'PVV/Agema/right/neutral/Gender equality': 97,
    'PVV/Agema/right/neutral/Immigration': 98, 'PVV/Agema/right/neutral/Integration': 99, 'PVV/Agema/right/neutral/Vegetarianism': 100,
    'PVV/Agema/left/positive/Climate': 101, 'PVV/Agema/left/positive/EU': 102, 'PVV/Agema/left/positive/Gender equality': 103,
    'PVV/Agema/left/positive/Immigration': 104, 'PVV/Agema/left/positive/Integration': 105, 'PVV/Agema/left/positive/Vegetarianism': 106,
    'PVV/Agema/right/positive/Climate': 107, 'PVV/Agema/right/positive/EU': 108, 'PVV/Agema/right/positive/Gender equality': 109,
    'PVV/Agema/right/positive/Immigration': 110, 'PVV/Agema/right/positive/Integration': 111, 'PVV/Agema/right/positive/Vegetarianism': 112,
    'PVV/Wilders/left/negative/Climate': 113, 'PVV/Wilders/left/negative/EU': 114, 'PVV/Wilders/left/negative/Gender equality': 115,
    'PVV/Wilders/left/negative/Immigration': 116, 'PVV/Wilders/left/negative/Integration': 117, 'PVV/Wilders/left/negative/Vegetarianism': 118,
    'PVV/Wilders/right/negative/Climate': 119, 'PVV/Wilders/right/negative/EU': 120, 'PVV/Wilders/right/negative/Gender equality': 121,
    'PVV/Wilders/right/negative/Immigration': 122, 'PVV/Wilders/right/negative/Integration': 123, 'PVV/Wilders/right/negative/Vegetarianism': 124,
    'PVV/Wilders/left/neutral/Climate': 125, 'PVV/Wilders/left/neutral/EU': 126, 'PVV/Wilders/left/neutral/Gender equality': 127,
    'PVV/Wilders/left/neutral/Immigration': 128, 'PVV/Wilders/left/neutral/Integration': 129, 'PVV/Wilders/left/neutral/Vegetarianism': 130,
    'PVV/Wilders/right/neutral/Climate': 131, 'PVV/Wilders/right/neutral/EU': 132, 'PVV/Wilders/right/neutral/Gender equality': 133,
    'PVV/Wilders/right/neutral/Immigration': 134, 'PVV/Wilders/right/neutral/Integration': 135, 'PVV/Wilders/right/neutral/Vegetarianism': 136,
    'PVV/Wilders/left/positive/Climate': 137, 'PVV/Wilders/left/positive/EU': 138, 'PVV/Wilders/left/positive/Gender equality': 139,
    'PVV/Wilders/left/positive/Immigration': 140, 'PVV/Wilders/left/positive/Integration': 141, 'PVV/Wilders/left/positive/Vegetarianism': 142,
    'PVV/Wilders/right/positive/Climate': 143, 'PVV/Wilders/right/positive/EU': 144, 'PVV/Wilders/right/positive/Gender equality': 145,
    'PVV/Wilders/right/positive/Immigration': 146, 'PVV/Wilders/right/positive/Integration': 147, 'PVV/Wilders/right/positive/Vegetarianism': 148,
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def print_timestamp(message):
    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] {message}")

def guess_ext_and_reader(sub, task):
    eeg_folder = SOURCE_DATA / sub / 'eeg'
    if (eeg_folder / f'{sub}_task-{task}_eeg.fif.gz').exists():
        return 'fif.gz'
    elif (eeg_folder / f'{sub}_task-{task}_eeg.bdf').exists():
        return 'bdf'
    return None

def preprocess_raw_and_ica(in_file, channel_type_dict, ref_method, report, subject, task):
    """Loads, filters, references, runs ICA, and adds to HTML report."""
    raw = mne.io.read_raw(in_file, preload=True, verbose=False)
    
    channels_to_keep = list(channel_type_dict.keys())
    if 'Status' in raw.ch_names and 'Status' not in channels_to_keep:
        channels_to_keep.append('Status')
        
    raw.drop_channels([ch for ch in raw.ch_names if ch not in channels_to_keep])
    if 'Status' in raw.ch_names: raw.set_channel_types({'Status': 'stim'}, verbose=False)
    raw.set_channel_types(channel_type_dict, verbose=False)
    
    if ref_method == 'virtual_Cz':
        avail_refs = [ch for ch in VIRTUAL_REF_CHANNELS if ch in raw.ch_names]
        ref_data = raw.get_data(picks=avail_refs)
        virtual_data = np.mean(ref_data, axis=0, keepdims=True)
        
        info = mne.create_info(['Virtual_Ref'], raw.info['sfreq'], ch_types=['eeg'])
        virtual_raw = mne.io.RawArray(virtual_data, info, verbose=False)
        raw.add_channels([virtual_raw], force_update_info=True)
        raw.set_eeg_reference(ref_channels=['Virtual_Ref'], verbose=False)
        raw.drop_channels(['Virtual_Ref']) 
    elif ref_method == 'average':
        raw.set_eeg_reference('average', projection=False, verbose=False)
    else:
        ref_chans = [ref_method] if isinstance(ref_method, str) else ref_method
        raw.set_eeg_reference(ref_channels=ref_chans, verbose=False)

    raw.set_montage('biosemi64', match_case=False, on_missing='ignore')
    raw.filter(l_freq=1.0, h_freq=40.0, n_jobs=N_JOBS, verbose=False)
    
    print_timestamp(f'Running ICA for {task}')
    ica = mne.preprocessing.ICA(n_components=25, method='fastica', random_state=23)
    ica.fit(inst=raw, decim=3, verbose=False)

    eog_average = mne.preprocessing.create_eog_epochs(raw).average() if 'eog' in raw.get_channel_types() else None
    eog_components, eog_scores = ica.find_bads_eog(raw, verbose=False)
    ica.exclude.extend(eog_components)

    ecg_average = mne.preprocessing.create_ecg_epochs(raw).average() if 'ecg' in raw.get_channel_types() else None
    ecg_components, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
    ica.exclude.extend(ecg_components)

    report.add_ica(ica=ica, title=f'ICA cleaning ({task})', inst=raw, eog_evoked=eog_average, ecg_evoked=ecg_average, eog_scores=eog_scores, ecg_scores=ecg_scores, n_jobs=N_JOBS)
    
    ica.apply(raw, verbose=False)
    return raw

def identify_and_clean_epochs(raw, events, event_dict, tmin, tmax, report, task_name):
    """Creates temporary proxy epochs strictly to identify bad channels and epochs."""
    print_timestamp(f'Detecting bad channels & epochs via Proxy-Epochs ({task_name})')
    
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, event_id=event_dict, 
                        preload=True, baseline=None, verbose=False, reject_by_annotation=False)
    if task_name != 'Baseline':
        epochs.shift_time(tshift=0.0125, relative=True)

    report.add_epochs(epochs=epochs, psd=False, title=f'Raw Epochs: {task_name}') 

    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False, ecg=False)
    ransac = Ransac(picks=picks, n_jobs=-1, n_resample=30, min_corr=0.60, 
                    unbroken_time=0.30, min_channels=0.25, verbose=False, random_state=23)
    epochs_ransac = ransac.fit_transform(epochs)
    bad_chs = ransac.bad_chs_
    
    reject_criteria, flat_criteria = 200e-6, 1e-6
    data = epochs_ransac.get_data(picks='eeg')
    ptps = np.ptp(data, axis=2)
    max_ptp = np.max(ptps, axis=1)
    min_ptp = np.min(ptps, axis=1)
    
    bad_ptp_mask = (max_ptp > reject_criteria) | (min_ptp < flat_criteria)
    survivor_indices_ptp = np.where(~bad_ptp_mask)[0]
    epochs_ptp_clean = epochs_ransac[~bad_ptp_mask]

    ar = autoreject.AutoReject(n_interpolate=[1, 2], consensus=[0.4, 0.5, 0.6], 
                               thresh_method='bayesian_optimization', cv=5, 
                               random_state=23, n_jobs=-1, verbose=False)
    ar.fit(epochs_ptp_clean)
    _, reject_log = ar.transform(epochs_ptp_clean, return_log=True)

    final_keep_mask = np.zeros(len(epochs), dtype=bool)
    final_survivors = survivor_indices_ptp[~reject_log.bad_epochs]
    final_keep_mask[final_survivors] = True

    stats = {
        'Total_Epochs': len(epochs), 'PTP_Dropped': np.sum(bad_ptp_mask),
        'AR_Dropped': reject_log.bad_epochs.sum(), 'Total_Dropped': len(epochs) - len(final_survivors),
        'Bad_Channels': bad_chs
    }
    
    # We explicitly return epochs.events because MNE drops out-of-bounds events silently upon initialization.
    # We must save the exact array MNE used so Cell 2 aligns perfectly with the boolean mask.
    return bad_chs, final_keep_mask, stats, epochs.events


# ==============================================================================
# Main Preprocessing Pipeline
# ==============================================================================
def process_preprocessing(subject, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    report_file = out_folder / f'{subject}_quality_report.html'
    report = mne.Report(title=f"Data Quality: {subject}", verbose=False)
    channel_type_dict = pd.read_csv(CHANNEL_MAP_PATH).set_index('ch_name').to_dict()['ch_type']
    participant_stats = []
    
    raws = {}
    for task in ['JustFaces', 'StatementsFaces']:
        ext = guess_ext_and_reader(subject, task)
        if ext is None: return None
        in_file = SOURCE_DATA / subject / 'eeg' / f'{subject}_task-{task}_eeg.{ext}'
        raws[task] = preprocess_raw_and_ica(in_file, channel_type_dict, REFERENCE_METHOD, report, subject, task)

    events_jf = mne.find_events(raws['JustFaces'], stim_channel='Status', mask=0xFF, mask_type='and', verbose=False)
    valid_events_jf = [e for e in events_jf if e[2] in EVENTS_JUSTFACES.values()]
    first_event_time = valid_events_jf[0][0] / raws['JustFaces'].info['sfreq'] if valid_events_jf else 60.0
    
    max_baseline_duration = min(60.0, first_event_time)
    n_epochs = int(np.floor(max_baseline_duration)) 
    tmin_base = first_event_time - n_epochs 
    raws['Baseline'] = raws['JustFaces'].copy().crop(tmin=tmin_base, tmax=first_event_time)
    
    for task in ['Baseline', 'JustFaces', 'StatementsFaces']:
        print_timestamp(f'Cleaning and exporting {task}...')
        
        if task == 'Baseline':
            events = mne.make_fixed_length_events(raws[task], duration=1.0)
            event_dict = {'Baseline': 1}
            tmin, tmax = 0.0, 1.0
        else:
            events = mne.find_events(raws[task], stim_channel='Status', mask=0xFF, mask_type='and', verbose=False)
            event_dict = EVENTS_JUSTFACES if task == 'JustFaces' else EVENTS_STATEMENTSFACES
            tmin, tmax = (0.0, 1.2) if task == 'JustFaces' else (-1.2, 1.2)

        # 1. Identify bad channels and epochs
        bad_channels, keep_mask, stats, exact_events = identify_and_clean_epochs(
            raws[task], events, event_dict, tmin, tmax, report, task
        )
        
        # 2. Interpolate bad channels on continuous data
        raw_continuous = raws[task].copy()
        if len(bad_channels) > 0:
            raw_continuous.info['bads'] = bad_channels
            raw_continuous.interpolate_bads(reset_bads=True, verbose=False)
            
        # 3. Save the clean continuous raw, the event array, and the mask
        raw_continuous.save(out_folder / f'{subject}_task-{task}_clean-raw.fif', overwrite=True)
        np.save(out_folder / f'{subject}_task-{task}_events.npy', exact_events)
        np.save(out_folder / f'{subject}_task-{task}_keepmask.npy', keep_mask)

        # 4. Generate the Epoched data for SNR calculations and general ERP use
        epochs_clean = mne.Epochs(raw_continuous, exact_events, event_id=event_dict, tmin=tmin, tmax=tmax, 
                                  preload=True, baseline=None, verbose=False, reject_by_annotation=False)
        if task != 'Baseline': epochs_clean.shift_time(tshift=0.0125, relative=True)

        epochs_clean = epochs_clean[keep_mask] # Drop bad epochs
        epochs_clean.save(out_folder / f'{subject}_task-{task}_clean-epo.fif', overwrite=True)

        frontal_chs = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']

        row_stat = {
            'Subject': subject, 'Task': task, 'Total_Epochs': stats['Total_Epochs'], 'Clean_Epochs': keep_mask.sum(),
            '%_Rejected': round((stats['Total_Dropped'] / stats['Total_Epochs']) * 100, 2) if stats['Total_Epochs'] > 0 else 0,
            'Total_Bad_Chs': len(bad_channels), 'Bad_Chs_List': ", ".join(bad_channels),
            **{f'{ch}_Bad': (ch in bad_channels) for ch in frontal_chs},
        }
        participant_stats.append(row_stat)

    report.save(report_file, open_browser=False, overwrite=True)
    return participant_stats

if __name__ == '__main__':
    subject_list = sorted([d.name for d in SOURCE_DATA.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    all_rejection_stats = []

    for subject in subject_list:
        print('\n' + '#' * 60)
        print(f'Preprocessing {subject} | Batch: {BATCH_NAME}')
        print('#' * 60)
        out_folder = DERIVATIVES_DIR / subject / 'eeg'
        try:
            subj_rejections = process_preprocessing(subject, out_folder)
            if subj_rejections: all_rejection_stats.extend(subj_rejections)
        except Exception as e: print(f'Error processing {subject}: {e}')

    if all_rejection_stats:
        print_timestamp('Generating Global Quality Reports...')
        df_rejects = pd.DataFrame(all_rejection_stats)
        df_rejects.to_csv(DERIVATIVES_DIR / f'Data_Quality_Report_{BATCH_NAME}.csv', index=False)
        
        # Simplistic HTML export snippet for brevity
        df_rejects.to_html(DERIVATIVES_DIR / f'Data_Quality_Report_{BATCH_NAME}.html', index=False, na_rep='-')
        print(f"Saved Quality CSV & HTML")

# %% Cell 2: Frontal Alpha Asymmetry (FAA) & Frequency Domain Analysis (can be run separately, after raw data is preprocessed)
import os
import time
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# Configuration & Paths 
# ==============================================================================
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"
FAA_METHOD = 'diff'                  # Options: 'diff' (ln F4 - ln F3) OR 'norm' ((ln F4 - ln F3) / (ln F4 + ln F3))
ALPHA_BAND = (8.0, 12.0)             

PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids' / 'derivatives' / 'processed' / BATCH_NAME
N_JOBS = -1

# Event Dicts needed for re-epoching
EVENTS_JUSTFACES = { 'PVV/Agema': 201, 'PVV/Wilders': 202, 'GL/Klaver': 211, 'GL/Timmermans': 212, 'Foil/Foil': 222 }
EVENTS_STATEMENTSFACES = {
    'GL/Klaver/left/negative/Climate': 5, 'GL/Klaver/left/negative/EU': 6, 'GL/Klaver/left/negative/Gender equality': 7,
    'GL/Klaver/left/negative/Immigration': 8, 'GL/Klaver/left/negative/Integration': 9, 'GL/Klaver/left/negative/Vegetarianism': 10,
    'GL/Klaver/right/negative/Climate': 11, 'GL/Klaver/right/negative/EU': 12, 'GL/Klaver/right/negative/Gender equality': 13,
    'GL/Klaver/right/negative/Immigration': 14, 'GL/Klaver/right/negative/Integration': 15, 'GL/Klaver/right/negative/Vegetarianism': 16,
    'GL/Klaver/left/neutral/Climate': 17, 'GL/Klaver/left/neutral/EU': 18, 'GL/Klaver/left/neutral/Gender equality': 19,
    'GL/Klaver/left/neutral/Immigration': 20, 'GL/Klaver/left/neutral/Integration': 21, 'GL/Klaver/left/neutral/Vegetarianism': 22,
    'GL/Klaver/right/neutral/Climate': 23, 'GL/Klaver/right/neutral/EU': 24, 'GL/Klaver/right/neutral/Gender equality': 25,
    'GL/Klaver/right/neutral/Immigration': 26, 'GL/Klaver/right/neutral/Integration': 27, 'GL/Klaver/right/neutral/Vegetarianism': 28,
    'GL/Klaver/left/positive/Climate': 29, 'GL/Klaver/left/positive/EU': 30, 'GL/Klaver/left/positive/Gender equality': 31,
    'GL/Klaver/left/positive/Immigration': 32, 'GL/Klaver/left/positive/Integration': 33, 'GL/Klaver/left/positive/Vegetarianism': 34,
    'GL/Klaver/right/positive/Climate': 35, 'GL/Klaver/right/positive/EU': 36, 'GL/Klaver/right/positive/Gender equality': 37,
    'GL/Klaver/right/positive/Immigration': 38, 'GL/Klaver/right/positive/Integration': 39, 'GL/Klaver/right/positive/Vegetarianism': 40,
    'GL/Timmermans/left/negative/Climate': 41, 'GL/Timmermans/left/negative/EU': 42, 'GL/Timmermans/left/negative/Gender equality': 43,
    'GL/Timmermans/left/negative/Immigration': 44, 'GL/Timmermans/left/negative/Integration': 45, 'GL/Timmermans/left/negative/Vegetarianism': 46,
    'GL/Timmermans/right/negative/Climate': 47, 'GL/Timmermans/right/negative/EU': 48, 'GL/Timmermans/right/negative/Gender equality': 49,
    'GL/Timmermans/right/negative/Immigration': 50, 'GL/Timmermans/right/negative/Integration': 51, 'GL/Timmermans/right/negative/Vegetarianism': 52,
    'GL/Timmermans/left/neutral/Climate': 53, 'GL/Timmermans/left/neutral/EU': 54, 'GL/Timmermans/left/neutral/Gender equality': 55,
    'GL/Timmermans/left/neutral/Immigration': 56, 'GL/Timmermans/left/neutral/Integration': 57, 'GL/Timmermans/left/neutral/Vegetarianism': 58,
    'GL/Timmermans/right/neutral/Climate': 59, 'GL/Timmermans/right/neutral/EU': 60, 'GL/Timmermans/right/neutral/Gender equality': 61,
    'GL/Timmermans/right/neutral/Immigration': 62, 'GL/Timmermans/right/neutral/Integration': 63, 'GL/Timmermans/right/neutral/Vegetarianism': 64,
    'GL/Timmermans/left/positive/Climate': 65, 'GL/Timmermans/left/positive/EU': 66, 'GL/Timmermans/left/positive/Gender equality': 67,
    'GL/Timmermans/left/positive/Immigration': 68, 'GL/Timmermans/left/positive/Integration': 69, 'GL/Timmermans/left/positive/Vegetarianism': 70,
    'GL/Timmermans/right/positive/Climate': 71, 'GL/Timmermans/right/positive/EU': 72, 'GL/Timmermans/right/positive/Gender equality': 73,
    'GL/Timmermans/right/positive/Immigration': 74, 'GL/Timmermans/right/positive/Integration': 75, 'GL/Timmermans/right/positive/Vegetarianism': 76,
    'PVV/Agema/left/negative/Climate': 77, 'PVV/Agema/left/negative/EU': 78, 'PVV/Agema/left/negative/Gender equality': 79,
    'PVV/Agema/left/negative/Immigration': 80, 'PVV/Agema/left/negative/Integration': 81, 'PVV/Agema/left/negative/Vegetarianism': 82,
    'PVV/Agema/right/negative/Climate': 83, 'PVV/Agema/right/negative/EU': 84, 'PVV/Agema/right/negative/Gender equality': 85,
    'PVV/Agema/right/negative/Immigration': 86, 'PVV/Agema/right/negative/Integration': 87, 'PVV/Agema/right/negative/Vegetarianism': 88,
    'PVV/Agema/left/neutral/Climate': 89, 'PVV/Agema/left/neutral/EU': 90, 'PVV/Agema/left/neutral/Gender equality': 91,
    'PVV/Agema/left/neutral/Immigration': 92, 'PVV/Agema/left/neutral/Integration': 93, 'PVV/Agema/left/neutral/Vegetarianism': 94,
    'PVV/Agema/right/neutral/Climate': 95, 'PVV/Agema/right/neutral/EU': 96, 'PVV/Agema/right/neutral/Gender equality': 97,
    'PVV/Agema/right/neutral/Immigration': 98, 'PVV/Agema/right/neutral/Integration': 99, 'PVV/Agema/right/neutral/Vegetarianism': 100,
    'PVV/Agema/left/positive/Climate': 101, 'PVV/Agema/left/positive/EU': 102, 'PVV/Agema/left/positive/Gender equality': 103,
    'PVV/Agema/left/positive/Immigration': 104, 'PVV/Agema/left/positive/Integration': 105, 'PVV/Agema/left/positive/Vegetarianism': 106,
    'PVV/Agema/right/positive/Climate': 107, 'PVV/Agema/right/positive/EU': 108, 'PVV/Agema/right/positive/Gender equality': 109,
    'PVV/Agema/right/positive/Immigration': 110, 'PVV/Agema/right/positive/Integration': 111, 'PVV/Agema/right/positive/Vegetarianism': 112,
    'PVV/Wilders/left/negative/Climate': 113, 'PVV/Wilders/left/negative/EU': 114, 'PVV/Wilders/left/negative/Gender equality': 115,
    'PVV/Wilders/left/negative/Immigration': 116, 'PVV/Wilders/left/negative/Integration': 117, 'PVV/Wilders/left/negative/Vegetarianism': 118,
    'PVV/Wilders/right/negative/Climate': 119, 'PVV/Wilders/right/negative/EU': 120, 'PVV/Wilders/right/negative/Gender equality': 121,
    'PVV/Wilders/right/negative/Immigration': 122, 'PVV/Wilders/right/negative/Integration': 123, 'PVV/Wilders/right/negative/Vegetarianism': 124,
    'PVV/Wilders/left/neutral/Climate': 125, 'PVV/Wilders/left/neutral/EU': 126, 'PVV/Wilders/left/neutral/Gender equality': 127,
    'PVV/Wilders/left/neutral/Immigration': 128, 'PVV/Wilders/left/neutral/Integration': 129, 'PVV/Wilders/left/neutral/Vegetarianism': 130,
    'PVV/Wilders/right/neutral/Climate': 131, 'PVV/Wilders/right/neutral/EU': 132, 'PVV/Wilders/right/neutral/Gender equality': 133,
    'PVV/Wilders/right/neutral/Immigration': 134, 'PVV/Wilders/right/neutral/Integration': 135, 'PVV/Wilders/right/neutral/Vegetarianism': 136,
    'PVV/Wilders/left/positive/Climate': 137, 'PVV/Wilders/left/positive/EU': 138, 'PVV/Wilders/left/positive/Gender equality': 139,
    'PVV/Wilders/left/positive/Immigration': 140, 'PVV/Wilders/left/positive/Integration': 141, 'PVV/Wilders/left/positive/Vegetarianism': 142,
    'PVV/Wilders/right/positive/Climate': 143, 'PVV/Wilders/right/positive/EU': 144, 'PVV/Wilders/right/positive/Gender equality': 145,
    'PVV/Wilders/right/positive/Immigration': 146, 'PVV/Wilders/right/positive/Integration': 147, 'PVV/Wilders/right/positive/Vegetarianism': 148,
}


def print_timestamp(message):
    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] {message}")

# ==============================================================================
# FAA Calculation on Continuous Data
# ==============================================================================
def extract_faa_continuous(raw, alpha_band, method):
    """Filters continuous data to alpha bounds, applies Hilbert, and extracts FAA & Power."""
    alpha_raw = raw.copy().filter(l_freq=alpha_band[0], h_freq=alpha_band[1], n_jobs=N_JOBS, verbose=False)
    alpha_raw.apply_hilbert(envelope=True, verbose=False) 
    
    f3_amp = (alpha_raw.get_data(picks='F3').squeeze() * 1e6) #shifted from volts to microvolts
    f4_amp = (alpha_raw.get_data(picks='F4').squeeze() * 1e6) #shifted from volts to microvolts
    
    f3_pow = f3_amp ** 2
    f4_pow = f4_amp ** 2
    
    if method == 'diff': faa_series = np.log(f4_pow) - np.log(f3_pow)
    elif method == 'norm': faa_series = (np.log(f4_pow) - np.log(f3_pow)) / (np.log(f4_pow) + np.log(f3_pow))
    else: raise ValueError("FAA_METHOD must be either 'diff' or 'norm'")
        
    info_faa = mne.create_info(['FAA'], sfreq=raw.info['sfreq'], ch_types=['misc'])
    info_pow = mne.create_info(['F3_Pow', 'F4_Pow'], sfreq=raw.info['sfreq'], ch_types=['misc', 'misc'])
    
    # Pack back into continuous Raw arrays ensuring first_samp matches so time aligns perfectly
    raw_faa = mne.io.RawArray(faa_series.reshape(1, -1), info_faa, first_samp=raw.first_samp, verbose=False)
    raw_pow = mne.io.RawArray(np.vstack([f4_pow, f3_pow]), info_pow, first_samp=raw.first_samp, verbose=False)
    
    return raw_faa, raw_pow

# ==============================================================================
# Main Frequency Pipeline
# ==============================================================================
def process_frequency_analyses(subject, out_folder):
    participant_metrics = []
    
    for task in ['Baseline', 'JustFaces', 'StatementsFaces']:
        raw_path = out_folder / f'{subject}_task-{task}_clean-raw.fif'
        events_path = out_folder / f'{subject}_task-{task}_events.npy'
        mask_path = out_folder / f'{subject}_task-{task}_keepmask.npy'
        
        if not raw_path.exists():
            print(f"Skipping {task} - preprocessed files not found for {subject}")
            continue
            
        print_timestamp(f'Calculating continuous FAA for {task}...')
        
        # 1. Load the continuous clean data, events array, and boolean rejection mask
        raw_clean = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        exact_events = np.load(events_path)
        keep_mask = np.load(mask_path)
        
        # 2. Extract FAA and Power timeseries from the CONTINUOUS data
        raw_faa, raw_pow = extract_faa_continuous(raw_clean, ALPHA_BAND, FAA_METHOD)
        
        # 3. Setup epoch parameters
        if task == 'Baseline':
            event_dict = {'Baseline': 1}
            tmin, tmax = 0.0, 1.0
        else:
            event_dict = EVENTS_JUSTFACES if task == 'JustFaces' else EVENTS_STATEMENTSFACES
            tmin, tmax = (0.0, 1.2) if task == 'JustFaces' else (-1.2, 1.2)

        # 4. Epoch the transformed data using the EXACT events array from Cell 1
        epochs_faa = mne.Epochs(raw_faa, exact_events, event_id=event_dict, tmin=tmin, tmax=tmax, 
                                baseline=None, preload=True, verbose=False)
        epochs_pow = mne.Epochs(raw_pow, exact_events, event_id=event_dict, tmin=tmin, tmax=tmax, 
                                baseline=None, preload=True, verbose=False)
        
        # Align time axis if shifted in Cell 1
        if task != 'Baseline':
            epochs_faa.shift_time(tshift=0.0125, relative=True)
            epochs_pow.shift_time(tshift=0.0125, relative=True)

        # 5. CRITICAL STEP: Drop the bad epochs using the saved mask
        epochs_faa_clean = epochs_faa[keep_mask]
        epochs_pow_clean = epochs_pow[keep_mask]
        
        # Save final clean FAA epochs
        epochs_faa_clean.save(out_folder / f'{subject}_task-{task}_faa_{FAA_METHOD}-epo.fif', overwrite=True)
        
        # Aggregate stats
        avg_f3 = np.mean(epochs_pow_clean.get_data(picks='F3_Pow'))
        avg_f4 = np.mean(epochs_pow_clean.get_data(picks='F4_Pow'))
        avg_faa = np.mean(epochs_faa_clean.get_data(picks='FAA'))
        
        participant_metrics.append({
            'Subject': subject, 'Task': task, 
            'Mean F3 Power': round(avg_f3, 6), 'Mean F4 Power': round(avg_f4, 6), 
            'Mean FAA Index': round(avg_faa, 6)
        })

        # Visualization
        heatmap_data = epochs_faa_clean.get_data(picks='FAA').squeeze()
        if heatmap_data.ndim == 1: heatmap_data = heatmap_data.reshape(1, -1) 
        
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
        vbound = 3.0 if FAA_METHOD == 'diff' else 0.1
        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                            extent=[epochs_faa_clean.tmin, epochs_faa_clean.tmax, heatmap_data.shape[0], 0], vmin=-vbound, vmax=vbound)
        ax_heat.axvline(0, color='black', linestyle='--', linewidth=2)
        ax_heat.set_title(f'Clean Epoch FAA Heatmap ({subject} - {task})')
        fig_heat.colorbar(im, ax=ax_heat, label='FAA Index')
        plt.savefig(out_folder / f'{subject}_task-{task}_faa_{FAA_METHOD}-heatmap.png')
        plt.close(fig_heat)

    return participant_metrics

if __name__ == '__main__':
    subject_dirs = sorted([d for d in DERIVATIVES_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    all_faa_stats = []

    for subj_dir in subject_dirs:
        subject = subj_dir.name
        print('\n' + '#' * 60)
        print(f'Running Frequency Domain & FAA: {subject}')
        print('#' * 60)
        out_folder = subj_dir / 'eeg'
        try:
            subj_metrics = process_frequency_analyses(subject, out_folder)
            if subj_metrics: all_faa_stats.extend(subj_metrics)
        except Exception as e: print(f'Error processing {subject}: {e}')

    if all_faa_stats:
        print_timestamp('Generating Global FAA Averages Reports...')
        df_faa = pd.DataFrame(all_faa_stats)
        df_faa.to_csv(DERIVATIVES_DIR / f'FAA_Averages_Report_{BATCH_NAME}_{FAA_METHOD}.csv', index=False)
        df_faa.to_html(DERIVATIVES_DIR / f'FAA_Averages_Report_{BATCH_NAME}_{FAA_METHOD}.html', index=False, na_rep='-')
        print(f"Saved FAA Averages CSV & HTML")
# %%