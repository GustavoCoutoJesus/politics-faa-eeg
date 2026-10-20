# %% #######################################################################################################################
# The goal of this script is to preprocess the raw data and prepare it for data analysis
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
import autoreject
from autoreject import Ransac

# ==============================================================================
# Configuration & Paths (USER SELECTABLE PARAMETERS)
# ==============================================================================
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
REFERENCE_METHOD = 'virtual_Cz'      
VIRTUAL_REF_CHANNELS = ['Cz', 'FCz', 'CPz', 'C1', 'C2'] 

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
SOURCE_DATA = PROJECT_FOLDER / 'bids_emotion' / 'sourcedata'
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'processed' / BATCH_NAME
CHANNEL_MAP_PATH = PROJECT_FOLDER / 'MConfig_ChTypes.csv'

# Define start and end subjects for chunked processing (Set both to the same ID to process just 1 subject)
# Leave as None to process the entire folder
START_SUBJECT = "None"  # e.g., "sub-002_MU"
END_SUBJECT = "None"    # e.g., "sub-005_MU_pilot"

N_JOBS = -1

# ==============================================================================
# Event Dictionaries 
# ==============================================================================
EVENTS_EMOTION = {
    'white/pol/inparty/angry/1': 61551,
    'white/pol/inparty/happy/1': 61552,
    'white/pol/inparty/neutral/1': 61553,
    'white/pol/inparty/angry/2': 61561,
    'white/pol/inparty/happy/2': 61562,
    'white/pol/inparty/neutral/2': 61563,
    'white/pol/outparty/angry/1': 61571,
    'white/pol/outparty/happy/1': 61572,
    'white/pol/outparty/neutral/1': 61573,
    'white/pol/outparty/angry/2': 61581,
    'white/pol/outparty/happy/2': 61582,
    'white/pol/outparty/neutral/2': 61583,

    'white/nonpol/angry/F01': 61651,
    'white/nonpol/happy/F01': 61652,
    'white/nonpol/neutral/F01': 61653,
    'white/nonpol/angry/F03': 61661,
    'white/nonpol/happy/F03': 61662,
    'white/nonpol/neutral/F03': 61663,
    'white/nonpol/angry/M02': 61671,
    'white/nonpol/happy/M02': 61672,
    'white/nonpol/neutral/M02': 61673,
    'white/nonpol/angry/M11': 61681,
    'white/nonpol/happy/M11': 61682,
    'white/nonpol/neutral/M11': 61683,
}

# ==============================================================================
# Helper Functions
# ==============================================================================
def print_timestamp(message):
    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] {message}")

def update_global_csv(new_data, csv_path):
    """Appends new stats to the global CSV, overwriting subjects if they already exist."""
    new_df = pd.DataFrame(new_data)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        # Remove old entries for the subjects currently being processed
        old_df = old_df[~old_df['Subject'].isin(new_df['Subject'])]
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Re-sort to maintain clean order
    combined_df.sort_values(by=['Subject'], inplace=True)
    combined_df.to_csv(csv_path, index=False)
    # Generate companion HTML
    combined_df.to_html(csv_path.with_suffix('.html'), index=False, na_rep='-')

def preprocess_raw_and_ica(in_file, channel_type_dict, ref_method, report, subject):
    """Loads, downsamples, filters, references, runs ICA, and adds to HTML report."""
    print_timestamp(f'Reading raw BDF data for {subject}')
    raw = mne.io.read_raw_bdf(in_file, preload=True, verbose=False)
    
    channels_to_keep = list(channel_type_dict.keys())
    if 'Status' in raw.ch_names and 'Status' not in channels_to_keep:
        channels_to_keep.append('Status')
        
    raw.drop_channels([ch for ch in raw.ch_names if ch not in channels_to_keep])
    if 'Status' in raw.ch_names: raw.set_channel_types({'Status': 'stim'}, verbose=False)
    raw.set_channel_types(channel_type_dict, verbose=False)
    
    # Referencing
    print_timestamp('Setting referencing')
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
    
    print_timestamp('Filtering 1.0-40Hz & Downsampling to 250Hz')
    raw.filter(l_freq=1.0, h_freq=40.0, n_jobs=N_JOBS, verbose=False)
    raw.resample(250, n_jobs=N_JOBS, verbose=False)
    
    print_timestamp('Running ICA')
    ica = mne.preprocessing.ICA(n_components=25, method='fastica', random_state=23)
    ica.fit(inst=raw, decim=3, verbose=False)

    eog_average = mne.preprocessing.create_eog_epochs(raw).average() if 'eog' in raw.get_channel_types() else None
    eog_components, eog_scores = ica.find_bads_eog(raw, verbose=False)
    ica.exclude.extend(eog_components)

    ecg_average, ecg_scores = None, None
    if 'ecg' in raw.get_channel_types():
        ecg_average = mne.preprocessing.create_ecg_epochs(raw).average()
        ecg_components, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
        ica.exclude.extend(ecg_components)

    report.add_ica(ica=ica, title='ICA cleaning (Emotion)', inst=raw, eog_evoked=eog_average, ecg_evoked=ecg_average, eog_scores=eog_scores, ecg_scores=ecg_scores, n_jobs=N_JOBS)
    
    ica.apply(raw, verbose=False)
    return raw

def identify_and_clean_epochs(raw, events, event_dict, tmin, tmax, report):
    """Creates temporary proxy epochs strictly to identify bad channels and epochs."""
    print_timestamp(f'Detecting bad channels & epochs via Proxy-Epochs (-2.0 to 6.0s)')
    
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, event_id=event_dict, 
                        preload=True, baseline=None, verbose=False, reject_by_annotation=False)

    report.add_epochs(epochs=epochs, psd=False, title='Raw Epochs: Emotion')

    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False, ecg=False)
    
    print_timestamp('Running RANSAC...')
    ransac = Ransac(picks=picks, n_jobs=N_JOBS, n_resample=30, min_corr=0.60, 
                    unbroken_time=0.30, min_channels=0.25, verbose=False, random_state=23)
    epochs_ransac = ransac.fit_transform(epochs)
    bad_chs = ransac.bad_chs_
    
    # Peak-to-Peak dropping
    reject_criteria, flat_criteria = 200e-6, 1e-6
    data = epochs_ransac.get_data(picks='eeg')
    ptps = np.ptp(data, axis=2)
    max_ptp = np.max(ptps, axis=1)
    min_ptp = np.min(ptps, axis=1)
    
    bad_ptp_mask = (max_ptp > reject_criteria) | (min_ptp < flat_criteria)
    survivor_indices_ptp = np.where(~bad_ptp_mask)[0]
    epochs_ptp_clean = epochs_ransac[~bad_ptp_mask]

    print_timestamp('Running Autoreject...')
    ar = autoreject.AutoReject(n_interpolate=[1, 2], consensus=[0.4, 0.5, 0.6], 
                               thresh_method='bayesian_optimization', cv=5, 
                               random_state=23, n_jobs=-1, verbose=False)
    ar.fit(epochs_ptp_clean)
    _, reject_log = ar.transform(epochs_ptp_clean, return_log=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    reject_log.plot(orientation='vertical', ax=ax, show_names='auto', show=False)
    report.add_figure(fig=fig, title='Rejection Log', caption='Channels and epochs rejected by autoreject', image_format='PNG')
    plt.close(fig)

    final_keep_mask = np.zeros(len(epochs), dtype=bool)
    final_survivors = survivor_indices_ptp[~reject_log.bad_epochs]
    final_keep_mask[final_survivors] = True

    stats = {
        'Total_Epochs': len(epochs), 'PTP_Dropped': np.sum(bad_ptp_mask),
        'AR_Dropped': reject_log.bad_epochs.sum(), 'Total_Dropped': len(epochs) - len(final_survivors),
        'Bad_Channels': bad_chs
    }
    
    return bad_chs, final_keep_mask, stats, epochs.events


# ==============================================================================
# Main Preprocessing Pipeline
# ==============================================================================
def process_preprocessing(subject, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    report_file = out_folder / f'{subject}_task-emotion_quality_report.html'
    report = mne.Report(title=f"Data Quality: {subject}", verbose=False)
    
    channel_type_dict = pd.read_csv(CHANNEL_MAP_PATH).set_index('ch_name').to_dict()['ch_type']
    
    in_file = SOURCE_DATA / subject / 'eeg' / f'{subject}_task-faces_eeg.bdf'
    if not in_file.exists():
        print_timestamp(f"File not found: {in_file}")
        return None

    raw = preprocess_raw_and_ica(in_file, channel_type_dict, REFERENCE_METHOD, report, subject)

    print_timestamp('Identifying events')
    events = mne.find_events(raw, min_duration=0.01, output='onset', verbose=False)
    report.add_events(events=events, title='Events', sfreq=raw.info["sfreq"])

    tmin, tmax = -2.0, 6.0 

    # 1. Identify bad channels and epochs
    bad_channels, keep_mask, stats, exact_events = identify_and_clean_epochs(
        raw, events, EVENTS_EMOTION, tmin, tmax, report
    )
    
    # 2. Interpolate bad channels on continuous data
    raw_continuous = raw.copy()
    if len(bad_channels) > 0:
        raw_continuous.info['bads'] = bad_channels
        raw_continuous.interpolate_bads(reset_bads=True, verbose=False)
        
    # 3. Save the clean continuous raw, the event array, and the mask
    raw_continuous.save(out_folder / f'{subject}_task-emotion_clean-raw.fif', overwrite=True)
    np.save(out_folder / f'{subject}_task-emotion_events.npy', exact_events)
    np.save(out_folder / f'{subject}_task-emotion_keepmask.npy', keep_mask)

    # 4. Generate the Epoched data
    print_timestamp('Saving final epoched data')
    epochs_clean = mne.Epochs(raw_continuous, exact_events, event_id=EVENTS_EMOTION, tmin=tmin, tmax=tmax, 
                              preload=True, baseline=None, verbose=False, reject_by_annotation=False)

    epochs_clean = epochs_clean[keep_mask] # Drop bad epochs
    epochs_clean.save(out_folder / f'{subject}_task-emotion_clean-epo.fif', overwrite=True)

    frontal_chs = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
    row_stat = {
        'Subject': subject, 'Task': 'Emotion_Faces', 'Total_Epochs': stats['Total_Epochs'], 'Clean_Epochs': keep_mask.sum(),
        '%_Rejected': round((stats['Total_Dropped'] / stats['Total_Epochs']) * 100, 2) if stats['Total_Epochs'] > 0 else 0,
        'Total_Bad_Chs': len(bad_channels), 'Bad_Chs_List': ", ".join(bad_channels),
        **{f'{ch}_Bad': (ch in bad_channels) for ch in frontal_chs},
    }

    report.save(report_file, open_browser=False, overwrite=True)
    return row_stat

if __name__ == '__main__':
    # Fetch and naturally sort subject list (ensures sub-XXX goes before sub-XXX_pilot)
    subject_list = sorted([d.name for d in SOURCE_DATA.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    # Process Start and End bounds
    start_idx = subject_list.index(START_SUBJECT) if START_SUBJECT in subject_list else 0
    end_idx = subject_list.index(END_SUBJECT) if END_SUBJECT in subject_list else len(subject_list) - 1
    
    selected_subjects = subject_list[start_idx:end_idx+1]
    all_rejection_stats = []

    for subject in selected_subjects:
        out_folder = DERIVATIVES_DIR / subject / 'eeg'
        
        # ---> NEW: Check if this participant has already been fully processed
        if (out_folder / f'{subject}_task-emotion_clean-epo.fif').exists():
            print(f"\nSkipping {subject}: Already preprocessed.")
            continue
        # <---
        
        print('\n' + '#' * 60)
        print(f'Preprocessing {subject} | Batch: {BATCH_NAME}')
        print('#' * 60)
        
        try:
            subj_rejections = process_preprocessing(subject, out_folder)
            if subj_rejections: all_rejection_stats.append(subj_rejections)
        except Exception as e: 
            print(f'Error processing {subject}: {e}')

    if all_rejection_stats:
        print_timestamp('Updating Global Quality Reports...')
        global_csv_path = DERIVATIVES_DIR / f'Data_Quality_Report_{BATCH_NAME}.csv'
        update_global_csv(all_rejection_stats, global_csv_path)
        print(f"Saved/Updated Quality CSV & HTML")

# %% Cell 2: Frontal Alpha Asymmetry (FAA) & Frequency Domain Analysis
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
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
FAA_METHOD = 'diff'                  # Options: 'diff' (ln F4 - ln F3) OR 'norm' ((ln F4 - ln F3) / (ln F4 + ln F3))
ALPHA_BAND = (8.0, 12.0)             

PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'processed' / BATCH_NAME
N_JOBS = -1

# Start and End subjects for FAA generation
START_SUBJECT = "None"  # e.g., "sub-002_MU"
END_SUBJECT = "None"    # e.g., "sub-005_MU_pilot"

EVENTS_EMOTION = {
    'white/pol/inparty/angry/1': 61551, 'white/pol/inparty/happy/1': 61552, 'white/pol/inparty/neutral/1': 61553,
    'white/pol/inparty/angry/2': 61561, 'white/pol/inparty/happy/2': 61562, 'white/pol/inparty/neutral/2': 61563,
    'white/pol/outparty/angry/1': 61571, 'white/pol/outparty/happy/1': 61572, 'white/pol/outparty/neutral/1': 61573,
    'white/pol/outparty/angry/2': 61581, 'white/pol/outparty/happy/2': 61582, 'white/pol/outparty/neutral/2': 61583,
    'white/nonpol/angry/F01': 61651, 'white/nonpol/happy/F01': 61652, 'white/nonpol/neutral/F01': 61653,
    'white/nonpol/angry/F03': 61661, 'white/nonpol/happy/F03': 61662, 'white/nonpol/neutral/F03': 61663,
    'white/nonpol/angry/M02': 61671, 'white/nonpol/happy/M02': 61672, 'white/nonpol/neutral/M02': 61673,
    'white/nonpol/angry/M11': 61681, 'white/nonpol/happy/M11': 61682, 'white/nonpol/neutral/M11': 61683,
}

def print_timestamp(message):
    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] {message}")

def update_global_csv(new_data, csv_path):
    """Appends new stats to the global CSV, overwriting subjects if they already exist."""
    new_df = pd.DataFrame(new_data)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        old_df = old_df[~old_df['Subject'].isin(new_df['Subject'])]
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    combined_df.sort_values(by=['Subject'], inplace=True)
    combined_df.to_csv(csv_path, index=False)
    combined_df.to_html(csv_path.with_suffix('.html'), index=False, na_rep='-')

# ==============================================================================
# FAA Calculation on Continuous Data
# ==============================================================================
def extract_faa_continuous(raw, alpha_band, method):
    """Filters continuous data to alpha bounds, applies Hilbert, and extracts FAA & Power."""
    alpha_raw = raw.copy().filter(l_freq=alpha_band[0], h_freq=alpha_band[1], n_jobs=N_JOBS, verbose=False)
    alpha_raw.apply_hilbert(envelope=True, verbose=False) 
    
    f3_amp = (alpha_raw.get_data(picks='F3').squeeze() * 1e6) # shifted from volts to microvolts
    f4_amp = (alpha_raw.get_data(picks='F4').squeeze() * 1e6) 
    
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
    raw_path = out_folder / f'{subject}_task-emotion_clean-raw.fif'
    events_path = out_folder / f'{subject}_task-emotion_events.npy'
    mask_path = out_folder / f'{subject}_task-emotion_keepmask.npy'
    
    if not raw_path.exists():
        print(f"Skipping - preprocessed files not found for {subject}")
        return None
        
    print_timestamp(f'Calculating continuous FAA...')
    
    # 1. Load the continuous clean data, events array, and boolean rejection mask
    raw_clean = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    exact_events = np.load(events_path)
    keep_mask = np.load(mask_path)
    
    # 2. Extract FAA and Power timeseries from the CONTINUOUS data
    raw_faa, raw_pow = extract_faa_continuous(raw_clean, ALPHA_BAND, FAA_METHOD)
    
    # 3. Setup epoch parameters
    tmin, tmax = -2.0, 6.0

    # 4. Epoch the transformed data
    epochs_faa = mne.Epochs(raw_faa, exact_events, event_id=EVENTS_EMOTION, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)
    epochs_pow = mne.Epochs(raw_pow, exact_events, event_id=EVENTS_EMOTION, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)

    # 5. Drop the bad epochs using the saved mask
    epochs_faa_clean = epochs_faa[keep_mask]
    epochs_pow_clean = epochs_pow[keep_mask]
    
    # Save final clean FAA epochs
    epochs_faa_clean.save(out_folder / f'{subject}_task-emotion_faa_{FAA_METHOD}-epo.fif', overwrite=True)
    
    # Aggregate stats
    avg_f3 = np.mean(epochs_pow_clean.get_data(picks='F3_Pow'))
    avg_f4 = np.mean(epochs_pow_clean.get_data(picks='F4_Pow'))
    avg_faa = np.mean(epochs_faa_clean.get_data(picks='FAA'))
    
    metrics = {
        'Subject': subject, 'Task': 'Emotion_Faces', 
        'Mean F3 Power': round(avg_f3, 6), 'Mean F4 Power': round(avg_f4, 6), 
        'Mean FAA Index': round(avg_faa, 6)
    }

    # Visualization
    heatmap_data = epochs_faa_clean.get_data(picks='FAA').squeeze()
    if heatmap_data.ndim == 1: heatmap_data = heatmap_data.reshape(1, -1) 
    
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
    vbound = 3.0 if FAA_METHOD == 'diff' else 0.1
    im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                        extent=[epochs_faa_clean.tmin, epochs_faa_clean.tmax, heatmap_data.shape[0], 0], vmin=-vbound, vmax=vbound)
    ax_heat.axvline(0, color='black', linestyle='--', linewidth=2) # Neutral Phase
    ax_heat.axvline(2, color='gray', linestyle='--', linewidth=1.5) # Morph Phase
    ax_heat.axvline(4, color='gray', linestyle='--', linewidth=1.5) # Emotion Phase
    ax_heat.set_title(f'Clean Epoch FAA Heatmap ({subject} - Emotion)')
    fig_heat.colorbar(im, ax=ax_heat, label='FAA Index')
    plt.savefig(out_folder / f'{subject}_task-emotion_faa_{FAA_METHOD}-heatmap.png')
    plt.close(fig_heat)

    return metrics

if __name__ == '__main__':
    subject_dirs = sorted([d.name for d in DERIVATIVES_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    # Process Start and End bounds
    start_idx = subject_dirs.index(START_SUBJECT) if START_SUBJECT in subject_dirs else 0
    end_idx = subject_dirs.index(END_SUBJECT) if END_SUBJECT in subject_dirs else len(subject_dirs) - 1
    
    selected_subjects = subject_dirs[start_idx:end_idx+1]
    all_faa_stats = []

    for subject in selected_subjects:
        out_folder = DERIVATIVES_DIR / subject / 'eeg'
        
        # ---> NEW: Check if this participant has already been fully processed for FAA
        if (out_folder / f'{subject}_task-emotion_faa_{FAA_METHOD}-epo.fif').exists():
            print(f"\nSkipping {subject}: FAA already processed.")
            continue
        # <---

        print('\n' + '#' * 60)
        print(f'Running Frequency Domain & FAA: {subject}')
        print('#' * 60)
        
        try:
            subj_metrics = process_frequency_analyses(subject, out_folder)
            if subj_metrics: all_faa_stats.append(subj_metrics)
        except Exception as e: 
            print(f'Error processing {subject}: {e}')

    if all_faa_stats:
        print_timestamp('Updating Global FAA Averages Reports...')
        global_faa_path = DERIVATIVES_DIR / f'FAA_Averages_Report_{BATCH_NAME}_{FAA_METHOD}.csv'
        update_global_csv(all_faa_stats, global_faa_path)
        print(f"Saved/Updated FAA Averages CSV & HTML")
# %%
