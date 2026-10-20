# %% #######################################################################################################################
# The goal of this script is to preprocess the raw data and prepare it for data analysis
# This version has relaxed its RANSAC, autorej, and has manually excluded channels to be included in ICA
# We also apply IQR-based exclusions for extreme alpha power in F3 or F4
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

# ==============================================================================
# Configuration & Paths (USER SELECTABLE PARAMETERS)
# ==============================================================================
BATCH_NAME = "Batch_02_Emotion_VirtualCz_8-12Hz_relaxed"
REFERENCE_METHOD = 'virtual_Cz'      
VIRTUAL_REF_CHANNELS = ['Cz', 'FCz', 'CPz', 'C1', 'C2'] 

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
SOURCE_DATA = PROJECT_FOLDER / 'bids_emotion' / 'sourcedata'
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'processed' / BATCH_NAME
CHANNEL_MAP_PATH = PROJECT_FOLDER / 'MConfig_ChTypes.csv'

# NEW: Path to the visual inspection tracker
TRACKER_CSV_PATH = PROJECT_FOLDER / 'bids_emotion' / 'code' / 'raw_inspection.csv'

# Epoch Boundaries (Defined globally for findbreaksICA logic)
EPOCH_TMIN = -2.0
EPOCH_TMAX = 6.0

# Define start and end subjects for chunked processing
START_SUBJECT = "None"  
END_SUBJECT = "None"    

N_JOBS = -1

# ==============================================================================
# Event Dictionaries 
# ==============================================================================
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
        old_df = old_df[~((old_df['Subject'].isin(new_df['Subject'])) & (old_df['Variant'] == new_df['Variant'].iloc[0]))]
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    combined_df.sort_values(by=['Subject', 'Variant'], inplace=True)
    combined_df.to_csv(csv_path, index=False)
    combined_df.to_html(csv_path.with_suffix('.html'), index=False, na_rep='-')

def preprocess_raw_and_ica(in_file, channel_type_dict, ref_method, manual_bads, report, subject):
    """Loads, applies manual bads, runs Break Detection, and computes ICA."""
    print_timestamp(f'Reading raw BDF data for {subject}')
    raw = mne.io.read_raw_bdf(in_file, preload=True, verbose=False)
    
    channels_to_keep = list(channel_type_dict.keys())
    if 'Status' in raw.ch_names and 'Status' not in channels_to_keep:
        channels_to_keep.append('Status')
        
    raw.drop_channels([ch for ch in raw.ch_names if ch not in channels_to_keep])
    if 'Status' in raw.ch_names: raw.set_channel_types({'Status': 'stim'}, verbose=False)
    raw.set_channel_types(channel_type_dict, verbose=False)
    
    # --------------------------------------------------------------------------
    # Apply Manual Bads from Tracker
    # --------------------------------------------------------------------------
    valid_bads = [ch for ch in manual_bads if ch in raw.ch_names]
    raw.info['bads'] = valid_bads
    print_timestamp(f'Applied Bad Channels from Tracker: {valid_bads}')

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
    
    # --------------------------------------------------------------------------
    # breakICA implementation
    # this allows us to avoid certain noisier sections of the data before extracting components
    # --------------------------------------------------------------------------
    print_timestamp('Detecting breaks from event gaps')
    existing = raw.annotations
    keep = [i for i, desc in enumerate(existing.description) if desc != 'BAD_break']
    if len(keep) != len(existing):
        raw.set_annotations(existing[keep])
        
    sfreq = raw.info['sfreq']
    stim_events = mne.find_events(raw, min_duration=0.01, output='onset', verbose=False)
    event_times = (stim_events[:, 0] - raw.first_samp) / sfreq
    gaps = np.diff(event_times)
    break_mask = gaps > 5.0
    
    if break_mask.any():
        break_starts = event_times[:-1][break_mask] + EPOCH_TMAX
        break_ends = event_times[1:][break_mask] - 1.0
        break_durations = break_ends - break_starts
        valid = break_durations > 0
        
        if valid.any():
            orig_time = raw.annotations.orig_time
            onset_offset = (raw.first_samp / sfreq) if orig_time is not None else 0.0
            break_annotations = mne.Annotations(
                onset=break_starts[valid] + onset_offset,
                duration=break_durations[valid],
                description=['BAD_break'] * int(valid.sum()),
                orig_time=orig_time,
            )
            raw.set_annotations(raw.annotations + break_annotations)
            print_timestamp(f'Marked {int(valid.sum())} break(s) as BAD_break (total {float(break_durations[valid].sum()):.1f}s)')

    # --------------------------------------------------------------------------
    # ICA
    # --------------------------------------------------------------------------
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

    report.add_ica(ica=ica, title='ICA cleaning', inst=raw, eog_evoked=eog_average, ecg_evoked=ecg_average, eog_scores=eog_scores, ecg_scores=ecg_scores, n_jobs=N_JOBS)
    
    ica.apply(raw, verbose=False)
    return raw

def identify_and_clean_epochs_simple(raw, events, event_dict, report):
    """Epochs the data and drops trials based on absolute amplitude and 5SD fluctuation."""
    print_timestamp(f'Epoching & Rejecting Extreme Values/Fluctuations (-2.0 to 6.0s)')
    
    epochs = mne.Epochs(raw, events, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, event_id=event_dict, 
                        preload=True, baseline=None, verbose=False, reject_by_annotation=False)

    report.add_epochs(epochs=epochs, psd=False, title='Raw Epochs: Emotion')

    data = epochs.get_data(picks='eeg') # Shape: (epochs, channels, times)
    
    # 1. Extreme Peak Rejection (1000 uV)
    extreme_mask = np.any(np.abs(data) > 1000e-6, axis=(1, 2))
    
    # 2. 5 SD Fluctuation Rejection
    # Calculate SD for each epoch across time, for each channel
    epoch_sds = np.std(data, axis=2) # Shape: (epochs, channels)
    
    # Find the normative mean and std of these fluctuations across all epochs for each channel
    mean_sd_per_ch = np.mean(epoch_sds, axis=0)
    std_sd_per_ch = np.std(epoch_sds, axis=0)
    
    # If an epoch's SD is > 5 SDs above the channel's mean SD, flag it
    fluctuation_mask_ch = epoch_sds > (mean_sd_per_ch + 5 * std_sd_per_ch)
    
    # Reject epoch if ANY channel breaks the fluctuation rule
    fluctuation_mask = np.any(fluctuation_mask_ch, axis=1)
    
    bad_epochs_mask = extreme_mask | fluctuation_mask
    keep_mask = ~bad_epochs_mask
    
    stats = {
        'Total_Epochs': len(epochs),
        'Extreme_Dropped': np.sum(extreme_mask),
        'Fluctuation_Dropped': np.sum(fluctuation_mask),
        'Total_Dropped': np.sum(bad_epochs_mask)
    }
    
    return keep_mask, stats, epochs.events


# ==============================================================================
# Main Preprocessing Pipeline Execution
# ==============================================================================
def process_pipeline_variant(subject, base_out_folder, manual_bads, variant_name):
    """Executes the pipeline for a specific variant (Standard vs F3/F4 Omitted)"""
    if variant_name == 'Standard':
        out_folder = base_out_folder
    else:
        out_folder = base_out_folder / 'F3_F4_bad_channel_omission'
        
    os.makedirs(out_folder, exist_ok=True)
    report_file = out_folder / f'{subject}_task-emotion_{variant_name}_quality_report.html'
    report = mne.Report(title=f"Data Quality: {subject} ({variant_name})", verbose=False)
    
    channel_type_dict = pd.read_csv(CHANNEL_MAP_PATH).set_index('ch_name').to_dict()['ch_type']
    in_file = SOURCE_DATA / subject / 'eeg' / f'{subject}_task-faces_eeg.bdf'
    
    if not in_file.exists():
        print_timestamp(f"File not found: {in_file}")
        return None

    # Skip if already processed
    if (out_folder / f'{subject}_task-emotion_clean-epo.fif').exists():
        print_timestamp(f"Skipping {subject} [{variant_name}]: Already preprocessed.")
        return None

    # Step 1: Prep, Bads, ICA
    raw = preprocess_raw_and_ica(in_file, channel_type_dict, REFERENCE_METHOD, manual_bads, report, subject)

    print_timestamp('Identifying events')
    events = mne.find_events(raw, min_duration=0.01, output='onset', verbose=False)
    report.add_events(events=events, title='Events', sfreq=raw.info["sfreq"])

    # Step 2: Epoch Rejection
    keep_mask, stats, exact_events = identify_and_clean_epochs_simple(raw, events, EVENTS_EMOTION, report)
    
    # Step 3: Interpolate bad channels on continuous data
    raw_continuous = raw.copy()
    if len(manual_bads) > 0:
        raw_continuous.interpolate_bads(reset_bads=True, verbose=False)
        
    # Step 4: Save continuous raw, events, and mask
    raw_continuous.save(out_folder / f'{subject}_task-emotion_clean-raw.fif', overwrite=True)
    np.save(out_folder / f'{subject}_task-emotion_events.npy', exact_events)
    np.save(out_folder / f'{subject}_task-emotion_keepmask.npy', keep_mask)

    # Step 5: Generate & Save Epoched Data
    print_timestamp('Saving final epoched data')
    epochs_clean = mne.Epochs(raw_continuous, exact_events, event_id=EVENTS_EMOTION, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, 
                              preload=True, baseline=None, verbose=False, reject_by_annotation=False)

    epochs_clean = epochs_clean[keep_mask]
    epochs_clean.save(out_folder / f'{subject}_task-emotion_clean-epo.fif', overwrite=True)

    frontal_chs = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
    row_stat = {
        'Subject': subject, 'Variant': variant_name, 'Task': 'Emotion_Faces', 
        'Total_Epochs': stats['Total_Epochs'], 'Clean_Epochs': keep_mask.sum(),
        '%_Rejected': round((stats['Total_Dropped'] / stats['Total_Epochs']) * 100, 2) if stats['Total_Epochs'] > 0 else 0,
        'Total_Bad_Chs': len(manual_bads), 'Bad_Chs_List': ", ".join(manual_bads),
        **{f'{ch}_Bad': (ch in manual_bads) for ch in frontal_chs},
    }

    report.save(report_file, open_browser=False, overwrite=True)
    return row_stat


if __name__ == '__main__':
    subject_list = sorted([d.name for d in SOURCE_DATA.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    start_idx = subject_list.index(START_SUBJECT) if START_SUBJECT in subject_list else 0
    end_idx = subject_list.index(END_SUBJECT) if END_SUBJECT in subject_list else len(subject_list) - 1
    selected_subjects = subject_list[start_idx:end_idx+1]
    
    # Load manual bad channels tracker
    if TRACKER_CSV_PATH.exists():
        tracker_df = pd.read_csv(TRACKER_CSV_PATH)
    else:
        print("WARNING: raw_inspection.csv not found! Assuming 0 manual bad channels.")
        tracker_df = pd.DataFrame()

    all_rejection_stats = []

    for subject in selected_subjects:
        base_out_folder = DERIVATIVES_DIR / subject / 'eeg'
        
        # 1. Fetch Manual Bads
        manual_bads = []
        subj_row = tracker_df[tracker_df['Subject'] == subject] if not tracker_df.empty else pd.DataFrame()
        if not subj_row.empty:
            bads_str = subj_row.iloc[0]['Bad_Channels']
            if pd.notna(bads_str) and isinstance(bads_str, str):
                manual_bads = [ch.strip() for ch in bads_str.split(',') if ch.strip()]
        
        print('\n' + '#' * 60)
        print(f'Preprocessing {subject} | Batch: {BATCH_NAME}')
        print('#' * 60)
        
        # 2. Determine Required Variants
        variants_to_run = [('Standard', manual_bads)]
        if 'F3' in manual_bads or 'F4' in manual_bads:
            # Create special list where F3 and F4 are explicitly OMITTED from the bads list 
            # (therefore treated as good, used in ICA, and NOT interpolated)
            omitted_bads = [ch for ch in manual_bads if ch not in ['F3', 'F4']]
            variants_to_run.append(('F3_F4_Omitted', omitted_bads))
            print_timestamp(">>> F3/F4 detected in bad channels! Preprocessing will run TWICE.")
            
        # 3. Execute Variants
        for variant_name, variant_bads in variants_to_run:
            try:
                subj_rejections = process_pipeline_variant(subject, base_out_folder, variant_bads, variant_name)
                if subj_rejections: all_rejection_stats.append(subj_rejections)
            except Exception as e: 
                print(f'Error processing {subject} [{variant_name}]: {e}')

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
BATCH_NAME = "Batch_02_Emotion_VirtualCz_8-12Hz_relaxed"
FAA_METHOD = 'diff'                  
ALPHA_BAND = (8.0, 12.0)             

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'processed' / BATCH_NAME
N_JOBS = -1

START_SUBJECT = "None"  
END_SUBJECT = "None"    

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
    """Filters continuous data to alpha bounds, applies Hilbert, and extracts FAA & Power.
    NOTE: Hilbert treats breakICA Annotations natively as standard continuous data."""
    alpha_raw = raw.copy().filter(l_freq=alpha_band[0], h_freq=alpha_band[1], n_jobs=N_JOBS, verbose=False)
    alpha_raw.apply_hilbert(envelope=True, verbose=False) 
    
    f3_amp = (alpha_raw.get_data(picks='F3').squeeze() * 1e6) 
    f4_amp = (alpha_raw.get_data(picks='F4').squeeze() * 1e6) 
    
    f3_pow = f3_amp ** 2
    f4_pow = f4_amp ** 2
    
    if method == 'diff': faa_series = np.log(f4_pow) - np.log(f3_pow)
    elif method == 'norm': faa_series = (np.log(f4_pow) - np.log(f3_pow)) / (np.log(f4_pow) + np.log(f3_pow))
    else: raise ValueError("FAA_METHOD must be either 'diff' or 'norm'")
        
    info_faa = mne.create_info(['FAA'], sfreq=raw.info['sfreq'], ch_types=['misc'])
    info_pow = mne.create_info(['F3_Pow', 'F4_Pow'], sfreq=raw.info['sfreq'], ch_types=['misc', 'misc'])
    
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
    
    raw_clean = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    exact_events = np.load(events_path)
    keep_mask = np.load(mask_path)
    
    raw_faa, raw_pow = extract_faa_continuous(raw_clean, ALPHA_BAND, FAA_METHOD)
    
    tmin, tmax = -2.0, 6.0
    epochs_faa = mne.Epochs(raw_faa, exact_events, event_id=EVENTS_EMOTION, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)
    epochs_pow = mne.Epochs(raw_pow, exact_events, event_id=EVENTS_EMOTION, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)

    # Base Rejection (from Cell 1 Mask)
    epochs_faa_clean = epochs_faa[keep_mask]
    epochs_pow_clean = epochs_pow[keep_mask]
    
    # --------------------------------------------------------------------------
    # 1.5 IQR Rejection on F3 & F4 Alpha Power
    # --------------------------------------------------------------------------
    print_timestamp('Applying 1.5 IQR Epoch Rejection on F3/F4 Alpha Power...')
    
    def get_iqr_mask(data_array):
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (data_array >= lower_bound) & (data_array <= upper_bound)
        
    # Get mean power for each epoch to check for IQR outliers
    f3_pow_epochs = epochs_pow_clean.get_data(picks='F3_Pow').squeeze().mean(axis=1)
    f4_pow_epochs = epochs_pow_clean.get_data(picks='F4_Pow').squeeze().mean(axis=1)
    
    f3_valid_mask = get_iqr_mask(f3_pow_epochs)
    f4_valid_mask = get_iqr_mask(f4_pow_epochs)
    
    # Keep epoch only if BOTH F3 and F4 are within IQR bounds
    final_iqr_mask = f3_valid_mask & f4_valid_mask
    epochs_faa_iqr = epochs_faa_clean[final_iqr_mask]
    
    print_timestamp(f"Dropped {len(epochs_faa_clean) - len(epochs_faa_iqr)} epochs via IQR Rejection.")

    # Save BOTH versions
    epochs_faa_clean.save(out_folder / f'{subject}_task-emotion_faa_{FAA_METHOD}-epo.fif', overwrite=True)
    epochs_faa_iqr.save(out_folder / f'{subject}_task-emotion_faa_{FAA_METHOD}_iqr-epo.fif', overwrite=True)
    
    # Aggregating Stats (Using the IQR-Cleaned data for final metrics)
    avg_f3 = np.mean(epochs_pow_clean[final_iqr_mask].get_data(picks='F3_Pow'))
    avg_f4 = np.mean(epochs_pow_clean[final_iqr_mask].get_data(picks='F4_Pow'))
    avg_faa = np.mean(epochs_faa_iqr.get_data(picks='FAA'))
    
    metrics = {
        'Subject': subject, 'Task': 'Emotion_Faces', 
        'Mean F3 Power': round(avg_f3, 6), 'Mean F4 Power': round(avg_f4, 6), 
        'Mean FAA Index': round(avg_faa, 6)
    }

    # Visualization (Plotting the fully cleaned IQR heatmap)
    heatmap_data = epochs_faa_iqr.get_data(picks='FAA').squeeze()
    if heatmap_data.ndim == 1: heatmap_data = heatmap_data.reshape(1, -1) 
    
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
    vbound = 3.0 if FAA_METHOD == 'diff' else 0.1
    im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                        extent=[epochs_faa_iqr.tmin, epochs_faa_iqr.tmax, heatmap_data.shape[0], 0], vmin=-vbound, vmax=vbound)
    ax_heat.axvline(0, color='black', linestyle='--', linewidth=2) 
    ax_heat.axvline(2, color='gray', linestyle='--', linewidth=1.5) 
    ax_heat.axvline(4, color='gray', linestyle='--', linewidth=1.5) 
    ax_heat.set_title(f'IQR-Cleaned Epoch FAA Heatmap ({subject} - Emotion)')
    fig_heat.colorbar(im, ax=ax_heat, label='FAA Index')
    plt.savefig(out_folder / f'{subject}_task-emotion_faa_{FAA_METHOD}-heatmap.png')
    plt.close(fig_heat)

    return metrics

if __name__ == '__main__':
    subject_dirs = sorted([d.name for d in DERIVATIVES_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    start_idx = subject_dirs.index(START_SUBJECT) if START_SUBJECT in subject_dirs else 0
    end_idx = subject_dirs.index(END_SUBJECT) if END_SUBJECT in subject_dirs else len(subject_dirs) - 1
    
    selected_subjects = subject_dirs[start_idx:end_idx+1]
    all_faa_stats = []

    for subject in selected_subjects:
        out_folder = DERIVATIVES_DIR / subject / 'eeg'
        
        if (out_folder / f'{subject}_task-emotion_faa_{FAA_METHOD}_iqr-epo.fif').exists():
            print(f"\nSkipping {subject}: FAA already processed.")
            continue

        print('\n' + '#' * 60)
        print(f'Running Frequency Domain & FAA: {subject}')
        print('#' * 60)
        
        try:
            # Note: The FAA analysis is run on the 'Standard' files here. 
            # The 'F3_F4_bad_channel_omission' versions exist as backups in their 
            # respective subfolders if you ever wish to manually point this script to them.
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