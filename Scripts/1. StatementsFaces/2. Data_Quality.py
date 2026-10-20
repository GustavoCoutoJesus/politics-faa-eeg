# %% #######################################################################################################################
# The goal of this script is to provide a quality check to the preprocessed data and build a report
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %%  Advanced Quality Control (Grouped SNR, Periodograms, Condition Overviews)
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import base64
from io import BytesIO
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# Configuration & Paths (USER SELECTABLE PARAMETERS)
# ==============================================================================
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"
FAA_METHOD = 'diff'  # Options: 'diff' or 'norm'
ALPHA_BAND = (8.0, 12.0)

EXCLUDE_SUBJECTS = [
    # 'sub-P03', 'sub-P12'
]

# Directories
PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids' / 'derivatives' / 'processed' / BATCH_NAME
QC_DIR = PROJECT_FOLDER / 'bids' / 'derivatives' / 'quality_check' / BATCH_NAME

os.makedirs(QC_DIR, exist_ok=True)

N_JOBS = -1
TASKS = ['JustFaces', 'StatementsFaces'] 

# ==============================================================================
# Base Event Dictionaries
# ==============================================================================
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

# ==============================================================================
# Grouping Sub-conditions into Theoretical Variables
# ==============================================================================
JF_GROUPS = {
    'JF_GL': [k for k in EVENTS_JUSTFACES if k.startswith('GL/')],
    'JF_PVV': [k for k in EVENTS_JUSTFACES if k.startswith('PVV/')]
}

SF_GROUPS = {
    'SF_GL_Left': [k for k in EVENTS_STATEMENTSFACES if k.startswith('GL/') and '/left/' in k],
    'SF_GL_Right': [k for k in EVENTS_STATEMENTSFACES if k.startswith('GL/') and '/right/' in k],
    'SF_PVV_Left': [k for k in EVENTS_STATEMENTSFACES if k.startswith('PVV/') and '/left/' in k],
    'SF_PVV_Right': [k for k in EVENTS_STATEMENTSFACES if k.startswith('PVV/') and '/right/' in k]
}

GROUPINGS = {'JustFaces': JF_GROUPS, 'StatementsFaces': SF_GROUPS}
SF_GRID_COLS = ['SF_GL_Left', 'SF_GL_Right', 'SF_PVV_Left', 'SF_PVV_Right']

# ==============================================================================
# Helper Functions
# ==============================================================================
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def compute_snr_on_subset(epochs_subset):
    if len(epochs_subset) == 0: return np.nan
    n_fft_safe = int(epochs_subset.info['sfreq'])
    psd = epochs_subset.compute_psd(fmin=4, fmax=16, method='welch', n_fft=n_fft_safe, n_jobs=N_JOBS, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)
    
    sig_mask = (freqs >= 8) & (freqs <= 12)
    noise_mask = ((freqs >= 4) & (freqs < 8)) | ((freqs > 12) & (freqs <= 16))
    
    sig_pow = np.mean(psd_data[:, :, sig_mask])
    noise_pow = np.mean(psd_data[:, :, noise_mask])
    return sig_pow / noise_pow if noise_pow > 0 else np.nan

# ==============================================================================
# Main Execution Loop
# ==============================================================================
def run_advanced_quality_checks():
    subject_dirs = sorted([d for d in DERIVATIVES_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    condition_data = []
    
    # Stores for global periodograms (6-14Hz)
    psd_6_14_f3, psd_6_14_f4 = [], []
    freqs_6_14 = None
    
    # Stores for 1-40 Hz Grid (SF Only)
    grid_psd = {}
    freqs_1_40 = None
    processed_subjects = []

    print(f"Starting Advanced QC. Excluding subjects: {EXCLUDE_SUBJECTS}")
    
    for subj_dir in subject_dirs:
        subject = subj_dir.name
        if subject in EXCLUDE_SUBJECTS:
            continue
            
        print(f"Processing: {subject}")
        grid_psd[subject] = {col: [] for col in SF_GRID_COLS}
        has_valid_sf_data = False

        for task in TASKS:
            raw_path = subj_dir / 'eeg' / f'{subject}_task-{task}_clean-raw.fif'
            events_path = subj_dir / 'eeg' / f'{subject}_task-{task}_events.npy'
            mask_path = subj_dir / 'eeg' / f'{subject}_task-{task}_keepmask.npy'
            
            if not raw_path.exists(): continue
                
            raw_clean = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
            exact_events = np.load(events_path)
            keep_mask = np.load(mask_path)
            
            event_dict = EVENTS_JUSTFACES if task == 'JustFaces' else EVENTS_STATEMENTSFACES
            tmin, tmax = (-1.2, 1.2) if task == 'StatementsFaces' else (0.0, 1.2)
            
            epochs = mne.Epochs(raw_clean, exact_events, event_id=event_dict, tmin=tmin, tmax=tmax, 
                                baseline=None, preload=True, verbose=False)
            epochs.shift_time(tshift=0.0125, relative=True)
            epochs_clean = epochs[keep_mask]
            
            # Precompute FAA continuous for Split-Half
            alpha_raw = raw_clean.copy().filter(l_freq=ALPHA_BAND[0], h_freq=ALPHA_BAND[1], verbose=False)
            alpha_raw.apply_hilbert(envelope=True, verbose=False)
            f3_pow = (alpha_raw.get_data(picks='F3').squeeze() * 1e6) ** 2
            f4_pow = (alpha_raw.get_data(picks='F4').squeeze() * 1e6) ** 2
            
            if FAA_METHOD == 'diff':
                faa_series = np.log(f4_pow) - np.log(f3_pow)
            else:
                faa_series = (np.log(f4_pow) - np.log(f3_pow)) / (np.log(f4_pow) + np.log(f3_pow))
            
            faa_raw = mne.io.RawArray(faa_series.reshape(1, -1), mne.create_info(['FAA'], raw_clean.info['sfreq'], ['misc']), first_samp=raw_clean.first_samp, verbose=False)
            epochs_faa = mne.Epochs(faa_raw, exact_events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)[keep_mask]

            # Calculate 6-14Hz Overall Periodograms (Task Agnostic, but per subject)
            if len(epochs_clean) > 0:
                n_fft_safe = int(epochs_clean.info['sfreq'])
                psd_6_14 = epochs_clean.compute_psd(fmin=6, fmax=14, picks=['F3', 'F4'], method='welch', n_fft=n_fft_safe, verbose=False)
                p_data, freqs_6_14 = psd_6_14.get_data(return_freqs=True)
                psd_6_14_f3.append(np.mean(p_data[:, 0, :], axis=0))
                psd_6_14_f4.append(np.mean(p_data[:, 1, :], axis=0))

            # Iterate over the JF/SF Groupings
            current_groups = GROUPINGS[task]
            for group_name, group_keys in current_groups.items():
                
                valid_keys = [k for k in group_keys if k in epochs_clean.event_id]
                if not valid_keys: continue
                    
                ep_cond = epochs_clean[valid_keys]
                if len(ep_cond) == 0: continue
                    
                # 1. SNR
                snr = compute_snr_on_subset(ep_cond)
                
                # 2. Split-Half FAA
                ep_faa = epochs_faa[valid_keys].get_data(picks='FAA').squeeze()
                if ep_faa.ndim == 1: ep_faa = ep_faa.reshape(1, -1)
                
                faa_means = np.mean(ep_faa, axis=-1)
                g1_faa = np.mean(faa_means[0::2]) if len(faa_means) >= 2 else np.nan
                g2_faa = np.mean(faa_means[1::2]) if len(faa_means) >= 2 else np.nan
                
                condition_data.append({
                    'Subject': subject, 'Task': task, 'Condition_Group': group_name, 
                    'Alpha_SNR': snr, 'FAA_Odd': g1_faa, 'FAA_Even': g2_faa
                })
                
                # 3. 1-40 Hz Grid Data (SF Only)
                if task == 'StatementsFaces':
                    has_valid_sf_data = True
                    n_fft_safe = int(epochs_clean.info['sfreq'])
                    psd_1_40 = epochs_clean[valid_keys].compute_psd(fmin=1, fmax=40, picks=['eeg'], method='welch', n_fft=n_fft_safe, verbose=False)
                    p_data_1_40, freqs_1_40 = psd_1_40.get_data(return_freqs=True)
                    grid_psd[subject][group_name] = np.mean(p_data_1_40, axis=(0, 1))

        if has_valid_sf_data: processed_subjects.append(subject)

    # ==============================================================================
    # Aggregation & Statistics
    # ==============================================================================
    df_cond = pd.DataFrame(condition_data)
    
    summary_list = []
    if not df_cond.empty:
        for cond, group in df_cond.groupby('Condition_Group'):
            mean_snr = group['Alpha_SNR'].mean()
            sem_snr = group['Alpha_SNR'].std() / np.sqrt(len(group))
            
            valid_split = group.dropna(subset=['FAA_Odd', 'FAA_Even'])
            if len(valid_split) > 2:
                r_val, _ = pearsonr(valid_split['FAA_Odd'], valid_split['FAA_Even'])
            else: r_val = np.nan
                
            summary_list.append({'Condition': cond, 'Mean_SNR': mean_snr, 'SEM_SNR': sem_snr, 'Split_Half_r': r_val})
            
    df_summary = pd.DataFrame(summary_list).sort_values('Condition')
    
    # ==============================================================================
    # Visualizations
    # ==============================================================================
    
    # A) Overview: SNR
    fig_snr, ax_snr = plt.subplots(figsize=(10, 6))
    if not df_summary.empty:
        x_pos = np.arange(len(df_summary))
        ax_snr.bar(x_pos, df_summary['Mean_SNR'], yerr=df_summary['SEM_SNR'], capsize=5, color='teal', alpha=0.7)
        ax_snr.set_xticks(x_pos)
        ax_snr.set_xticklabels(df_summary['Condition'], rotation=45, ha="right", fontsize=10)
        ax_snr.set_ylabel("Alpha SNR")
        ax_snr.set_title("Mean Alpha SNR (+/- SEM) by Party/Congruence Groupings")
    fig_snr.tight_layout()
    b64_snr = fig_to_base64(fig_snr)

    # B) Overview: Split-Half
    fig_split, ax_split = plt.subplots(figsize=(10, 6))
    if not df_summary.empty:
        ax_split.bar(x_pos, df_summary['Split_Half_r'], color='coral', alpha=0.7)
        ax_split.axhline(0, color='black', linewidth=0.8)
        ax_split.set_xticks(x_pos)
        ax_split.set_xticklabels(df_summary['Condition'], rotation=45, ha="right", fontsize=10)
        ax_split.set_ylabel("Pearson r")
        ax_split.set_title("Split-Half Reliability (Odd vs Even) by Party/Congruence Groupings")
    fig_split.tight_layout()
    b64_split = fig_to_base64(fig_split)

    # C) 6-14 Hz Periodogram
    if psd_6_14_f3 and freqs_6_14 is not None:
        psd_f3_arr = np.array(psd_6_14_f3) * 1e12 
        psd_f4_arr = np.array(psd_6_14_f4) * 1e12
        
        mean_f3, sem_f3 = np.mean(psd_f3_arr, axis=0), np.std(psd_f3_arr, axis=0)/np.sqrt(len(psd_f3_arr))
        mean_f4, sem_f4 = np.mean(psd_f4_arr, axis=0), np.std(psd_f4_arr, axis=0)/np.sqrt(len(psd_f4_arr))
        
        fig_perio, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(freqs_6_14, mean_f3, color='blue', label='F3 Mean')
        ax1.fill_between(freqs_6_14, mean_f3 - sem_f3, mean_f3 + sem_f3, color='blue', alpha=0.2)
        ax1.set_title("Grand Average F3 Power (6-14 Hz)")
        ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Power")
        
        ax2.plot(freqs_6_14, mean_f4, color='orange', label='F4 Mean')
        ax2.fill_between(freqs_6_14, mean_f4 - sem_f4, mean_f4 + sem_f4, color='orange', alpha=0.2)
        ax2.set_title("Grand Average F4 Power (6-14 Hz)")
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Power")
        b64_perio = fig_to_base64(fig_perio)
    else:
        b64_perio = ""

    # D) 1-40 Hz Grid
    if processed_subjects and freqs_1_40 is not None:
        fig_grid, axes_grid = plt.subplots(nrows=len(processed_subjects), ncols=4, 
                                           figsize=(20, 3 * len(processed_subjects)), sharex=True)
        if len(processed_subjects) == 1: axes_grid = np.expand_dims(axes_grid, axis=0) 

        for i, sub in enumerate(processed_subjects):
            for j, cond_col in enumerate(SF_GRID_COLS):
                ax = axes_grid[i, j]
                data = grid_psd[sub].get(cond_col, [])
                if len(data) > 0:
                    ax.plot(freqs_1_40, data * 1e12, color='darkgreen')
                
                if i == 0: ax.set_title(cond_col.replace('SF_', '').replace('_', ' '))
                if j == 0: ax.set_ylabel(f"{sub}\nPower")
                if i == len(processed_subjects) - 1: ax.set_xlabel("Frequency (Hz)")
        
        fig_grid.tight_layout()
        b64_grid = fig_to_base64(fig_grid)
    else:
        b64_grid = ""

    # ==============================================================================
    # HTML Export
    # ==============================================================================
    html_content = f"""
    <html>
    <head><title>Advanced Grouped QC Report</title></head>
    <body style="font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9;">
        <h1 style="color: #2c3e50;">Advanced Grouped Quality Check Report</h1>
        <p><strong>Batch:</strong> {BATCH_NAME} | <strong>FAA Method:</strong> {FAA_METHOD}</p>
        <p><strong>Excluded Subjects:</strong> {EXCLUDE_SUBJECTS}</p>
        <hr>
        
        <h2 style="color: #34495e;">1. Overview: SNR & Split-Half Correlation by Theoretical Condition</h2>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <img src="data:image/png;base64,{b64_snr}" style="max-width: 100%; margin-bottom: 20px;"><br>
            <img src="data:image/png;base64,{b64_split}" style="max-width: 100%;">
        </div>
        
        <hr>
        <h2 style="color: #34495e;">2. Alpha Band Power (6-14 Hz) - Grand Average Across All Conditions</h2>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            {f'<img src="data:image/png;base64,{b64_perio}" style="max-width: 100%;">' if b64_perio else '<p>No 6-14Hz data generated.</p>'}
        </div>
        
        <hr>
        <h2 style="color: #34495e;">3. Broad Band Power (1-40 Hz) - Per Participant, Per Congruence Group (SF Only)</h2>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            {f'<img src="data:image/png;base64,{b64_grid}" style="max-width: 100%;">' if b64_grid else '<p>No SF data processed.</p>'}
        </div>
        
        <hr>
        <h2 style="color: #34495e;">4. Aggregate Data Summary</h2>
        {df_summary.to_html(index=False, float_format="%.5f", border=1)}
        
    </body>
    </html>
    """
    
    with open(QC_DIR / 'Advanced_Grouped_QC_Report.html', 'w') as f:
        f.write(html_content)
    
    df_cond.to_csv(QC_DIR / 'Advanced_Grouped_QC_ConditionLevelData.csv', index=False)
    
    print(f"\nSaved Grouped QC HTML to: {QC_DIR / 'Advanced_Grouped_QC_Report.html'}")
    print(f"Saved Grouped Raw CSV to: {QC_DIR / 'Advanced_Grouped_QC_ConditionLevelData.csv'}")

if __name__ == '__main__':
    run_advanced_quality_checks()
    print("Advanced Grouped QC completed successfully.")
# %% Standalone F3 / F4 Bad Channel Detector
import os
import pandas as pd
from pathlib import Path

# ==============================================================================
# Configuration & Paths 
# ==============================================================================
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"

# Path to the primary Data Quality Report generated by Cell 1
PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
CSV_PATH = PROJECT_FOLDER / 'bids' / 'derivatives' / 'processed' / BATCH_NAME / f"Data_Quality_Report_{BATCH_NAME}.csv"
QC_DIR = PROJECT_FOLDER / 'bids' / 'derivatives' / 'quality_check' / BATCH_NAME

os.makedirs(QC_DIR, exist_ok=True)

def find_and_flag_bad_channels():
    print("==========================================================")
    print("Running Standalone F3 / F4 Bad Channel Detector")
    print("==========================================================")
    
    if not CSV_PATH.exists():
        print(f"ERROR: Cannot find the report file at:\n{CSV_PATH}")
        print("Please ensure Cell 1 has run and generated the Data Quality CSV.")
        return

    # Read the data quality report
    df = pd.read_csv(CSV_PATH)
    
    # Check if the specific columns exist in the dataframe
    required_cols = ['Subject', 'Task', 'F3_Bad', 'F4_Bad']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ERROR: The CSV is missing required columns: {missing_cols}")
        return

    # Filter rows where either F3 or F4 was marked as True (Bad)
    bad_mask = (df['F3_Bad'] == True) | (df['F4_Bad'] == True)
    flagged_df = df[bad_mask].copy()

    if flagged_df.empty:
        print("\nSUCCESS: No participants have bad F3 or F4 channels in any task!")
        
        # Create an empty CSV just to confirm it ran and found nothing
        empty_df = pd.DataFrame(columns=['Subject', 'Task', 'Bad_Channels'])
        empty_df.to_csv(QC_DIR / 'Flagged_Bad_F3_F4_Participants.csv', index=False)
        return

    print(f"\nWARNING: Found {len(flagged_df)} instance(s) where F3 or F4 was rejected.\n")
    
    export_list = []
    
    for index, row in flagged_df.iterrows():
        bads = []
        if row['F3_Bad'] == True: bads.append('F3')
        if row['F4_Bad'] == True: bads.append('F4')
        
        bads_str = ", ".join(bads)
        print(f" - Subject: {row['Subject']:<10} | Task: {row['Task']:<15} | Bad Channels: {bads_str}")
        
        export_list.append({
            'Subject': row['Subject'],
            'Task': row['Task'],
            'Bad_Channels': bads_str
        })

    # Export to a clean CSV in the advanced QC folder
    export_df = pd.DataFrame(export_list)
    export_path = QC_DIR / 'Flagged_Bad_F3_F4_Participants.csv'
    export_df.to_csv(export_path, index=False)
    
    print(f"\nFlagged participants saved to:\n{export_path}")

if __name__ == '__main__':
    find_and_flag_bad_channels()
# %%