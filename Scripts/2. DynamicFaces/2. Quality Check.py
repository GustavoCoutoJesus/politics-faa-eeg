# %% #######################################################################################################################
# The goal of this script is to provide a quality check to the preprocessed data and build a report
# The first run corresponds to the first batch, before we relaxed the preprocessing procedure.
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Advanced Quality Control (Grouped SNR, Periodograms, Condition Overviews) & Flagging
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
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
FAA_METHOD = 'diff'  # Options: 'diff' or 'norm'
ALPHA_BAND = (8.0, 12.0)

# USER EXCLUSION LIST: Add 'sub-XXX' strings to exclude specific participants
EXCLUDE_SUBJECTS = [
    # 'sub-005_MU_pilot'
]

# Directories
PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'processed' / BATCH_NAME
QC_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'quality_check' / BATCH_NAME
CSV_PATH = DERIVATIVES_DIR / f"Data_Quality_Report_{BATCH_NAME}.csv"

os.makedirs(QC_DIR, exist_ok=True)

N_JOBS = -1
TASK = 'emotion'

# ==============================================================================
# Base Event Dictionaries
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
# Grouping Sub-conditions into Theoretical Variables
# ==============================================================================
EMOTION_GROUPS = {
    'In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty' in k],
    'Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty' in k],
    'Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol' in k],
    
    'Happy In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty/happy' in k],
    'Happy Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty/happy' in k],
    'Angry In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty/angry' in k],
    'Angry Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty/angry' in k],
    'Neutral In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty/neutral' in k],
    'Neutral Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty/neutral' in k],
    
    'Angry Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol/angry' in k],
    'Happy Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol/happy' in k],
    'Neutral Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol/neutral' in k]
}

# Sub-selection for the 1-40 Hz Grid View
GRID_COLS = ['In-Party', 'Out-Party', 'Non-Politician']

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
# Main QC Execution
# ==============================================================================
def run_advanced_quality_checks():
    subject_dirs = sorted([d for d in DERIVATIVES_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    condition_data = []
    psd_6_14_f3, psd_6_14_f4 = [], []
    freqs_6_14 = None
    
    grid_psd = {}
    freqs_1_40 = None
    processed_subjects = []

    print(f"Starting Advanced QC. Excluding subjects: {EXCLUDE_SUBJECTS}")
    
    for subj_dir in subject_dirs:
        subject = subj_dir.name
        if subject in EXCLUDE_SUBJECTS:
            continue
            
        print(f"Processing: {subject}")
        grid_psd[subject] = {col: [] for col in GRID_COLS}
        
        raw_path = subj_dir / 'eeg' / f'{subject}_task-{TASK}_clean-raw.fif'
        events_path = subj_dir / 'eeg' / f'{subject}_task-{TASK}_events.npy'
        mask_path = subj_dir / 'eeg' / f'{subject}_task-{TASK}_keepmask.npy'
        
        if not raw_path.exists(): 
            print(f"  Skipping (missing files).")
            continue
            
        raw_clean = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        exact_events = np.load(events_path)
        keep_mask = np.load(mask_path)
        
        tmin, tmax = -2.0, 6.0 
        
        epochs = mne.Epochs(raw_clean, exact_events, event_id=EVENTS_EMOTION, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)
        epochs_clean = epochs[keep_mask]
        
        # Precompute FAA continuous for Split-Half
        alpha_raw = raw_clean.copy().filter(l_freq=ALPHA_BAND[0], h_freq=ALPHA_BAND[1], n_jobs=N_JOBS, verbose=False)
        alpha_raw.apply_hilbert(envelope=True, verbose=False)
        f3_pow = (alpha_raw.get_data(picks='F3').squeeze() * 1e6) ** 2
        f4_pow = (alpha_raw.get_data(picks='F4').squeeze() * 1e6) ** 2
        
        if FAA_METHOD == 'diff':
            faa_series = np.log(f4_pow) - np.log(f3_pow)
        else:
            faa_series = (np.log(f4_pow) - np.log(f3_pow)) / (np.log(f4_pow) + np.log(f3_pow))
        
        faa_raw = mne.io.RawArray(faa_series.reshape(1, -1), mne.create_info(['FAA'], raw_clean.info['sfreq'], ['misc']), first_samp=raw_clean.first_samp, verbose=False)
        epochs_faa = mne.Epochs(faa_raw, exact_events, event_id=EVENTS_EMOTION, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)[keep_mask]

        # Calculate 6-14Hz Overall Periodograms (Task Agnostic, but per subject)
        if len(epochs_clean) > 0:
            n_fft_safe = int(epochs_clean.info['sfreq'])
            psd_6_14 = epochs_clean.compute_psd(fmin=6, fmax=14, picks=['F3', 'F4'], method='welch', n_fft=n_fft_safe, n_jobs=N_JOBS, verbose=False)
            p_data, freqs_6_14 = psd_6_14.get_data(return_freqs=True)
            psd_6_14_f3.append(np.mean(p_data[:, 0, :], axis=0))
            psd_6_14_f4.append(np.mean(p_data[:, 1, :], axis=0))

        # Iterate over Emotion Groupings
        for group_name, group_keys in EMOTION_GROUPS.items():
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
                'Subject': subject, 'Task': TASK, 'Condition_Group': group_name, 
                'Alpha_SNR': snr, 'FAA_Odd': g1_faa, 'FAA_Even': g2_faa
            })
            
            # 3. 1-40 Hz Grid Data (for primary columns only to avoid grid bloat)
            if group_name in GRID_COLS:
                n_fft_safe = int(epochs_clean.info['sfreq'])
                psd_1_40 = epochs_clean[valid_keys].compute_psd(fmin=1, fmax=40, picks=['eeg'], method='welch', n_fft=n_fft_safe, n_jobs=N_JOBS, verbose=False)
                p_data_1_40, freqs_1_40 = psd_1_40.get_data(return_freqs=True)
                grid_psd[subject][group_name] = np.mean(p_data_1_40, axis=(0, 1))

        processed_subjects.append(subject)

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
    fig_snr, ax_snr = plt.subplots(figsize=(12, 6))
    if not df_summary.empty:
        x_pos = np.arange(len(df_summary))
        ax_snr.bar(x_pos, df_summary['Mean_SNR'], yerr=df_summary['SEM_SNR'], capsize=5, color='teal', alpha=0.7)
        ax_snr.set_xticks(x_pos)
        ax_snr.set_xticklabels(df_summary['Condition'], rotation=45, ha="right", fontsize=10)
        ax_snr.set_ylabel("Alpha SNR")
        ax_snr.set_title("Mean Alpha SNR (+/- SEM) by Emotion Groupings")
    fig_snr.tight_layout()
    b64_snr = fig_to_base64(fig_snr)

    # B) Overview: Split-Half
    fig_split, ax_split = plt.subplots(figsize=(12, 6))
    if not df_summary.empty:
        ax_split.bar(x_pos, df_summary['Split_Half_r'], color='coral', alpha=0.7)
        ax_split.axhline(0, color='black', linewidth=0.8)
        ax_split.set_xticks(x_pos)
        ax_split.set_xticklabels(df_summary['Condition'], rotation=45, ha="right", fontsize=10)
        ax_split.set_ylabel("Pearson r")
        ax_split.set_title("Split-Half Reliability (Odd vs Even) by Emotion Groupings")
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
        fig_grid, axes_grid = plt.subplots(nrows=len(processed_subjects), ncols=len(GRID_COLS), 
                                           figsize=(15, 3 * len(processed_subjects)), sharex=True)
        if len(processed_subjects) == 1: axes_grid = np.expand_dims(axes_grid, axis=0) 

        for i, sub in enumerate(processed_subjects):
            for j, cond_col in enumerate(GRID_COLS):
                ax = axes_grid[i, j]
                data = grid_psd[sub].get(cond_col, [])
                if len(data) > 0:
                    ax.plot(freqs_1_40, data * 1e12, color='darkgreen')
                
                if i == 0: ax.set_title(cond_col)
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
        
        <h2 style="color: #34495e;">1. Overview: SNR & Split-Half Correlation by Emotion Condition</h2>
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
        <h2 style="color: #34495e;">3. Broad Band Power (1-40 Hz) - Per Participant, Per Group</h2>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            {f'<img src="data:image/png;base64,{b64_grid}" style="max-width: 100%;">' if b64_grid else '<p>No frequency grid data generated.</p>'}
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
    print(f"\nSaved Grouped QC HTML & CSV to: {QC_DIR}")

# ==============================================================================
# Standalone Bad Channel Detector
# ==============================================================================
def find_and_flag_bad_channels():
    print("\n==========================================================")
    print("Running Standalone F3 / F4 Bad Channel Detector")
    print("==========================================================")
    
    if not CSV_PATH.exists():
        print(f"ERROR: Cannot find the report file at:\n{CSV_PATH}")
        print("Please ensure your preprocessing script (Cell 1) generated the Data Quality CSV.")
        return

    df = pd.read_csv(CSV_PATH)
    required_cols = ['Subject', 'Task', 'F3_Bad', 'F4_Bad']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ERROR: The CSV is missing required columns: {missing_cols}")
        return

    # Filter to current task and bad channels
    df_task = df[df['Task'] == 'Emotion_Faces']
    bad_mask = (df_task['F3_Bad'] == True) | (df_task['F4_Bad'] == True)
    flagged_df = df_task[bad_mask].copy()

    export_path = QC_DIR / 'Flagged_Bad_F3_F4_Participants.csv'

    if flagged_df.empty:
        print("\nSUCCESS: No participants have bad F3 or F4 channels in the Emotion task!")
        empty_df = pd.DataFrame(columns=['Subject', 'Task', 'Bad_Channels'])
        empty_df.to_csv(export_path, index=False)
        return

    print(f"\nWARNING: Found {len(flagged_df)} instance(s) where F3 or F4 was rejected.\n")
    
    export_list = []
    for _, row in flagged_df.iterrows():
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

    pd.DataFrame(export_list).to_csv(export_path, index=False)
    print(f"\nFlagged participants saved to:\n{export_path}")

if __name__ == '__main__':
    run_advanced_quality_checks()
    find_and_flag_bad_channels()

# %% Cell 2: Epoch Rejection Flagging (30% and 50% Thresholds)
def find_and_flag_rejected_epochs():
    print("\n==========================================================")
    print("Running Epoch Rejection Detector (>=30% and >=50%)")
    print("==========================================================")
    
    if not CSV_PATH.exists():
        print(f"ERROR: Cannot find the report file at:\n{CSV_PATH}")
        print("Please ensure your preprocessing script generated the Data Quality CSV.")
        return

    df = pd.read_csv(CSV_PATH)
    
    required_cols = ['Subject', 'Task', '%_Rejected']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ERROR: The CSV is missing required columns: {missing_cols}")
        return

    # Filter to current task
    df_task = df[df['Task'] == 'Emotion_Faces'].copy()
    
    # Create the boolean flag columns
    df_task['Flag_>=30%'] = df_task['%_Rejected'] >= 30.0
    df_task['Flag_>=50%'] = df_task['%_Rejected'] >= 50.0

    # Filter the dataframe to keep only participants who hit at least the 30% threshold
    flagged_df = df_task[(df_task['Flag_>=30%'] == True) | (df_task['Flag_>=50%'] == True)].copy()

    export_path = QC_DIR / 'Flagged_Rejected_Epochs_Participants.csv'

    # Get subset of columns for cleaner export
    output_df = flagged_df[['Subject', 'Task', '%_Rejected', 'Flag_>=30%', 'Flag_>=50%']]

    # Count totals
    count_30_plus = df_task['Flag_>=30%'].sum()
    count_50_plus = df_task['Flag_>=50%'].sum()

    if flagged_df.empty:
        print("\nSUCCESS: No participants lost 30% or more of their epochs!")
        pd.DataFrame(columns=['Subject', 'Task', '%_Rejected', 'Flag_>=30%', 'Flag_>=50%']).to_csv(export_path, index=False)
        return

    print(f"\nWARNING: Found participants with high epoch rejection rates!")
    print(f" - Participants with >= 30% rejected epochs: {count_30_plus}")
    print(f" - Participants with >= 50% rejected epochs: {count_50_plus}\n")
    
    for _, row in output_df.iterrows():
        print(f" - Subject: {row['Subject']:<10} | Rejected: {row['%_Rejected']:<5}% | Flag >=30%: {row['Flag_>=30%']} | Flag >=50%: {row['Flag_>=50%']}")
        
    output_df.to_csv(export_path, index=False)
    print(f"\nFlagged participants saved to:\n{export_path}")

if __name__ == '__main__':
    # Add this call to the execution block if running all cells at once
    # run_advanced_quality_checks()
    # find_and_flag_bad_channels()
    find_and_flag_rejected_epochs()

# %% Cell 3: Comprehensive Exclusion List (Bad Channels + High Rejection)
def generate_comprehensive_exclusion_list():
    print("\n==========================================================")
    print("Generating Comprehensive Exclusion List (Channels + Epochs)")
    print("==========================================================")
    
    if not CSV_PATH.exists():
        print(f"ERROR: Cannot find the report file at:\n{CSV_PATH}")
        print("Please ensure your preprocessing script generated the Data Quality CSV.")
        return

    df = pd.read_csv(CSV_PATH)
    
    required_cols = ['Subject', 'Task', 'F3_Bad', 'F4_Bad', '%_Rejected']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ERROR: The CSV is missing required columns: {missing_cols}")
        return

    # Filter to current task
    df_task = df[df['Task'] == 'Emotion_Faces'].copy()
    
    # 1. Determine specific flags based on previous logic
    df_task['Bad_F3_or_F4'] = df_task['F3_Bad'] | df_task['F4_Bad']
    df_task['Reject_>=30%'] = df_task['%_Rejected'] >= 30.0
    df_task['Reject_>=50%'] = df_task['%_Rejected'] >= 50.0
    
    # 2. Create combined exclusion columns (Strict vs Lenient)
    # Strict: Exclude if bad channels OR >= 30% epochs lost
    df_task['Exclude_Strict'] = df_task['Bad_F3_or_F4'] | df_task['Reject_>=30%']
    
    # Lenient: Exclude if bad channels OR >= 50% epochs lost
    df_task['Exclude_Lenient'] = df_task['Bad_F3_or_F4'] | df_task['Reject_>=50%']

    # 3. Filter out only those who trigger at least the strict exclusion 
    # (This captures everyone who has *any* of the flags so your CSV stays clean)
    flagged_df = df_task[df_task['Exclude_Strict']].copy()

    export_path = QC_DIR / 'Comprehensive_Exclusions_List.csv'

    out_cols = [
        'Subject', 'Task', 
        'Bad_F3_or_F4', 'Reject_>=30%', 'Reject_>=50%', 
        'Exclude_Strict', 'Exclude_Lenient'
    ]
    output_df = flagged_df[out_cols]

    if flagged_df.empty:
        print("\nSUCCESS: No participants flagged for exclusion (Channels or Epochs)!")
        pd.DataFrame(columns=out_cols).to_csv(export_path, index=False)
        return

    # Count totals for printing
    count_strict = output_df['Exclude_Strict'].sum()
    count_lenient = output_df['Exclude_Lenient'].sum()

    print(f"\nSUMMARY OF POTENTIAL EXCLUSIONS:")
    print(f" - Strict Criteria (Bad F3/F4 OR >=30% rejected): {count_strict} participants")
    print(f" - Lenient Criteria (Bad F3/F4 OR >=50% rejected): {count_lenient} participants\n")
    
    for _, row in output_df.iterrows():
        # Build a readable string of *why* they were flagged
        reasons = []
        if row['Bad_F3_or_F4']: reasons.append('Bad Channels')
        if row['Reject_>=50%']: reasons.append('>=50% Epochs')
        elif row['Reject_>=30%']: reasons.append('>=30% Epochs')
            
        print(f" - {row['Subject']:<10} | Flagged for: {', '.join(reasons):<25} | Strict Exclude: {row['Exclude_Strict']} | Lenient Exclude: {row['Exclude_Lenient']}")
        
    output_df.to_csv(export_path, index=False)
    print(f"\nComprehensive exclusion list saved to:\n{export_path}")

if __name__ == '__main__':
    # run_advanced_quality_checks()
    # find_and_flag_bad_channels()
    # find_and_flag_rejected_epochs()
    generate_comprehensive_exclusion_list()

#This run corresponds to the updated batch, with a relaxed preprocessing script
# %% Advanced Quality Control, Epoch Tracking & Comprehensive Flagging
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
# Configuration & Paths 
# ==============================================================================
BATCH_NAME = "Batch_02_Emotion_VirtualCz_8-12Hz_relaxed"
FAA_METHOD = 'diff'  # Options: 'diff' or 'norm'
ALPHA_BAND = (8.0, 12.0)

# USER EXCLUSION LIST: Add 'sub-XXX' strings to manually exclude specific participants from plots
EXCLUDE_SUBJECTS = []

# Directories
PROJECT_FOLDER = Path("C:/Users/gcoutod/OneDrive - UvA/Amsterdam Team/Research/Projects - Gustavo/Politics_EEG/TYP/analysis")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'processed' / BATCH_NAME
QC_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives' / 'quality_check' / BATCH_NAME
CSV_PATH = DERIVATIVES_DIR / f"Data_Quality_Report_{BATCH_NAME}.csv"
TRACKER_CSV_PATH = PROJECT_FOLDER / 'bids_emotion' / 'code' / 'raw_inspection.csv'

os.makedirs(QC_DIR, exist_ok=True)

N_JOBS = -1
TASK = 'emotion'

# ==============================================================================
# Base Event Dictionaries & Groupings
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

EMOTION_GROUPS = {
    'In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty' in k],
    'Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty' in k],
    'Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol' in k],
    'Happy In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty/happy' in k],
    'Happy Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty/happy' in k],
    'Angry In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty/angry' in k],
    'Angry Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty/angry' in k],
    'Neutral In-Party': [k for k in EVENTS_EMOTION if 'pol/inparty/neutral' in k],
    'Neutral Out-Party': [k for k in EVENTS_EMOTION if 'pol/outparty/neutral' in k],
    'Angry Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol/angry' in k],
    'Happy Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol/happy' in k],
    'Neutral Non-Politician': [k for k in EVENTS_EMOTION if 'nonpol/neutral' in k]
}

# Sub-selection for the 1-40 Hz Grid View
GRID_COLS = ['In-Party', 'Out-Party', 'Non-Politician']

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
# Main QC Execution
# ==============================================================================
def run_comprehensive_quality_checks():
    subject_dirs = sorted([d for d in DERIVATIVES_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    # Load External CSVs for flag logic
    tracker_df = pd.read_csv(TRACKER_CSV_PATH) if TRACKER_CSV_PATH.exists() else pd.DataFrame()
    dq_df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame()
    
    condition_data = []
    participant_summary = []
    
    psd_6_14_f3, psd_6_14_f4 = [], []
    freqs_6_14, freqs_1_40 = None, None
    grid_psd = {}
    processed_subjects = []

    print(f"Starting Comprehensive QC. Excluding subjects: {EXCLUDE_SUBJECTS}\n")
    
    for subj_dir in subject_dirs:
        subject = subj_dir.name
        if subject in EXCLUDE_SUBJECTS: continue
            
        # Define File Paths
        clean_raw_path = subj_dir / 'eeg' / f'{subject}_task-{TASK}_clean-raw.fif'
        clean_epo_path = subj_dir / 'eeg' / f'{subject}_task-{TASK}_clean-epo.fif'
        iqr_epo_path = subj_dir / 'eeg' / f'{subject}_task-{TASK}_faa_{FAA_METHOD}_iqr-epo.fif'
        
        if not (clean_epo_path.exists() and iqr_epo_path.exists()): 
            print(f"  Skipping {subject} (missing epoched files).")
            continue
            
        print(f"Processing: {subject}")
        grid_psd[subject] = {col: {} for col in GRID_COLS}
        
        # ----------------------------------------------------------------------
        # 1. Fetch Flags & Track Epoch Attrition
        # ----------------------------------------------------------------------
        # Tracker Data
        is_bad_data = False
        if not tracker_df.empty:
            s_row = tracker_df[tracker_df['Subject'] == subject]
            if not s_row.empty and str(s_row.iloc[0].get('Bad_Data', '')).strip().lower() == 'bad_data':
                is_bad_data = True

        # Preprocessing Data (From Data Quality Report)
        f3_bad, f4_bad = False, False
        total_raw_epochs = 0
        if not dq_df.empty:
            dq_row = dq_df[(dq_df['Subject'] == subject) & (dq_df['Variant'] == 'Standard')]
            if not dq_row.empty:
                f3_bad = bool(dq_row.iloc[0].get('F3_Bad', False))
                f4_bad = bool(dq_row.iloc[0].get('F4_Bad', False))
                total_raw_epochs = int(dq_row.iloc[0].get('Total_Epochs', 0))

        # Epoch Tracking
        epochs_clean = mne.read_epochs(clean_epo_path, preload=True, verbose=False)
        epochs_iqr = mne.read_epochs(iqr_epo_path, preload=True, verbose=False)
        
        preproc_clean_count = len(epochs_clean)
        final_iqr_count = len(epochs_iqr)
        
        preproc_dropped = total_raw_epochs - preproc_clean_count
        iqr_dropped = preproc_clean_count - final_iqr_count
        perc_dropped = ((preproc_dropped + iqr_dropped) / total_raw_epochs) * 100 if total_raw_epochs > 0 else 100.0

        # Master Flag Logic (Lenient definition: >50% data lost OR flagged bad)
        master_bad_flag = is_bad_data or f3_bad or f4_bad or (perc_dropped >= 50.0)
        
        bad_ch_str = []
        if f3_bad: bad_ch_str.append('F3')
        if f4_bad: bad_ch_str.append('F4')

        participant_summary.append({
            'Subject': subject,
            'Tracker_Bad_Data': is_bad_data,
            'Bad_F3_F4': ", ".join(bad_ch_str) if bad_ch_str else "None",
            'Total_Raw_Epochs': total_raw_epochs,
            'Preproc_Dropped': preproc_dropped,
            'IQR_Dropped': iqr_dropped,
            'Final_Valid_Epochs': final_iqr_count,
            'Total_Dropped_%': round(perc_dropped, 2),
            'Master_Bad_Flag': master_bad_flag
        })

        # ----------------------------------------------------------------------
        # 2. Extract Electrophysiological Metrics
        # ----------------------------------------------------------------------
        # Periodogram (Task Agnostic)
        if len(epochs_clean) > 0:
            n_fft_safe = int(epochs_clean.info['sfreq'])
            psd_6_14 = epochs_clean.compute_psd(fmin=6, fmax=14, picks=['F3', 'F4'], method='welch', n_fft=n_fft_safe, n_jobs=N_JOBS, verbose=False)
            p_data, freqs_6_14 = psd_6_14.get_data(return_freqs=True)
            psd_6_14_f3.append(np.mean(p_data[:, 0, :], axis=0))
            psd_6_14_f4.append(np.mean(p_data[:, 1, :], axis=0))

        # Condition Iteration
        for group_name, group_keys in EMOTION_GROUPS.items():
            valid_keys_clean = [k for k in group_keys if k in epochs_clean.event_id]
            valid_keys_iqr = [k for k in group_keys if k in epochs_iqr.event_id]
            
            if not valid_keys_clean: continue
                
            # SNR computed on Preproc-Cleaned (General Alpha Health)
            snr = compute_snr_on_subset(epochs_clean[valid_keys_clean])
            
            # Split-Half computed on Final IQR-Cleaned FAA data
            if valid_keys_iqr and len(epochs_iqr[valid_keys_iqr]) > 0:
                ep_faa = epochs_iqr[valid_keys_iqr].get_data(picks='FAA').squeeze()
                if ep_faa.ndim == 1: ep_faa = ep_faa.reshape(1, -1)
                
                faa_means = np.mean(ep_faa, axis=-1)
                g1_faa = np.mean(faa_means[0::2]) if len(faa_means) >= 2 else np.nan
                g2_faa = np.mean(faa_means[1::2]) if len(faa_means) >= 2 else np.nan
            else:
                g1_faa, g2_faa = np.nan, np.nan
            
            condition_data.append({
                'Subject': subject, 'Condition_Group': group_name, 
                'Alpha_SNR': snr, 'FAA_Odd': g1_faa, 'FAA_Even': g2_faa
            })
            
            # 1-40 Hz PSD for F3 & F4 Grid Visualization
            if group_name in GRID_COLS and len(epochs_clean[valid_keys_clean]) > 0:
                n_fft_safe = int(epochs_clean.info['sfreq'])
                psd_1_40 = epochs_clean[valid_keys_clean].compute_psd(fmin=1, fmax=40, picks=['F3', 'F4'], method='welch', n_fft=n_fft_safe, n_jobs=N_JOBS, verbose=False)
                p_data_1_40, freqs_1_40 = psd_1_40.get_data(return_freqs=True)
                # p_data_1_40 shape: (epochs, channels (F3, F4), freqs)
                mean_p_data = np.mean(p_data_1_40, axis=0)
                grid_psd[subject][group_name] = {'F3': mean_p_data[0], 'F4': mean_p_data[1]}

        processed_subjects.append(subject)

    # ==============================================================================
    # Aggregation & DataFrame Creation
    # ==============================================================================
    df_cond = pd.DataFrame(condition_data)
    df_participant = pd.DataFrame(participant_summary)
    
    # Save Comprehensive Participant Export
    df_participant.to_csv(QC_DIR / 'Participant_Level_QC_Flags.csv', index=False)
    
    # Generate Condition Summary
    summary_list = []
    if not df_cond.empty:
        for cond, group in df_cond.groupby('Condition_Group'):
            mean_snr = group['Alpha_SNR'].mean()
            sem_snr = group['Alpha_SNR'].std() / np.sqrt(len(group))
            
            valid_split = group.dropna(subset=['FAA_Odd', 'FAA_Even'])
            r_val = pearsonr(valid_split['FAA_Odd'], valid_split['FAA_Even'])[0] if len(valid_split) > 2 else np.nan
            summary_list.append({'Condition': cond, 'Mean_SNR': mean_snr, 'SEM_SNR': sem_snr, 'Split_Half_r': r_val})
            
    df_summary = pd.DataFrame(summary_list).sort_values('Condition')
    
    # ==============================================================================
    # Visualizations
    # ==============================================================================
    
    # A) Epoch Survival Stacked Bar Chart
    fig_epochs, ax_epochs = plt.subplots(figsize=(14, 6))
    if not df_participant.empty:
        subs = df_participant['Subject']
        final = df_participant['Final_Valid_Epochs']
        iqr_drop = df_participant['IQR_Dropped']
        pre_drop = df_participant['Preproc_Dropped']
        
        ax_epochs.bar(subs, final, label='Final Usable Epochs', color='forestgreen')
        ax_epochs.bar(subs, iqr_drop, bottom=final, label='Dropped by IQR (Alpha Outlier)', color='gold')
        ax_epochs.bar(subs, pre_drop, bottom=final+iqr_drop, label='Dropped by Preproc (Voltage/Fluct)', color='crimson')
        
        ax_epochs.set_xticks(range(len(subs)))
        ax_epochs.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
        ax_epochs.set_ylabel("Number of Epochs")
        ax_epochs.set_title("Epoch Attrition Pipeline Tracker per Participant")
        ax_epochs.legend()
    fig_epochs.tight_layout()
    b64_epochs = fig_to_base64(fig_epochs)

    # B) SNR Overview
    fig_snr, ax_snr = plt.subplots(figsize=(12, 6))
    if not df_summary.empty:
        x_pos = np.arange(len(df_summary))
        ax_snr.bar(x_pos, df_summary['Mean_SNR'], yerr=df_summary['SEM_SNR'], capsize=5, color='teal', alpha=0.7)
        ax_snr.set_xticks(x_pos)
        ax_snr.set_xticklabels(df_summary['Condition'], rotation=45, ha="right", fontsize=10)
        ax_snr.set_ylabel("Alpha SNR")
        ax_snr.set_title("Mean Alpha SNR (+/- SEM) by Emotion Groupings")
    fig_snr.tight_layout()
    b64_snr = fig_to_base64(fig_snr)

    # C) Split-Half Overview
    fig_split, ax_split = plt.subplots(figsize=(12, 6))
    if not df_summary.empty:
        ax_split.bar(x_pos, df_summary['Split_Half_r'], color='coral', alpha=0.7)
        ax_split.axhline(0, color='black', linewidth=0.8)
        ax_split.set_xticks(x_pos)
        ax_split.set_xticklabels(df_summary['Condition'], rotation=45, ha="right", fontsize=10)
        ax_split.set_ylabel("Pearson r")
        ax_split.set_title("Split-Half Reliability (Odd vs Even) on IQR-Cleaned FAA Data")
    fig_split.tight_layout()
    b64_split = fig_to_base64(fig_split)

    # D) 6-14 Hz Periodogram
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
        ax2.set_xlabel("Frequency (Hz)")
        b64_perio = fig_to_base64(fig_perio)
    else: b64_perio = ""

    # E) 1-40 Hz F3 & F4 Grid
    if processed_subjects and freqs_1_40 is not None:
        fig_grid, axes_grid = plt.subplots(nrows=len(processed_subjects), ncols=len(GRID_COLS), 
                                           figsize=(15, 3 * len(processed_subjects)), sharex=True)
        if len(processed_subjects) == 1: axes_grid = np.expand_dims(axes_grid, axis=0) 

        for i, sub in enumerate(processed_subjects):
            for j, cond_col in enumerate(GRID_COLS):
                ax = axes_grid[i, j]
                data = grid_psd[sub].get(cond_col, {})
                if 'F3' in data and 'F4' in data:
                    # Plot F3 and F4 overlayed
                    ax.plot(freqs_1_40, data['F3'] * 1e12, color='dodgerblue', alpha=0.8, label='F3' if i==0 and j==0 else "")
                    ax.plot(freqs_1_40, data['F4'] * 1e12, color='darkorange', alpha=0.8, label='F4' if i==0 and j==0 else "")
                
                if i == 0: ax.set_title(cond_col)
                if j == 0: ax.set_ylabel(f"{sub}\nPower")
                if i == len(processed_subjects) - 1: ax.set_xlabel("Frequency (Hz)")
                
        fig_grid.legend(loc='upper right')
        fig_grid.tight_layout()
        b64_grid = fig_to_base64(fig_grid)
    else: b64_grid = ""

    # ==============================================================================
    # HTML Export
    # ==============================================================================
    # Convert bools for prettier HTML
    df_participant_html = df_participant.copy()
    for col in ['Tracker_Bad_Data', 'Master_Bad_Flag']:
        df_participant_html[col] = df_participant_html[col].apply(lambda x: '🚩 YES' if x else 'No')

    html_content = f"""
    <html>
    <head><title>Advanced Quality Check & Flag Report</title></head>
    <body style="font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9;">
        <h1 style="color: #2c3e50;">Comprehensive Quality Check & Flagging Report</h1>
        <p><strong>Batch:</strong> {BATCH_NAME} | <strong>FAA Method:</strong> {FAA_METHOD}</p>
        <hr>
        
        <h2 style="color: #c0392b;">1. Participant Master Flags & Epoch Attrition</h2>
        <p><strong>Master Bad Flag</strong> is triggered if a participant was flagged in the visual tracker, had a bad F3/F4 channel, or lost >50% of their total raw epochs.</p>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow-x: auto;">
            {df_participant_html.to_html(index=False, float_format="%.1f", border=1, classes='table')}
        </div>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;">
            <img src="data:image/png;base64,{b64_epochs}" style="max-width: 100%;">
        </div>
        
        <hr>
        <h2 style="color: #34495e;">2. Aggregate Overview: SNR & Split-Half Reliability</h2>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <img src="data:image/png;base64,{b64_snr}" style="max-width: 100%; margin-bottom: 20px;"><br>
            <img src="data:image/png;base64,{b64_split}" style="max-width: 100%;">
            <br><br>
            {df_summary.to_html(index=False, float_format="%.4f", border=1)}
        </div>
        
        <hr>
        <h2 style="color: #34495e;">3. Alpha Band Power (6-14 Hz) - Grand Average Across All Conditions</h2>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            {f'<img src="data:image/png;base64,{b64_perio}" style="max-width: 100%;">' if b64_perio else '<p>No 6-14Hz data generated.</p>'}
        </div>
        
        <hr>
        <h2 style="color: #34495e;">4. Broad Band Power (1-40 Hz) F3 & F4 - Per Participant, Per Group</h2>
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            {f'<img src="data:image/png;base64,{b64_grid}" style="max-width: 100%;">' if b64_grid else '<p>No frequency grid data generated.</p>'}
        </div>
        
    </body>
    </html>
    """
    
    with open(QC_DIR / 'Comprehensive_QC_Report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nSaved Grouped QC HTML & CSV Flags to: {QC_DIR}")

if __name__ == '__main__':
    run_comprehensive_quality_checks()
# %%