# %% #######################################################################################################################
# The goal of this script is to run supplementary analyses correlating political variables (from a post-exp survey) with FAA
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Cell 1: Extract FAA Data & Compute Participant-Level Condition Means
import os
import re
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"
FAA_METHOD = "diff"

# Toggles for processing pipeline
USE_Z_SCORES = False
# Baseline options: 'raw', 'party', 'politician'
BASELINE_TYPE = 'politician'  

PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids' / 'derivatives'
QC_DIR = DERIVATIVES_DIR / 'quality_check' / BATCH_NAME

# Survey Paths (Aligned with 1_survey_preprocessing and 2_survey_sample_description)
PROJECT_FOLDER_SURVEY = Path("DEFINE SURVEY FOLDER")
SURVEY_DIR = PROJECT_FOLDER_SURVEY / "survey" #remove subfolder if necessary (the replication package survey files are located in the Scripts folder)

FIGURES_DIR = DERIVATIVES_DIR / "figures" / "Batch_06_VirtualCz_8-12Hz" / "diff" / "FAA_survey"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_SUBJECTS = {"sub-P03"}  

# Auto-Exclude Flagged Participants
flag_file = QC_DIR / 'Flagged_Bad_F3_F4_Participants.csv'
if flag_file.exists():
    df_flags = pd.read_csv(flag_file)
    if not df_flags.empty and 'Subject' in df_flags.columns:
        EXCLUDE_SUBJECTS.update(df_flags['Subject'].unique())

# Event Dictionaries
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
ID_TO_EVENT = {v: k for k, v in EVENTS_STATEMENTSFACES.items()}
JF_EVENT_MAP = {201: ('PVV', 'Agema'), 202: ('PVV', 'Wilders'), 211: ('GL', 'Klaver'), 212: ('GL', 'Timmermans')}

print("=============================================================================================")
print(f"1. Extracting FAA Participant-Level Means (Baseline: {BASELINE_TYPE}, Z-Score: {USE_Z_SCORES})")
print("=============================================================================================")

eeg_processed_dir = DERIVATIVES_DIR / 'processed' / BATCH_NAME
subject_folders = sorted([d for d in eeg_processed_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])

all_trials = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    eeg_dir = subj_dir / 'eeg'
    sf_file = eeg_dir / f"{subject}_task-StatementsFaces_faa_{FAA_METHOD}-epo.fif" 
    jf_file = eeg_dir / f"{subject}_task-JustFaces_faa_{FAA_METHOD}-epo.fif"

    if sf_file.exists() and jf_file.exists():
        
        # JF Baselines 
        epochs_jf = mne.read_epochs(jf_file, preload=True, verbose=False)
        time_mask_jf = (epochs_jf.times >= 0.0) & (epochs_jf.times <= 1.2)
        mean_faa_jf_trials = np.mean(epochs_jf.get_data(picks='FAA')[:, 0, time_mask_jf], axis=-1)
        
        party_vals = {'GL': [], 'PVV': []}
        pol_vals = {'Klaver': [], 'Timmermans': [], 'Agema': [], 'Wilders': []}
        
        for idx, ev_id in enumerate(epochs_jf.events[:, 2]):
            if ev_id in JF_EVENT_MAP:
                party, pol = JF_EVENT_MAP[ev_id]
                val = mean_faa_jf_trials[idx]
                party_vals[party].append(val)
                pol_vals[pol].append(val)
                
        overall_mean = np.mean(mean_faa_jf_trials)
        party_means = {k: np.mean(v) if len(v) > 0 else overall_mean for k, v in party_vals.items()}
        pol_means = {k: np.mean(v) if len(v) > 0 else party_means.get('GL' if k in ['Klaver','Timmermans'] else 'PVV', overall_mean) for k, v in pol_vals.items()}

        # SF Extraction
        epochs = mne.read_epochs(sf_file, preload=True, verbose=False)
        time_mask = (epochs.times >= 0.0) & (epochs.times <= 1.2)
        mean_faa_per_trial = np.mean(epochs.get_data(picks='FAA')[:, 0, time_mask], axis=-1)
        
        for i, event_id in enumerate(epochs.events[:, 2]):
            event_name = ID_TO_EVENT.get(event_id)
            if not event_name: continue
                
            parts = event_name.split('/')
            party, pol, statement = parts[0], parts[1], parts[2]
            
            raw_val = mean_faa_per_trial[i]
            if BASELINE_TYPE == 'party':
                final_val = raw_val - party_means.get(party, overall_mean)
            elif BASELINE_TYPE == 'politician':
                final_val = raw_val - pol_means.get(pol, overall_mean)
            else:
                final_val = raw_val
            
            # Map Subject to canonical "P1" formatting to match survey data
            subject_num = int(subject.replace("sub-P", ""))
            canon_subject = f"P{subject_num}"
            
            all_trials.append({
                'Participant': canon_subject,
                'Condition': f"FAA_{party}_{statement}",
                'FAA_Value': final_val
            })

df_trials = pd.DataFrame(all_trials)

# Z-score standardisation toggle
if USE_Z_SCORES:
    df_trials['FAA_Value'] = (df_trials['FAA_Value'] - df_trials['FAA_Value'].mean()) / df_trials['FAA_Value'].std()

# Group by Participant and Condition to extract Participant-level condition means
df_faa_means = df_trials.groupby(['Participant', 'Condition'])['FAA_Value'].mean().reset_index()

# Pivot to wide format (1 row per participant)
df_faa_wide = df_faa_means.pivot(index='Participant', columns='Condition', values='FAA_Value').reset_index()

print(f"Extracted FAA means for {len(df_faa_wide)} valid participants.")
print(df_faa_wide.head())


# %% Cell 2: Extract Survey Data, Compute Metrics & Align
print("\n=============================================================================================")
print("2. Extracting Survey Variables & Aligning with Valid FAA Participants")
print("=============================================================================================")

# Locate Survey Files
def _resolve_pre_post_spss(sdir: Path):
    cands_post = sorted(sdir.glob("*post*experiment*.sav"), key=lambda p: p.stat().st_mtime, reverse=True)
    post_path = cands_post[0] if cands_post else None
    all_sav = sorted(sdir.glob("*.sav"), key=lambda p: p.stat().st_mtime, reverse=True)
    pre_path = next((p for p in all_sav if "post" not in p.name.lower()), None)
    return pre_path, post_path

pre_path, post_path = _resolve_pre_post_spss(SURVEY_DIR)
if not pre_path: raise FileNotFoundError(f"No PRE .sav found in {SURVEY_DIR}")

df_pre = pd.read_spss(pre_path)
df_post = pd.read_spss(post_path) if post_path else pd.DataFrame()

# Clean Participant IDs
def _extract_pid(raw):
    if pd.isna(raw): return None
    s = str(raw).strip().replace("\n", "").replace("\r", "")
    m = re.search(r'[_-]?[Pp]\s*([0-9]{1,3})\s*$', s)
    if m: return int(m.group(1))
    hits = re.findall(r'[Pp]\s*([0-9]{1,3})', s)
    return int(hits[-1]) if hits else None

def _clean_participants(df):
    if "P_ID" not in df.columns: return df.copy()
    tmp = df.copy()
    tmp["pid_num"] = tmp["P_ID"].apply(_extract_pid)
    tmp = tmp[~tmp["P_ID"].astype(str).str.contains(r"(?i)\btest\b", na=False) & tmp["pid_num"].notna()].copy()
    tmp["Participant"] = tmp["pid_num"].apply(lambda n: f"P{int(n)}")
    return tmp.drop(columns=["pid_num"])

df_pre_clean = _clean_participants(df_pre)
df_post_clean = _clean_participants(df_post)

# Extract PRE Variables
df_survey_vars = df_pre_clean[['Participant']].copy()

pis_map = {"nooit/zelden": 1, "soms": 2, "vaak": 3, "altijd": 4}
def _get_pis(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols: return pd.Series(np.nan, index=df.index)
    frames = []
    for c in cols:
        s_str = df[c].astype(str).str.strip().str.lower()
        frames.append(s_str.map(pis_map).fillna(pd.to_numeric(s_str, errors='coerce')).astype(float))
    return pd.concat(frames, axis=1).mean(axis=1)

df_survey_vars['avg_PIS_positive'] = _get_pis(df_pre_clean, "PIS_positive_")
df_survey_vars['avg_PIS_negative'] = _get_pis(df_pre_clean, "PIS_negative_")

def _clean_numeric(s, limit):
    return pd.to_numeric(s.astype(str).str.strip(), errors='coerce').clip(lower=0, upper=limit)

if 'AP_Party_PVV_1' in df_pre_clean.columns: df_survey_vars['valence_PVV'] = _clean_numeric(df_pre_clean['AP_Party_PVV_1'], 10)
if 'AP_Party_GL_1' in df_pre_clean.columns: df_survey_vars['valence_GL'] = _clean_numeric(df_pre_clean['AP_Party_GL_1'], 10)
if 'lr_placement_1' in df_pre_clean.columns: df_survey_vars['lr_Ideology'] = _clean_numeric(df_pre_clean['lr_placement_1'], 10)

# Extract POST Variables
post_map = {
    'AP_Wilders_1': 'valence_Wilders', 'AP_Agema_1': 'valence_Agema',
    'AP_Klaver_1': 'valence_Klaver', 'AP_Timmermans_1': 'valence_Timmermans'
}
if not df_post_clean.empty:
    for raw, clean in post_map.items():
        if raw in df_post_clean.columns:
            df_survey_vars = df_survey_vars.merge(
                df_post_clean[['Participant', raw]].rename(columns={raw: clean}),
                on='Participant', how='left'
            )
            df_survey_vars[clean] = _clean_numeric(df_survey_vars[clean], 10)

# Create Composite & Diff Variables
df_survey_vars['valence_Right_Politicians'] = df_survey_vars[['valence_Wilders', 'valence_Agema']].mean(axis=1)
df_survey_vars['valence_Left_Politicians'] = df_survey_vars[['valence_Klaver', 'valence_Timmermans']].mean(axis=1)

df_survey_vars['diff_Valence_GL_PVV'] = df_survey_vars['valence_GL'] - df_survey_vars['valence_PVV']
df_survey_vars['diff_Valence_Left_Right_Pols'] = df_survey_vars['valence_Left_Politicians'] - df_survey_vars['valence_Right_Politicians']
df_survey_vars['avg_PIS_overall'] = df_survey_vars[['avg_PIS_positive', 'avg_PIS_negative']].mean(axis=1)

# Inner Join with FAA Data to ensure strictly valid extant participants
df_master = pd.merge(df_faa_wide, df_survey_vars, on='Participant', how='inner')

print(f"Alignment complete. Master dataset contains {len(df_master)} participants with both valid FAA and survey data.")


# %% Cell 3: Correlation Matrices & Table Exports (FAA vs Survey Variables)
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

print("\n=============================================================================================")
print("3. Generating Cross-Correlations, Heatmaps & LaTeX/CSV Tables")
print("=============================================================================================")

# TOGGLES
SHOW_NS = False
USE_PERMUTATION = True  
N_PERMS = 1000          
APPLY_FDR = True        # Toggle for False Discovery Rate (FDR) correction

faa_cols = ['FAA_GL_left', 'FAA_GL_right', 'FAA_PVV_left', 'FAA_PVV_right']
survey_base_cols = [
    'avg_PIS_positive', 'avg_PIS_negative', 
    'valence_GL', 'valence_PVV', 
    'valence_Left_Politicians', 'valence_Right_Politicians',
    'lr_Ideology'
]
survey_diff_cols = [
    'diff_Valence_GL_PVV', 
    'diff_Valence_Left_Right_Pols', 
    'avg_PIS_overall'
]

def fmt_p_tex(p):
    if pd.isna(p): return ""
    if p < 0.001:
        base, exp = f"{p:.2e}".split('e')
        return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"
    return f"{p:.3f}"

def run_correlations_and_plot(df, cols_y, cols_x, title_base, filename_base, show_ns=False, use_permutation=False, n_perms=1000, apply_fdr=False):
    r_matrix = np.zeros((len(cols_y), len(cols_x)))
    p_matrix = np.zeros((len(cols_y), len(cols_x)))
    annot_matrix = []
    stats_list = []
    
    np.random.seed(42)

    for i, var_y in enumerate(cols_y):
        row_annot = []
        for j, var_x in enumerate(cols_x):
            valid_data = df[[var_x, var_y]].dropna()
            
            if len(valid_data) > 5 and valid_data.iloc[:, 0].std() > 0 and valid_data.iloc[:, 1].std() > 0:
                x = valid_data.iloc[:, 0].values
                y = valid_data.iloc[:, 1].values
                
                r_val, p_val = pearsonr(x, y)
                
                if use_permutation:
                    n_subs = len(x)
                    x_c = x - np.mean(x)
                    y_c = y - np.mean(y)
                    denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))
                    
                    null_r = np.zeros(n_perms)
                    for k in range(n_perms):
                        signs = np.random.choice([-1, 1], size=n_subs)
                        null_cov = np.sum(signs * x_c * y_c)
                        null_r[k] = null_cov / denom
                    p_val = np.sum(np.abs(null_r) >= np.abs(r_val)) / n_perms

                r_matrix[i, j] = r_val
                p_matrix[i, j] = p_val
                
                stats_list.append({'Survey_Variable': var_y, 'FAA_Condition': var_x, 'r': r_val, 'p_raw': p_val})
            else:
                r_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan
                stats_list.append({'Survey_Variable': var_y, 'FAA_Condition': var_x, 'r': np.nan, 'p_raw': np.nan})
                
    # Process Stats Table
    df_stats = pd.DataFrame(stats_list).dropna()
    
    if apply_fdr:
        _, p_corr, _, _ = multipletests(df_stats['p_raw'], method='fdr_bh')
        df_stats['p_corr'] = p_corr
        eval_p = 'p_corr'
    else:
        eval_p = 'p_raw'

    df_stats = df_stats.sort_values(by=eval_p, ascending=True).reset_index(drop=True)
    
    # Assign significance based on the evaluated p-value (raw or corrected)
    def get_stars(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        return ''
    
    df_stats['sig'] = df_stats[eval_p].apply(get_stars)
    
    # Re-build annotation matrix for heatmap using standard p-value
    for i, var_y in enumerate(cols_y):
        row_annot = []
        for j, var_x in enumerate(cols_x):
            r_val = r_matrix[i, j]
            p_val = p_matrix[i, j]
            if not np.isnan(p_val):
                stars = get_stars(p_val) if not show_ns else (get_stars(p_val) if p_val < 0.05 else "ns")
                row_annot.append(f"{r_val:.2f}\n{stars}")
            else:
                row_annot.append("")
        annot_matrix.append(row_annot)

    # Plotting
    sig_mask = np.isnan(p_matrix) if show_ns else ((p_matrix >= 0.05) | np.isnan(p_matrix))
    final_title = f"{'All' if show_ns else 'Significant'} Cross-Correlations\n{title_base}"
    if use_permutation: final_title += f"\n(Permutation p-values, {n_perms} iters)"

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="white", context="paper")
    sns.heatmap(
        pd.DataFrame(r_matrix, index=cols_y, columns=cols_x), 
        mask=sig_mask, annot=np.array(annot_matrix), fmt="", cmap='RdBu_r', 
        center=0, vmin=-1, vmax=1, square=True, linewidths=.5, 
        cbar_kws={"shrink": .8, "label": "Pearson's r"},
        annot_kws={"size": 10, "fontweight": "bold"}
    )
    plt.title(final_title, fontsize=16, pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename_base}.jpg", dpi=300, format='jpg', bbox_inches='tight', facecolor='white')
    plt.show()

    # Saving Tables 
    csv_out = df_stats.copy()
    csv_out['r'] = csv_out['r'].apply(lambda x: f"{x:.3f}")
    csv_out['p_raw'] = csv_out['p_raw'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    if apply_fdr:
        csv_out['p_corr'] = csv_out['p_corr'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    
    csv_out.to_csv(FIGURES_DIR / f"{filename_base}_stats.csv", index=False)

# LaTeX Table
    tex_rows = ""
    for _, row in df_stats.iterrows():
        r_val = f"{row['r']:.3f}{row['sig']}"
        p_raw_tex = fmt_p_tex(row['p_raw'])
        
        # FIX: Define escaped strings outside of the f-string expressions
        survey_var_escaped = str(row['Survey_Variable']).replace('_', '\\_')
        faa_cond_escaped = str(row['FAA_Condition']).replace('_', '\\_')
        
        if apply_fdr:
            p_corr_tex = fmt_p_tex(row['p_corr'])
            tex_rows += f"{survey_var_escaped} & {faa_cond_escaped} & {r_val} & {p_raw_tex} & {p_corr_tex} \\\\\n"
        else:
            tex_rows += f"{survey_var_escaped} & {faa_cond_escaped} & {r_val} & {p_raw_tex} \\\\\n"

    table_cols = "ll S[table-format=1.3] S[table-format=1.2e-2] S[table-format=1.2e-2]" if apply_fdr else "ll S[table-format=1.3] S[table-format=1.2e-2]"
    header = "\\textbf{Survey Variable} & \\textbf{FAA Condition} & \\textbf{r} & \\textbf{$p_\\text{raw}$}"
    if apply_fdr: header += " & \\textbf{$p_\\text{FDR}$}"

    # Move title translation outside the f-string here too
    title_base_escaped = title_base.replace('_', '\\_')

    tex_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{Survey Correlations ({title_base_escaped})}}\n"
        f"\\label{{tab:corr_{filename_base}}}\n"
        "\\begin{threeparttable}\n"
        f"\\begin{{tabular}}{{@{{}}{table_cols}@{{}}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{tex_rows}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n"
        f"\\item \\textit{{Note.}} * $p < .05$, ** $p < .01$, *** $p < .001$. $p$-values calculated via {n_perms} permutations. Sorted from lowest to highest significance.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}"
    )
    with open(FIGURES_DIR / f"{filename_base}_stats.tex", 'w') as f:
        f.write(tex_table)

# Run for Base Survey Variables
run_correlations_and_plot(
    df=df_master, cols_y=survey_base_cols, cols_x=faa_cols,
    title_base="FAA vs Partisanship Metrics", filename_base=f"corr_base_survey_{BASELINE_TYPE}",
    show_ns=SHOW_NS, use_permutation=USE_PERMUTATION, n_perms=N_PERMS, apply_fdr=APPLY_FDR
)

# Run for Diff Survey Variables
run_correlations_and_plot(
    df=df_master, cols_y=survey_diff_cols, cols_x=faa_cols,
    title_base="FAA vs Aggregated Partisanship Metrics", filename_base=f"corr_diff_survey_{BASELINE_TYPE}",
    show_ns=SHOW_NS, use_permutation=USE_PERMUTATION, n_perms=N_PERMS, apply_fdr=APPLY_FDR
)

# %% Cell 5: Post-Hoc - PIS Negative vs. Aggregated FAA Conditions (with Permutation)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

print("\n=============================================================================================")
print("5. Post-Hoc: PIS Negative vs. Aggregated FAA Conditions (w/ Sign-Flipping)")
print("=============================================================================================")

# 1. Create Aggregated Variables in df_master
# Analysis 1: Right-Wing Statements (GL/right + PVV/right)
df_master['FAA_Right_Statements'] = df_master[['FAA_GL_right', 'FAA_PVV_right']].mean(axis=1)

# Analysis 2: Incongruent Statements (PVV/left + GL/right)
df_master['FAA_Incongruent'] = df_master[['FAA_PVV_left', 'FAA_GL_right']].mean(axis=1)

# Analysis 3: All Statements (GL/left + GL/right + PVV/left + PVV/right)
df_master['FAA_All_Statements'] = df_master[['FAA_GL_left', 'FAA_GL_right', 'FAA_PVV_left', 'FAA_PVV_right']].mean(axis=1)

# 2. Define Reusable Analysis & Plotting Function
def run_permuted_scatter(df, x_col, y_col, x_label, y_label, title, filename):
    print(f"\n--- Running Analysis: {title} ---")
    
    # Extract valid paired data
    df_clean = df[[x_col, y_col]].dropna()
    x = df_clean[x_col].values
    y = df_clean[y_col].values
    n_subs = len(df_clean)
    
    print(f"Valid participants: N = {n_subs}")
    
    if n_subs < 5:
        print("Not enough data to run analysis.")
        return

    # Standard Pearson Correlation
    obs_r, obs_p = pearsonr(x, y)

    # Sign-Flipping Permutation Test (1000 iterations)
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))

    n_perms = 1000
    null_r = np.zeros(n_perms)

    np.random.seed(42) # Seed for reproducibility
    for i in range(n_perms):
        # Generate random signs (-1 or 1) for each participant
        signs = np.random.choice([-1, 1], size=n_subs)
        # Flip the sign of x_c (equivalent to flipping the paired product)
        null_cov = np.sum(signs * x_c * y_c)
        null_r[i] = null_cov / denom

    # Calculate two-tailed permutation p-value
    perm_p = np.sum(np.abs(null_r) >= np.abs(obs_r)) / n_perms

    print(f"Observed Pearson r : {obs_r:.3f}")
    print(f"Parametric p-value : {obs_p:.4f}")
    print(f"Permutation p-value: {perm_p:.4f} (1000)")

    # Visualization: Scatterplot with Regression Line
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", context="talk")

    # Plot using the specified distinct color palette
    ax = sns.regplot(
        x=x, y=y, 
        scatter_kws={'alpha': 0.7, 's': 80, 'color': '#8e44ad', 'edgecolor': 'white'}, 
        line_kws={'color': '#2c3e50', 'linewidth': 2, 'linestyle': '--'}
    )

    plt.title(title, pad=20, fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Annotate stats on the plot
    stats_text = (
        f"N = {n_subs}\n"
        f"Pearson r = {obs_r:.3f}\n"
        f"Parametric p = {obs_p:.3f}\n"
        f"Permutation p = {perm_p:.3f}"
    )

    plt.annotate(
        stats_text, 
        xy=(0.05, 0.95), xycoords='axes fraction',
        ha='left', va='top',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#cccccc", alpha=0.9)
    )

    sns.despine()
    plt.tight_layout()

    # Save and Show
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight')
    print(f"Scatterplot saved to: {save_path}")

    plt.show()

# 3. Execute the 3 Sequential Analyses
# Constructing descriptive strings based on BASELINE_TYPE dynamically
baseline_label = "Raw" if BASELINE_TYPE == "raw" else f"{BASELINE_TYPE.capitalize()}-baselined"
y_axis_label = f"FAA ({baseline_label})"

# --- Analysis 1: Right-Wing Statements ---
run_permuted_scatter(
    df=df_master,
    x_col='avg_PIS_negative',
    y_col='FAA_Right_Statements',
    x_label="Average PIS Negative (1-4)",
    y_label=y_axis_label,
    title=f"Faces & Right-Wing Statements ({baseline_label}) vs. PIS Negative",
    filename=f"scatter_PIS_neg_vs_Right_Statements_{BASELINE_TYPE}.jpg"
)

# --- Analysis 2: Incongruent Statements ---
run_permuted_scatter(
    df=df_master,
    x_col='avg_PIS_negative',
    y_col='FAA_Incongruent',
    x_label="Average PIS Negative (1-4)",
    y_label=y_axis_label,
    title=f"Incongruent Faces & Statements ({baseline_label}) vs. PIS Negative",
    filename=f"scatter_PIS_neg_vs_Incongruent_Statements_{BASELINE_TYPE}.jpg"
)

# --- Analysis 3: All Statements ---
run_permuted_scatter(
    df=df_master,
    x_col='avg_PIS_negative',
    y_col='FAA_All_Statements',
    x_label="Average PIS Negative (1-4)",
    y_label=y_axis_label,
    title=f"All Faces & Statements ({baseline_label}) vs. PIS Negative",
    filename=f"scatter_PIS_neg_vs_All_Statements_{BASELINE_TYPE}.jpg"
)

# %% Cell 6: PIS Negative Split and FAA Condition Means
print("\n=============================================================================================")
print("6. Grouping by Negative PIS & Visualising FAA Conditions")
print("=============================================================================================")

# Choose from: 'median' (50/50), 'quartile' (25/25), or 'sd' (+/- 1 SD)
SPLIT_METHOD = 'median' 

# 1. Determine Thresholds and Assign Groups
pis_col = 'avg_PIS_negative'
df_plot = df_master.copy()

if SPLIT_METHOD == 'median':
    thresh = df_plot[pis_col].median()
    df_plot['PIS_Neg_Group'] = np.where(df_plot[pis_col] > thresh, 'High Neg PIS', 'Low Neg PIS')
    print(f"Split Method: Median (Threshold: {thresh:.2f})")

elif SPLIT_METHOD == 'quartile':
    q1 = df_plot[pis_col].quantile(0.25)
    q3 = df_plot[pis_col].quantile(0.75)
    conditions = [
        (df_plot[pis_col] >= q3),
        (df_plot[pis_col] <= q1)
    ]
    choices = ['High Neg PIS', 'Low Neg PIS']
    df_plot['PIS_Neg_Group'] = np.select(conditions, choices, default=np.nan)
    print(f"Split Method: Quartiles (Low <= {q1:.2f}, High >= {q3:.2f})")
    
elif SPLIT_METHOD == 'sd':
    mean_val = df_plot[pis_col].mean()
    sd_val = df_plot[pis_col].std()
    high_thresh = mean_val + sd_val
    low_thresh = mean_val - sd_val
    conditions = [
        (df_plot[pis_col] >= high_thresh),
        (df_plot[pis_col] <= low_thresh)
    ]
    choices = ['High Neg PIS', 'Low Neg PIS']
    df_plot['PIS_Neg_Group'] = np.select(conditions, choices, default=np.nan)
    print(f"Split Method: +/- 1 SD (Low <= {low_thresh:.2f}, High >= {high_thresh:.2f})")
else:
    raise ValueError("Invalid SPLIT_METHOD. Choose 'median', 'quartile', or 'sd'.")

# Drop intermediate/NaN participants based on split
df_plot = df_plot.dropna(subset=['PIS_Neg_Group'])
print(f"Valid participants for plot: {len(df_plot)} "
      f"(Low: {sum(df_plot['PIS_Neg_Group'] == 'Low Neg PIS')}, "
      f"High: {sum(df_plot['PIS_Neg_Group'] == 'High Neg PIS')})")

# 2. Reshape Data for Plotting
faa_target_cols = ['FAA_PVV_right', 'FAA_PVV_left', 'FAA_GL_right', 'FAA_GL_left']

df_melt = df_plot.melt(
    id_vars=['Participant', 'PIS_Neg_Group'],
    value_vars=faa_target_cols,
    var_name='Condition_Raw',
    value_name='FAA_Value'
).dropna(subset=['FAA_Value'])

# 3. Map to clean condition names and strictly define X-axis order
condition_map = {
    'FAA_PVV_right': 'Congruent\nout-party',
    'FAA_PVV_left': 'Incongruent\nout-party',
    'FAA_GL_right': 'Incongruent\nin-party',
    'FAA_GL_left': 'Congruent\nin-party'
}

df_melt['Condition'] = df_melt['Condition_Raw'].map(condition_map)
x_order = [condition_map[col] for col in faa_target_cols]

# 4. Plotting
plt.figure(figsize=(11, 7))
sns.set_theme(style="whitegrid", context="talk")

# Pointplot handles Means + SE calculation. 
# errorbar='se' forces Standard Error, capsize=0 removes whiskers, linestyle='none' removes connecting lines.
ax = sns.pointplot(
    data=df_melt,
    x='Condition',
    y='FAA_Value',
    hue='PIS_Neg_Group',
    order=x_order,
    hue_order=['Low Neg PIS', 'High Neg PIS'],
    dodge=0.3,
    errorbar='se', 
    capsize=0,
    err_kws={'linewidth': 2.5},
    markers='o',
    linestyle='none', 
    palette=['#34495e','#8e44ad']  # Purple & Dark Slate for slight contrast
)

plt.title(f"FAA Condition Means by Negative PIS ({SPLIT_METHOD.capitalize()} Split)", pad=20, fontweight='bold')
plt.xlabel("") 
plt.ylabel("Frontal Alpha Asymmetry (FAA)")

# Adjust legend
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Negative PIS", loc='best', frameon=True)

sns.despine()
plt.tight_layout()

# Save and Show
filename = f"pointplot_FAA_by_PIS_neg_{SPLIT_METHOD}_{BASELINE_TYPE}.jpg"
save_path = FIGURES_DIR / filename
plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight', facecolor='white')
print(f"\nPlot saved to: {save_path}\n")

plt.show()
# %% Cell 7: Pairwise Paired t-tests for FAA Conditions by PIS Group
import itertools
from scipy.stats import ttest_rel
import pandas as pd

print("\n=============================================================================================")
print("7. Pairwise Paired t-tests for FAA Conditions by Negative PIS Group")
print("=============================================================================================")

# Ensure we have the df_plot from the previous cell with the groups defined
if 'PIS_Neg_Group' not in df_plot.columns:
    print("Error: 'PIS_Neg_Group' not found in df_plot. Please run Cell 6 first.")
else:
    # Define the conditions and their clean labels (matching Cell 6)
    faa_target_cols = ['FAA_PVV_right', 'FAA_PVV_left', 'FAA_GL_right', 'FAA_GL_left']
    
    # Single-line labels for cleaner console output
    condition_map = {
        'FAA_PVV_right': 'Congruent out-party',
        'FAA_PVV_left': 'Incongruent out-party',
        'FAA_GL_right': 'Incongruent in-party',
        'FAA_GL_left': 'Congruent in-party'
    }
    
    # Generate the 6 unique pairs using combinations
    pairs = list(itertools.combinations(faa_target_cols, 2))
    
    # Process both groups
    groups = ['Low Neg PIS', 'High Neg PIS']
    
    for group in groups:
        print(f"\n--- Group: {group} ---")
        
        # Filter the master plotting dataframe for the current group
        df_group = df_plot[df_plot['PIS_Neg_Group'] == group]
        print(f"Total available participants in group: {len(df_group)}")
        
        # Print table header
        print(f"{'Condition 1':<25} | {'Condition 2':<25} | {'N':<4} | {'t-stat':<8} | {'p-value':<8} | {'Sig'}")
        print("-" * 87)
        
        for col1, col2 in pairs:
            # Drop NaNs for this specific pair to ensure a valid paired test
            valid_data = df_group[[col1, col2]].dropna()
            n_valid = len(valid_data)
            
            name1 = condition_map[col1]
            name2 = condition_map[col2]
            
            if n_valid > 2:  # Need minimum data points to run a t-test
                stat, p_val = ttest_rel(valid_data[col1], valid_data[col2])
                
                # Assign significance markers
                if p_val < 0.001: stars = "***"
                elif p_val < 0.01: stars = "**"
                elif p_val < 0.05: stars = "*"
                else: stars = "n.s."
                
                print(f"{name1:<25} | {name2:<25} | {n_valid:<4} | {stat:>8.3f} | {p_val:>8.4f} | {stars}")
            else:
                print(f"{name1:<25} | {name2:<25} | {n_valid:<4} | {'NaN':>8} | {'NaN':>8} | Not enough data")
                
        print("-" * 87)

# %% Cell 8: JustFaces (JF) FAA - Correlations, t-tests, and Visualisations
import itertools
from scipy.stats import pearsonr, ttest_rel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("\n=============================================================================================")
print("8. JustFaces (JF) FAA: Correlations, Median Splits, and Visualisations")
print("=============================================================================================")

# ------------------------------------------------------------------------------
# 1. Extract un-baselined JF means per participant
# ------------------------------------------------------------------------------
print("Extracting participant-level un-baselined JF means...")
jf_data = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue
    
    jf_file = subj_dir / 'eeg' / f"{subject}_task-JustFaces_faa_{FAA_METHOD}-epo.fif"
    
    if jf_file.exists():
        epochs_jf = mne.read_epochs(jf_file, preload=True, verbose=False)
        time_mask_jf = (epochs_jf.times >= 0.0) & (epochs_jf.times <= 1.2)
        mean_faa_jf_trials = np.mean(epochs_jf.get_data(picks='FAA')[:, 0, time_mask_jf], axis=-1)
        
        party_vals = {'GL': [], 'PVV': []}
        for idx, ev_id in enumerate(epochs_jf.events[:, 2]):
            if ev_id in JF_EVENT_MAP:
                party, _ = JF_EVENT_MAP[ev_id]
                party_vals[party].append(mean_faa_jf_trials[idx])
        
        # Map Subject to canonical "P1" formatting
        subject_num = int(subject.replace("sub-P", ""))
        canon_subject = f"P{subject_num}"
        
        jf_data.append({
            'Participant': canon_subject,
            'FAA_JF_PVV': np.mean(party_vals['PVV']) if party_vals['PVV'] else np.nan,
            'FAA_JF_GL': np.mean(party_vals['GL']) if party_vals['GL'] else np.nan
        })

df_jf = pd.DataFrame(jf_data)

# Merge with survey data from df_master
df_jf_master = pd.merge(df_jf, df_master, on='Participant', how='inner')
print(f"Alignment complete. N = {len(df_jf_master)} valid participants for JF analyses.\n")

# ------------------------------------------------------------------------------
# 2. Correlation Matrices (Permutation Tests)
# ------------------------------------------------------------------------------
jf_cols = ['FAA_JF_GL', 'FAA_JF_PVV']

print("--- Running Cross-Correlations: JF Conditions vs. Survey Metrics ---")

# Plot 1: Base Survey Variables
run_correlations_and_plot(
    df=df_jf_master,
    cols_y=survey_base_cols,
    cols_x=jf_cols,
    title_base="JustFaces (JF) FAA vs. Base Survey Metrics",
    filename_base="heatmap_jf_base_survey",
    show_ns=SHOW_NS,
    use_permutation=USE_PERMUTATION,
    n_perms=N_PERMS
)

# Plot 2: Diff Survey Variables
run_correlations_and_plot(
    df=df_jf_master,
    cols_y=survey_diff_cols,
    cols_x=jf_cols,
    title_base="JustFaces (JF) FAA vs. In/Out Party Differences",
    filename_base="heatmap_jf_diff_survey",
    show_ns=SHOW_NS,
    use_permutation=USE_PERMUTATION,
    n_perms=N_PERMS
)

# ------------------------------------------------------------------------------
# 3. Median Split on Negative PIS & Pairwise T-Tests
# ------------------------------------------------------------------------------
print("\n--- Running Pairwise t-tests (JF GL vs JF PVV) by Negative PIS Group ---")

# Define Split
thresh = df_jf_master['avg_PIS_negative'].median()
df_jf_master['PIS_Neg_Group'] = np.where(df_jf_master['avg_PIS_negative'] > thresh, 'High Neg PIS', 'Low Neg PIS')

print(f"Split Method: Median (Threshold: {thresh:.2f})")

groups = ['Low Neg PIS', 'High Neg PIS']

for group in groups:
    print(f"\n--- Group: {group} ---")
    df_group = df_jf_master[df_jf_master['PIS_Neg_Group'] == group]
    
    print(f"{'Condition 1':<20} | {'Condition 2':<20} | {'N':<4} | {'t-stat':<8} | {'p-value':<8} | {'Sig'}")
    print("-" * 75)
    
    valid_data = df_group[['FAA_JF_GL', 'FAA_JF_PVV']].dropna()
    n_valid = len(valid_data)
    
    if n_valid > 2:
        stat, p_val = ttest_rel(valid_data['FAA_JF_GL'], valid_data['FAA_JF_PVV'])
        
        if p_val < 0.001: stars = "***"
        elif p_val < 0.01: stars = "**"
        elif p_val < 0.05: stars = "*"
        else: stars = "n.s."
        
        print(f"{'In-party (GL)':<20} | {'Out-party (PVV)':<20} | {n_valid:<4} | {stat:>8.3f} | {p_val:>8.4f} | {stars}")
    else:
        print(f"{'In-party (GL)':<20} | {'Out-party (PVV)':<20} | {n_valid:<4} | {'NaN':>8} | {'NaN':>8} | Not enough data")
    print("-" * 75)

# ------------------------------------------------------------------------------
# 4. Visualise JF FAA by Negative PIS Split (Pointplot)
# ------------------------------------------------------------------------------
# Reshape for plotting
df_melt_jf = df_jf_master.melt(
    id_vars=['Participant', 'PIS_Neg_Group'],
    value_vars=['FAA_JF_PVV', 'FAA_JF_GL'],
    var_name='Condition_Raw',
    value_name='FAA_Value'
).dropna(subset=['FAA_Value'])

# Map to clean condition names and strictly define X-axis order
condition_map_jf = {
    'FAA_JF_PVV': 'Out-party',
    'FAA_JF_GL': 'In-party'
}
df_melt_jf['Condition'] = df_melt_jf['Condition_Raw'].map(condition_map_jf)
x_order_jf = [condition_map_jf['FAA_JF_PVV'], condition_map_jf['FAA_JF_GL']]

plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid", context="talk")

ax = sns.pointplot(
    data=df_melt_jf,
    x='Condition',
    y='FAA_Value',
    hue='PIS_Neg_Group',
    order=x_order_jf,
    hue_order=['Low Neg PIS', 'High Neg PIS'],
    dodge=0.2, 
    errorbar='se', 
    capsize=0,
    err_kws={'linewidth': 2.5},
    markers='o',
    linestyle='none', 
    palette=['#34495e','#8e44ad']
)

plt.title("JustFaces (JF) FAA Means by Negative PIS (Median Split)", pad=20, fontweight='bold')
plt.xlabel("")
plt.ylabel("Raw Frontal Alpha Asymmetry (FAA)")

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Negative PIS", loc='lower right', frameon=True,
    fontsize='small',         # Makes the category labels smaller
    title_fontsize='small',   # Makes the "Negative PIS" title smaller
    labelspacing=0.3,         # Reduces vertical space between rows
    handletextpad=0.4         # Reduces horizontal space between marker and text            
           )

sns.despine()
plt.tight_layout()

# Save 
save_path = FIGURES_DIR / "pointplot_FAA_JF_by_PIS_neg_median.jpg"
plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight', facecolor='white')

plt.show()
# %% Cell 9: Post-Hoc - PIS Negative vs. Aggregated JF FAA (with Permutation)
print("\n=============================================================================================")
print("9. Post-Hoc: PIS Negative vs. Aggregated JustFaces (JF) FAA (w/ Sign-Flipping)")
print("=============================================================================================")

# 1. Create Aggregated JF Variable in df_jf_master
# Averages the raw FAA for GL and PVV faces together for each participant
df_jf_master['FAA_JF_All'] = df_jf_master[['FAA_JF_GL', 'FAA_JF_PVV']].mean(axis=1)

# 2. Run Permuted Scatterplot
# Note: This relies on the `run_permuted_scatter` function defined in Cell 5. 
# Ensure Cell 5 has been run in your kernel before executing this cell.

try:
    run_permuted_scatter(
        df=df_jf_master,
        x_col='avg_PIS_negative',
        y_col='FAA_JF_All',
        x_label="Average PIS Negative (1-4)",
        y_label="Raw FAA (All JF Faces)",
        title="JustFaces FAA (All Faces) vs. PIS Negative",
        filename="scatter_PIS_neg_vs_JF_All.jpg"
    )
except NameError:
    print("Error: 'run_permuted_scatter' is not defined. Please run Cell 5 first so the function is in memory.")

# %% Cell 10: Permutation Independent Samples t-test (High vs. Low Neg PIS on Aggregated JF)
print("\n=============================================================================================")
print("10. Permutation Independent Samples t-test: High vs. Low Neg PIS on Aggregated JF FAA")
print("=============================================================================================")

# 1. Define groups based on existing split from Cell 8
group_high = df_jf_master[df_jf_master['PIS_Neg_Group'] == 'High Neg PIS']['FAA_JF_All'].dropna().values
group_low = df_jf_master[df_jf_master['PIS_Neg_Group'] == 'Low Neg PIS']['FAA_JF_All'].dropna().values

n_high = len(group_high)
n_low = len(group_low)
obs_diff = np.mean(group_high) - np.mean(group_low)

print(f"High Neg PIS (n={n_high}): Mean={np.mean(group_high):.4f}")
print(f"Low Neg PIS  (n={n_low}): Mean={np.mean(group_low):.4f}")
print(f"Observed Difference: {obs_diff:.4f}")

# 2. Permutation Test (Difference of Means)
# We combine data and shuffle group labels
combined_data = np.concatenate([group_high, group_low])
n_perms = 1000
perm_diffs = np.zeros(n_perms)

np.random.seed(42)
for i in range(n_perms):
    np.random.shuffle(combined_data)
    # Re-split into two groups of original sizes
    perm_high = combined_data[:n_high]
    perm_low = combined_data[n_high:]
    perm_diffs[i] = np.mean(perm_high) - np.mean(perm_low)

# Two-tailed p-value
perm_p = np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) / n_perms

print(f"\nPermutation p-value ({n_perms} iterations): {perm_p:.4f}")

# %% Cell 11: Final Aggregated Task T-tests (Between and Within PIS Groups)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import itertools

print("\n=============================================================================================")
print("11. Permutation t-tests & Visualisations: SF Conditions (Between & Within Groups)")
print("=============================================================================================")

# 1. Prepare the Master Data with Overall Aggregates
df_end = df_jf_master.copy()

# Ensure we have JF_All, SF_All, and the Differences
df_end['JF_All'] = df_end[['FAA_JF_GL', 'FAA_JF_PVV']].mean(axis=1)
df_end['SF_All'] = df_end[['FAA_GL_left', 'FAA_GL_right', 'FAA_PVV_left', 'FAA_PVV_right']].mean(axis=1)
df_end['SF_minus_JF'] = df_end['SF_All'] - df_end['JF_All']

# Difference scores per party (kept for reference)
df_end['SF_GL'] = df_end[['FAA_GL_left', 'FAA_GL_right']].mean(axis=1)
df_end['SF_PVV'] = df_end[['FAA_PVV_left', 'FAA_PVV_right']].mean(axis=1)
df_end['Diff_GL'] = df_end['SF_GL'] - df_end['FAA_JF_GL']
df_end['Diff_PVV'] = df_end['SF_PVV'] - df_end['FAA_JF_PVV']

# Refresh Median Split to ensure strict categorization
thresh = df_end['avg_PIS_negative'].median()
df_end['PIS_Neg_Group'] = np.where(df_end['avg_PIS_negative'] > thresh, 'High Neg PIS', 'Low Neg PIS')

N_PERMS = 1000

# -------------------------------------------------------------------------
# A. Visualise SF FAA by Negative PIS Split (Pointplot)
# -------------------------------------------------------------------------
print("--- Generating SF Conditions Pointplot ---")
sf_cols = ['FAA_GL_left', 'FAA_GL_right', 'FAA_PVV_left', 'FAA_PVV_right']

df_melt_sf = df_end.melt(
    id_vars=['Participant', 'PIS_Neg_Group'],
    value_vars=sf_cols,
    var_name='Condition_Raw',
    value_name='FAA_Value'
).dropna(subset=['FAA_Value'])

condition_map_sf = {
    'FAA_GL_left': 'In-party Face\nCongruent',
    'FAA_GL_right': 'In-party Face\nIncongruent',
    'FAA_PVV_left': 'Out-party Face\nIncongruent',
    'FAA_PVV_right': 'Out-party Face\nCongruent'
}
df_melt_sf['Condition'] = df_melt_sf['Condition_Raw'].map(condition_map_sf)
x_order_sf = [condition_map_sf[col] for col in sf_cols]

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid", context="talk")

ax = sns.pointplot(
    data=df_melt_sf,
    x='Condition',
    y='FAA_Value',
    hue='PIS_Neg_Group',
    order=x_order_sf,
    hue_order=['Low Neg PIS', 'High Neg PIS'],
    dodge=0.2, 
    errorbar='se', 
    capsize=0,
    err_kws={'linewidth': 2.5},
    markers='o',
    linestyle='none', 
    palette=['#34495e','#8e44ad']
)

plt.title("StatementsFaces (SF) FAA Means by Negative PIS (Median Split)", pad=20, fontweight='bold')
plt.xlabel("")
plt.ylabel("Frontal Alpha Asymmetry (FAA)")

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Negative PIS", loc='best', frameon=True,
    fontsize='small', title_fontsize='small', labelspacing=0.3, handletextpad=0.4)

sns.despine()
plt.tight_layout()

save_path = FIGURES_DIR / "pointplot_FAA_SF_by_PIS_neg_median.jpg"
plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight', facecolor='white')
plt.show()

# -------------------------------------------------------------------------
# B. Independent T-Tests: High vs Low PIS (Between-Subjects)
# -------------------------------------------------------------------------
def run_indep_perm_ttest(df, target_col):
    grp_high = df[df['PIS_Neg_Group'] == 'High Neg PIS'][target_col].dropna().values
    grp_low = df[df['PIS_Neg_Group'] == 'Low Neg PIS'][target_col].dropna().values
    
    obs_diff = np.mean(grp_high) - np.mean(grp_low)
    pooled_sd = np.sqrt(((len(grp_high)-1)*np.var(grp_high) + (len(grp_low)-1)*np.var(grp_low)) / (len(grp_high) + len(grp_low) - 2))
    d = obs_diff / pooled_sd if pooled_sd != 0 else np.nan
    
    combined = np.concatenate([grp_high, grp_low])
    n_high = len(grp_high)
    
    perm_diffs = np.zeros(N_PERMS)
    np.random.seed(42)
    for i in range(N_PERMS):
        np.random.shuffle(combined)
        perm_diffs[i] = np.mean(combined[:n_high]) - np.mean(combined[n_high:])
        
    p_val = np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) / N_PERMS
    return obs_diff, d, p_val

between_results = []
metrics = ['JF_All', 'SF_All', 'SF_minus_JF']
labels = ['JF Only (All)', 'SF Only (All)', 'SF - JF (Difference)']

for metric, label in zip(metrics, labels):
    diff, d, p = run_indep_perm_ttest(df_end, metric)
    between_results.append({'Metric': label, 'Delta (High - Low)': diff, 'd': d, 'p_raw': p})

df_between = pd.DataFrame(between_results)

# -------------------------------------------------------------------------
# C. Paired T-Tests: Within-Subject Condition Contrasts (For Both Groups)
# -------------------------------------------------------------------------
def run_paired_perm_ttest(df, col_A, col_B):
    valid = df[[col_A, col_B]].dropna()
    diffs = valid[col_A].values - valid[col_B].values
    n = len(diffs)
    
    if n < 3: return np.nan, np.nan, np.nan
    
    obs_mean = np.mean(diffs)
    d = obs_mean / np.std(diffs, ddof=1)
    
    perm_means = np.zeros(N_PERMS)
    np.random.seed(42)
    for i in range(N_PERMS):
        signs = np.random.choice([-1, 1], size=n)
        perm_means[i] = np.mean(diffs * signs)
        
    p_val = np.sum(np.abs(perm_means) >= np.abs(obs_mean)) / N_PERMS
    return obs_mean, d, p_val

# Define the contrasts: 1 JF pair + 6 pairwise combinations of the 4 SF conditions
paired_contrasts = [
    ('FAA_JF_GL', 'FAA_JF_PVV', 'JF: GL vs PVV'),
    ('FAA_GL_left', 'FAA_GL_right', 'SF: GL Left vs GL Right'),
    ('FAA_GL_left', 'FAA_PVV_left', 'SF: GL Left vs PVV Left'),
    ('FAA_GL_left', 'FAA_PVV_right', 'SF: GL Left vs PVV Right'),
    ('FAA_GL_right', 'FAA_PVV_left', 'SF: GL Right vs PVV Left'),
    ('FAA_GL_right', 'FAA_PVV_right', 'SF: GL Right vs PVV Right'),
    ('FAA_PVV_left', 'FAA_PVV_right', 'SF: PVV Left vs PVV Right')
]

within_results = []
for group in ['Low Neg PIS', 'High Neg PIS']:
    grp_df = df_end[df_end['PIS_Neg_Group'] == group]
    for colA, colB, label in paired_contrasts:
        diff, d, p = run_paired_perm_ttest(grp_df, colA, colB)
        within_results.append({'Group': group, 'Contrast': label, 'Delta': diff, 'd': d, 'p_raw': p})

df_within = pd.DataFrame(within_results)

# -------------------------------------------------------------------------
# D. Export Functions (LaTeX and CSV)
# -------------------------------------------------------------------------
def get_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''

# 1. Export Between-Subjects Table
df_between_out = df_between.copy()
df_between_out['sig'] = df_between_out['p_raw'].apply(get_stars)
df_between_out.to_csv(FIGURES_DIR / 'ttest_Between_PIS_Aggregates.csv', index=False)

tex_between = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Independent Permutation t-tests: High vs Low Negative PIS on Aggregated Metrics}\n"
    "\\label{tab:ttest_between_pis_agg}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.2] S[table-format=1.2e-2]@{}}\n"
    "\\toprule\n"
    "\\textbf{Metric} & \\textbf{$\\Delta$ (High - Low)} & \\textbf{d} & \\textbf{$p_\\text{raw}$} \\\\\n"
    "\\midrule\n"
)
for _, row in df_between_out.iterrows():
    p_tex = fmt_p_tex(row['p_raw'])
    tex_between += f"{row['Metric']} & {row['Delta (High - Low)']:.3f} & {row['d']:.2f} & {p_tex}{row['sig']} \\\\\n"
tex_between += (
    "\\bottomrule\n\\end{tabular}\n"
    "\\begin{tablenotes}\n\\footnotesize\n"
    f"\\item \\textit{{Note.}} Positive $\\Delta$ indicates higher FAA for High Neg PIS group. $p$-values calculated via {N_PERMS} permutations.\n"
    "\\end{tablenotes}\n"
    "\\end{threeparttable}\n"
    "\\end{table}"
)
with open(FIGURES_DIR / 'ttest_Between_PIS_Aggregates.tex', 'w') as f: f.write(tex_between)

# 2. Export Within-Subjects Table
df_within_out = df_within.copy()
df_within_out['sig'] = df_within_out['p_raw'].apply(get_stars)
df_within_out.to_csv(FIGURES_DIR / 'ttest_Within_PIS_Contrasts.csv', index=False)

tex_within = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Paired Permutation Contrasts within PIS Groups}\n"
    "\\label{tab:ttest_within_pis}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabular}{@{}ll S[table-format=1.3] S[table-format=1.2] S[table-format=1.2e-2]@{}}\n"
    "\\toprule\n"
    "\\textbf{Group} & \\textbf{Contrast} & \\textbf{$\\Delta$} & \\textbf{d} & \\textbf{$p_\\text{raw}$} \\\\\n"
    "\\midrule\n"
)
for _, row in df_within_out.iterrows():
    p_tex = fmt_p_tex(row['p_raw'])
    tex_within += f"{row['Group']} & {row['Contrast']} & {row['Delta']:.3f} & {row['d']:.2f} & {p_tex}{row['sig']} \\\\\n"
tex_within += (
    "\\bottomrule\n\\end{tabular}\n"
    "\\begin{tablenotes}\n\\footnotesize\n"
    f"\\item \\textit{{Note.}} $\\Delta$ represents the mean difference (First listed condition - Second listed condition). $p$-values calculated via {N_PERMS} permutations.\n"
    "\\end{tablenotes}\n"
    "\\end{threeparttable}\n"
    "\\end{table}"
)
with open(FIGURES_DIR / 'ttest_Within_PIS_Contrasts.tex', 'w') as f: f.write(tex_within)

print(f"Successfully saved SF condition pointplot and updated t-test tables to {FIGURES_DIR}.")

#%%