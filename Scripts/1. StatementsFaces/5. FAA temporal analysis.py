# %% #######################################################################################################################
# The goal of this script is to run temporal bin analysis, where we test whether FAA has changed from beginning to end of tasks
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Cell 1: Setup, Single-Trial Extraction & Temporal Binning
import os
import mne
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", module="statsmodels")

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"
FAA_METHOD = "diff"
USE_Z_SCORES = False  # Toggle: Set to False to analyze and plot unstandardized raw values

PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids' / 'derivatives'
QC_DIR = DERIVATIVES_DIR / 'quality_check' / BATCH_NAME

# NEW: Temporal Directory
TEMPORAL_DIR = DERIVATIVES_DIR / 'temporal' / BATCH_NAME / FAA_METHOD
TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

# Aligned exclusion list with script 4
EXCLUDE_SUBJECTS = {"sub-P03"}  
flag_file = QC_DIR / 'Flagged_Bad_F3_F4_Participants.csv'
if flag_file.exists():
    df_flags = pd.read_csv(flag_file)
    if not df_flags.empty and 'Subject' in df_flags.columns:
        EXCLUDE_SUBJECTS.update(df_flags['Subject'].unique())

# ==============================================================================
# SUBJECT AFFILIATION MAPPING 
# ==============================================================================
# In-party is always GL; Out-party is always PVV
PARTY_AFFILIATION = {
    'GL': 'In-Party',
    'PVV': 'Out-Party'
}

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

ID_TO_EVENT_JF = {v: k for k, v in EVENTS_JUSTFACES.items()}
ID_TO_EVENT_SF = {v: k for k, v in EVENTS_STATEMENTSFACES.items()}

print("==================================================")
data_label = "Z-scored" if USE_Z_SCORES else "Raw"
print(f"Extracting Single-Trial Data ({data_label}) for Temporal Analysis...")
print("==================================================")

eeg_processed_dir = DERIVATIVES_DIR / 'processed' / BATCH_NAME
subject_folders = sorted([d for d in eeg_processed_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])

jf_trials_list = []
sf_trials_list = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    eeg_dir = subj_dir / 'eeg'
    jf_file = eeg_dir / f"{subject}_task-JustFaces_faa_{FAA_METHOD}-epo.fif"
    sf_file = eeg_dir / f"{subject}_task-StatementsFaces_faa_{FAA_METHOD}-epo.fif" 

    # Extract JF
    if jf_file.exists():
        epochs_jf = mne.read_epochs(jf_file, preload=True, verbose=False)
        time_mask_jf = (epochs_jf.times >= 0.0) & (epochs_jf.times <= 1.2)
        faa_data_jf = epochs_jf.get_data(picks='FAA')[:, 0, :]
        mean_faa_jf = np.mean(faa_data_jf[:, time_mask_jf], axis=-1)
        
        for idx, ev_id in enumerate(epochs_jf.events[:, 2]):
            event_name = ID_TO_EVENT_JF.get(ev_id)
            if not event_name or 'Foil' in event_name: continue
            
            face_party = event_name.split('/')[0]
            party_match = PARTY_AFFILIATION.get(face_party, 'UNKNOWN')
            
            # THE FIX: Absolute Trial Index
            absolute_trial_idx = epochs_jf.selection[idx] + 1
            
            jf_trials_list.append({
                'Subject': subject, 'Trial': absolute_trial_idx,
                'Face': face_party, 'Party_Match': party_match,
                'FAA_Value': mean_faa_jf[idx]
            })

    # Extract SF
    if sf_file.exists():
        epochs_sf = mne.read_epochs(sf_file, preload=True, verbose=False)
        time_mask_sf = (epochs_sf.times >= 0.0) & (epochs_sf.times <= 1.2)
        faa_data_sf = epochs_sf.get_data(picks='FAA')[:, 0, :]
        mean_faa_sf = np.mean(faa_data_sf[:, time_mask_sf], axis=-1)
        
        for idx, ev_id in enumerate(epochs_sf.events[:, 2]):
            event_name = ID_TO_EVENT_SF.get(ev_id)
            if not event_name: continue
            
            face_party = event_name.split('/')[0]
            party_match = PARTY_AFFILIATION.get(face_party, 'UNKNOWN')
            
            # Absolute Trial Index
            absolute_trial_idx = epochs_sf.selection[idx] + 1
            
            sf_trials_list.append({
                'Subject': subject, 'Trial': absolute_trial_idx,
                'Face': face_party, 'Party_Match': party_match,
                'FAA_Value': mean_faa_sf[idx]
            })

df_jf = pd.DataFrame(jf_trials_list)
df_sf = pd.DataFrame(sf_trials_list)

# Standardize values per participant 
if USE_Z_SCORES:
    df_jf['FAA_Value'] = df_jf.groupby('Subject')['FAA_Value'].transform(lambda x: (x - x.mean()) / x.std())
    df_sf['FAA_Value'] = df_sf.groupby('Subject')['FAA_Value'].transform(lambda x: (x - x.mean()) / x.std())

# ==============================================================================
# Define 25% Time Bins per Subject 
# ==============================================================================
def assign_time_bins(df):
    df_out = []
    for subj, grp in df.groupby('Subject'):
        # Percentiles now calculated on the absolute trial timeline
        q1 = np.percentile(grp['Trial'], 25)
        q3 = np.percentile(grp['Trial'], 75)
        
        grp = grp.copy()
        grp['Time_Bin'] = 'Middle_50'
        grp.loc[grp['Trial'] <= q1, 'Time_Bin'] = 'First_25'
        grp.loc[grp['Trial'] >= q3, 'Time_Bin'] = 'Last_25'
        df_out.append(grp)
    return pd.concat(df_out, ignore_index=True)

df_jf = assign_time_bins(df_jf)
df_sf = assign_time_bins(df_sf)

print(f"JF Data: {len(df_jf)} trials prepared.")
print(f"SF Data: {len(df_sf)} trials prepared.")

# %% Cell 2: Temporal Analysis Handler Logic
import statsmodels.formula.api as smf
import scipy.stats as stats
from patsy import dmatrix

def run_temporal_analysis(df, task_name, condition_name):
    print(f"\n--- Running Temporal Analysis: {task_name} | {condition_name} ---")
    
    # Isolate first and last 25%
    df_eval = df[df['Time_Bin'].isin(['First_25', 'Last_25'])].copy()
    
    # Model specification: Only evaluating the categorical bin, omitting trial order
    models_to_test = {
        "GLM": {"type": "ols", "formula": "FAA_Value ~ C(Time_Bin, Treatment('First_25'))"},
        "LMM": {"type": "mixedlm", "formula": "FAA_Value ~ C(Time_Bin, Treatment('First_25'))", "groups": "Subject"}
    }
    
    # 1. Model Selection
    fit_results = []
    for name, params in models_to_test.items():
        try:
            if params["type"] == "ols":
                model = smf.ols(params["formula"], data=df_eval).fit()
            else:
                model = smf.mixedlm(params["formula"], data=df_eval, groups=df_eval[params["groups"]]).fit(reml=False)
            fit_results.append({'Model': name, 'AIC': model.aic, 'BIC': model.bic, 'LogLik': model.llf})
        except Exception:
            fit_results.append({'Model': name, 'AIC': np.inf, 'BIC': np.inf, 'LogLik': np.nan})
            
    df_fits = pd.DataFrame(fit_results).sort_values(by='AIC').reset_index(drop=True)
    best_config = models_to_test[df_fits.iloc[0]['Model']]
    print(f"Winning Structure: {df_fits.iloc[0]['Model']}")
    
    # 2. Fit Winning Model
    formula = best_config['formula']
    if best_config['type'] == 'mixedlm':
        final_model = smf.mixedlm(formula, data=df_eval, groups=df_eval[best_config['groups']]).fit(reml=True)
    else:
        final_model = smf.ols(formula, data=df_eval).fit()
        
    res_fixed = pd.DataFrame({
        'Predictor': final_model.params.index, 'Estimate': final_model.params.values,
        'SE': final_model.bse.values, 't_z_value': final_model.tvalues.values,
        'p_value': final_model.pvalues.values,
        'CI_Lower': final_model.conf_int()[0].values, 'CI_Upper': final_model.conf_int()[1].values
    })
    res_fixed = res_fixed[~res_fixed['Predictor'].str.contains('Var')]
    
    # 3. EMMs & Contrasts manually via patsy
    df_pred = pd.DataFrame({'Time_Bin': ['First_25', 'Last_25']})
    X = dmatrix(formula.split('~')[1], df_pred, return_type='dataframe')
    fe_cols = [col for col in final_model.params.index if col in X.columns]
    X_mat, params, cov = X[fe_cols].values, final_model.params[fe_cols].values, final_model.cov_params().loc[fe_cols, fe_cols].values
    
    resid_sd = np.sqrt(final_model.scale if hasattr(final_model, 'scale') else final_model.mse_resid)
    emms = X_mat.dot(params)
    se_emms = np.sqrt(np.diag(X_mat.dot(cov).dot(X_mat.T)))
    
    df_emms = pd.DataFrame({'Time_Bin': df_pred['Time_Bin'], 'EMM': emms, 'SE': se_emms})
    
    # Contrast: Last_25 - First_25
    c_vec = X_mat[1] - X_mat[0]
    diff = c_vec.dot(params)
    se_diff = np.sqrt(c_vec.dot(cov).dot(c_vec.T))
    z_stat = diff / se_diff
    p_raw = 2 * stats.norm.sf(np.abs(z_stat))
    
    df_contrast = pd.DataFrame([{
        'Comparison': 'Last_25 vs First_25', 'Estimate': diff, 'SE': se_diff, 
        'z_stat': z_stat, 'd': diff / resid_sd, 'p_value': p_raw
    }])
    
    # 4. Exporting 
    dtype_prefix = "Zscored" if USE_Z_SCORES else "Raw"
    prefix = f"{task_name}_{condition_name}_{dtype_prefix}"
    
    res_fixed.to_csv(TEMPORAL_DIR / f'FixedEffects_{prefix}.csv', index=False)
    df_emms.to_csv(TEMPORAL_DIR / f'EMMs_{prefix}.csv', index=False)
    df_contrast.to_csv(TEMPORAL_DIR / f'Contrast_{prefix}.csv', index=False)
    
    tex_rows = ""
    for _, row in res_fixed.iterrows():
        p = row['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.10 else ''
        pred = row['Predictor'].replace('_', '\\_').replace('C(', '').replace(')', '').replace('[T.', ' [').replace(']', ']')
        tex_rows += f"{pred} & {row['Estimate']:.3f} & {row['SE']:.3f} & {row['t_z_value']:.2f} & {p:.3f} {sig} \\\\\n"

    tex_table = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Temporal Fixed Effects: {task_name} ({condition_name}, {dtype_prefix})}}\n"
        "\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.3] S[table-format=2.2] l@{}}\n"
        "\\toprule\n\\textbf{Predictor} & \\textbf{$b$} & \\textbf{SE} & \\textbf{$t/z$} & \\textbf{$p$} \\\\\n\\midrule\n"
        f"{tex_rows}\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"
    )
    with open(TEMPORAL_DIR / f'FixedEffects_{prefix}.tex', 'w') as f: f.write(tex_table)
    
    print(df_contrast.to_string(index=False))
    return final_model, df_emms, df_contrast

# %% Cell 3: JF Temporal Analyses
print("=============================================================================================")
print("JUSTFACES (JF) TEMPORAL ANALYSES")
print("=============================================================================================")

# 1. Overall JF
mod_jf_all, emms_jf_all, cont_jf_all = run_temporal_analysis(df_jf, 'JF', 'Overall')

# 2. In-Party JF (GL)
df_jf_in = df_jf[df_jf['Party_Match'] == 'In-Party']
mod_jf_in, emms_jf_in, cont_jf_in = run_temporal_analysis(df_jf_in, 'JF', 'In-Party')

# 3. Out-Party JF (PVV)
df_jf_out = df_jf[df_jf['Party_Match'] == 'Out-Party']
mod_jf_out, emms_jf_out, cont_jf_out = run_temporal_analysis(df_jf_out, 'JF', 'Out-Party')

# %% Cell 4: SF Temporal Analyses
print("\n=============================================================================================")
print("STATEMENTSFACES (SF) TEMPORAL ANALYSES")
print("=============================================================================================")

# 1. Overall SF
mod_sf_all, emms_sf_all, cont_sf_all = run_temporal_analysis(df_sf, 'SF', 'Overall')

# 2. In-Party SF (GL)
df_sf_in = df_sf[df_sf['Party_Match'] == 'In-Party']
mod_sf_in, emms_sf_in, cont_sf_in = run_temporal_analysis(df_sf_in, 'SF', 'In-Party')

# 3. Out-Party SF (PVV)
df_sf_out = df_sf[df_sf['Party_Match'] == 'Out-Party']
mod_sf_out, emms_sf_out, cont_sf_out = run_temporal_analysis(df_sf_out, 'SF', 'Out-Party')

# %% Cell 5: Visualization of Temporal Trajectories
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

print("\n=============================================================================================")
print("GENERATING TEMPORAL VISUALIZATIONS")
print("=============================================================================================")

# Define the smoothing window (number of adjacent trials)
SMOOTHING_WINDOW = 10 

def assign_dense_condition_index(df, condition_col):
    """
    Maps absolute trial indices to a dense condition-specific sequence (1, 2, 3...).
    This compresses the x-axis (removing gaps from other conditions) while ensuring 
    that if a participant misses t=1, their next trial correctly aligns at t=2.
    """
    df_out = df.copy()
    df_out['Condition_Trial'] = np.nan
    for cond in df_out[condition_col].dropna().unique():
        # Find all absolute trial numbers where this condition appeared across ALL participants
        unique_abs_trials = np.sort(df_out[df_out[condition_col] == cond]['Trial'].unique())
        # Map absolute trial to dense index 1, 2, 3...
        mapping = {abs_trial: idx + 1 for idx, abs_trial in enumerate(unique_abs_trials)}
        # Apply mapping
        mask = df_out[condition_col] == cond
        df_out.loc[mask, 'Condition_Trial'] = df_out.loc[mask, 'Trial'].map(mapping)
    return df_out

def smooth_data(df, window, condition_col):
    """Applies rolling average smoothing chronologically per subject and condition."""
    df_smoothed = df.copy()
    # Sort chronologically using the new dense trial index
    df_smoothed = df_smoothed.sort_values(by=['Subject', 'Condition_Trial'])
    # Group by subject and condition, then apply rolling mean
    df_smoothed['FAA_Value'] = df_smoothed.groupby(['Subject', condition_col])['FAA_Value'].transform(
        lambda x: x.rolling(window=window, min_periods=1, center=True).mean()
    )
    return df_smoothed

# 1. Map absolute trials to condensed condition indices
df_jf = assign_dense_condition_index(df_jf, 'Party_Match')
df_sf = assign_dense_condition_index(df_sf, 'Party_Match')

# 2. Apply smoothing using the condensed timeline
df_jf_smooth = smooth_data(df_jf, SMOOTHING_WINDOW, 'Party_Match')
df_sf_smooth = smooth_data(df_sf, SMOOTHING_WINDOW, 'Party_Match')

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)

# Define clean color palette for In-Party (GL) and Out-Party (PVV)
party_palette = {'In-Party': '#377EB8', 'Out-Party': '#E41A1C'} 
y_axis_label = f"FAA Index ({'Z-scored' if USE_Z_SCORES else 'Raw'})"

max_trial_jf = df_jf['Condition_Trial'].max()
max_trial_sf = df_sf['Condition_Trial'].max()

#  JustFaces Smoothed 
sns.lineplot(
    data=df_jf_smooth, x='Condition_Trial', y='FAA_Value', hue='Party_Match',
    palette=party_palette, errorbar='se', ax=axes[0, 0], linewidth=2.5
)
axes[0, 0].set_title(f'JustFaces (JF): Smoothed FAA (Window: {SMOOTHING_WINDOW})', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel(y_axis_label, fontsize=12)
axes[0, 0].set_xlabel('Condition-Specific Trial Index', fontsize=12)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
axes[0, 0].set_xlim(1, max_trial_jf)
axes[0, 0].legend(title="Stimulus Match")

#  StatementsFaces Smoothed 
sns.lineplot(
    data=df_sf_smooth, x='Condition_Trial', y='FAA_Value', hue='Party_Match',
    palette=party_palette, errorbar='se', ax=axes[0, 1], linewidth=2.5
)
axes[0, 1].set_title(f'StatementsFaces (SF): Smoothed FAA (Window: {SMOOTHING_WINDOW})', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('')
axes[0, 1].set_xlabel('Condition-Specific Trial Index', fontsize=12)
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
axes[0, 1].set_xlim(1, max_trial_sf)
axes[0, 1].legend(title="Stimulus Match")

# Function to plot regressions 
def plot_regressions(ax, df, title, max_val):
    for party in ['In-Party', 'Out-Party']:
        party_data = df[df['Party_Match'] == party].dropna(subset=['Condition_Trial', 'FAA_Value'])
        
        # Aggregate across subjects mapping to dense condition time
        agg_data = party_data.groupby('Condition_Trial')['FAA_Value'].mean().reset_index()
        
        X = agg_data[['Condition_Trial']]
        y = agg_data['FAA_Value']
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        
        # Curvilinear (Polynomial degree 2) Regression
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        pr = LinearRegression()
        pr.fit(X_poly, y)
        y_pred_pr = pr.predict(X_poly)
        
        # Plotting
        color = party_palette[party]
        ax.plot(X, y_pred_lr, color=color, linestyle='--', label=f'{party} (Linear)')
        ax.plot(X, y_pred_pr, color=color, linestyle=':', label=f'{party} (Curvilinear)')
        # Add a subtle scatter of the aggregated points for reference
        ax.scatter(X, y, color=color, alpha=0.3, s=12)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition-Specific Trial Index', fontsize=12)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlim(1, max_val)
    ax.legend(title="Regression Fits")

# JustFaces Regressions 
axes[1, 0].set_ylabel(y_axis_label, fontsize=12)
plot_regressions(axes[1, 0], df_jf, 'JustFaces (JF): Regression Models', max_trial_jf)

# StatementsFaces Regressions 
plot_regressions(axes[1, 1], df_sf, 'StatementsFaces (SF): Regression Models', max_trial_sf)

plt.tight_layout()
dtype_prefix = "Zscored" if USE_Z_SCORES else "Raw"
viz_path = TEMPORAL_DIR / f'Temporal_Progression_Visualizations_Expanded_{dtype_prefix}.png'
fig.savefig(viz_path, dpi=300)
plt.show()

print(f"Temporal visualizations successfully saved to: {viz_path}")

# %% SF Condition-Specific Temporal Analysis (Restructured for Leaning Congruency)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

print("\n=============================================================================================")
print("STATEMENTSFACES (SF): CONDITION-SPECIFIC TEMPORAL ANALYSIS (LEANING CONGRUENCY)")
print("=============================================================================================")

# Define the smoothing window
SMOOTHING_WINDOW = 10

# 1. Prepare Data with Leaning Congruency 
sf_detailed_list = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    eeg_dir = subj_dir / 'eeg'
    sf_file = eeg_dir / f"{subject}_task-StatementsFaces_faa_{FAA_METHOD}-epo.fif" 

    if sf_file.exists():
        epochs_sf = mne.read_epochs(sf_file, preload=True, verbose=False)
        time_mask_sf = (epochs_sf.times >= 0.0) & (epochs_sf.times <= 1.2)
        faa_data_sf = epochs_sf.get_data(picks='FAA')[:, 0, :]
        mean_faa_sf = np.mean(faa_data_sf[:, time_mask_sf], axis=-1)
        
        for idx, ev_id in enumerate(epochs_sf.events[:, 2]):
            event_name = ID_TO_EVENT_SF.get(ev_id)
            if not event_name: continue
            
            parts = event_name.split('/')
            face_party = parts[0]
            statement_lean = parts[2] # Pulls 'left' or 'right'
            
            party_match = PARTY_AFFILIATION.get(face_party, 'UNKNOWN')
            
            # Determine Congruency based on Party and Political Leaning
            if party_match == 'In-Party':
                congruency = 'Congruent' if statement_lean == 'left' else 'Incongruent'
            elif party_match == 'Out-Party':
                congruency = 'Congruent' if statement_lean == 'right' else 'Incongruent'
            else:
                continue

            condition_label = f"{congruency} {party_match}"
            
            # Use epochs.selection to get the EXACT original trial position
            absolute_trial_idx = epochs_sf.selection[idx] + 1
            
            sf_detailed_list.append({
                'Subject': subject, 
                'Trial': absolute_trial_idx, 
                'Party_Match': party_match,
                'Leaning': statement_lean,
                'Congruency': congruency,
                'Condition': condition_label,
                'FAA_Value': mean_faa_sf[idx]
            })

df_sf_cond = pd.DataFrame(sf_detailed_list)

# Standardize values per participant if requested
if USE_Z_SCORES:
    df_sf_cond['FAA_Value'] = df_sf_cond.groupby('Subject')['FAA_Value'].transform(lambda x: (x - x.mean()) / x.std())

# 2. Dense Indexing and Smoothing 

def assign_dense_condition_index_cond(df, condition_col):
    """Maps absolute trial indices to a sequential 1..N index PER SUBJECT."""
    df_out = df.copy()
    df_out = df_out.sort_values(by=['Subject', 'Trial'])
    # Group by Subject and Condition, then assign a clean sequential count
    df_out['Condition_Trial'] = df_out.groupby(['Subject', condition_col]).cumcount() + 1
    return df_out

def smooth_data_cond(df, window):
    df_smoothed = df.copy()
    df_smoothed = df_smoothed.sort_values(by=['Subject', 'Condition_Trial'])
    # Grouping by Condition ensures the rolling window correctly smooths occurrences
    df_smoothed['FAA_Value'] = df_smoothed.groupby(['Subject', 'Condition'])['FAA_Value'].transform(
        lambda x: x.rolling(window=window, min_periods=1, center=True).mean()
    )
    return df_smoothed

# Apply transformations
df_sf_cond = assign_dense_condition_index_cond(df_sf_cond, 'Condition')
df_sf_cond_smooth = smooth_data_cond(df_sf_cond, SMOOTHING_WINDOW)

# 3. Visualization 
cond_palette = {
    'Congruent Out-Party': '#E41A1C',   
    'Incongruent Out-Party': '#F781BF', 
    'Incongruent In-Party': '#1FBFBF',  
    'Congruent In-Party': '#225EA8'     
}

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

y_axis_label = f"FAA Index ({'Z-scored' if USE_Z_SCORES else 'Raw'})"
max_trial_cond = df_sf_cond['Condition_Trial'].max()

# Plot A: Smoothed Lineplot across dense condition time
sns.lineplot(
    data=df_sf_cond_smooth, x='Condition_Trial', y='FAA_Value', hue='Condition',
    palette=cond_palette, errorbar='se', ax=axes[0], linewidth=2.5
)
axes[0].set_title(f'SF Conditions: Smoothed FAA (Window: {SMOOTHING_WINDOW})', fontsize=14, fontweight='bold')
axes[0].set_ylabel(y_axis_label, fontsize=12)
axes[0].set_xlabel('Condition-Specific Trial Index', fontsize=12)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
axes[0].set_xlim(1, max_trial_cond)
axes[0].legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot B: Regressions across dense condition time
for cond in cond_palette.keys():
    cond_data = df_sf_cond[df_sf_cond['Condition'] == cond].dropna(subset=['Condition_Trial', 'FAA_Value'])
    if cond_data.empty: continue

    # Aggregate across subjects for the condition timeline
    agg_data = cond_data.groupby('Condition_Trial')['FAA_Value'].mean().reset_index()
    
    X = agg_data[['Condition_Trial']]
    y = agg_data['FAA_Value']
    color = cond_palette[cond]
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    axes[1].plot(X, y_pred_lr, color=color, linestyle='--', label=f'{cond} (Linear)')
    
    # Curvilinear (Polynomial degree 2) Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    pr = LinearRegression()
    pr.fit(X_poly, y)
    y_pred_pr = pr.predict(X_poly)
    axes[1].plot(X, y_pred_pr, color=color, linestyle=':', label=f'{cond} (Curv)')
    
    # Add scatter to show the actual aggregated points
    axes[1].scatter(X, y, color=color, alpha=0.3, s=12)

axes[1].set_title('SF Conditions: Regression Models', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Condition-Specific Trial Index', fontsize=12)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
axes[1].set_xlim(1, max_trial_cond)

# Shrink current axis by 20% to fit legend outside
box = axes[1].get_position()
axes[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
axes[1].legend(title="Regression Fits", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
dtype_prefix = "Zscored" if USE_Z_SCORES else "Raw"
viz_path = TEMPORAL_DIR / f'Temporal_Progression_SF_Leaning_Conditions_{dtype_prefix}_smooth{SMOOTHING_WINDOW}.png'
fig.savefig(viz_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"SF Condition visualizations successfully saved to: {viz_path}")
# %% Cell 7: SF Condition-Specific Temporal Analyses (First 50 vs Last 50)
import statsmodels.formula.api as smf
import scipy.stats as stats
from patsy import dmatrix
import pandas as pd
import numpy as np

print("\n=============================================================================================")
print("STATEMENTSFACES (SF): CONDITION-SPECIFIC TEMPORAL ANALYSIS (FIRST 50 vs LAST 50)")
print("=============================================================================================")

# 1. Define 50/50 Time Bins per Subject
def assign_5050_time_bins(df):
    """Splits trials into First_50 and Last_50 based on the median absolute trial per participant."""
    df_out = []
    for subj, grp in df.groupby('Subject'):
        median = np.percentile(grp['Trial'], 50)
        grp = grp.copy()
        grp['Time_Bin'] = 'Last_50'
        grp.loc[grp['Trial'] <= median, 'Time_Bin'] = 'First_50'
        df_out.append(grp)
    return pd.concat(df_out, ignore_index=True)

# Apply the 50/50 binning to the Condition DataFrame from Cell 6
df_sf_cond_binned = assign_5050_time_bins(df_sf_cond)

# 2. Localized Analysis Handler for 50/50 Split
def run_condition_temporal_analysis_5050(df_binned, condition_label):
    print(f"\n--- Running Temporal Analysis (50/50): SF | {condition_label} ---")
    
    # Isolate the specific condition
    df_eval = df_binned[df_binned['Condition'] == condition_label].copy()
    
    if len(df_eval) == 0:
        print("No trials available for this condition.")
        return None

    # Define Models
    models_to_test = {
        "GLM": {"type": "ols", "formula": "FAA_Value ~ C(Time_Bin, Treatment('First_50'))"},
        "LMM": {"type": "mixedlm", "formula": "FAA_Value ~ C(Time_Bin, Treatment('First_50'))", "groups": "Subject"}
    }
    
    # Selection Loop
    fit_results = []
    for name, params in models_to_test.items():
        try:
            if params["type"] == "ols":
                model = smf.ols(params["formula"], data=df_eval).fit()
            else:
                model = smf.mixedlm(params["formula"], data=df_eval, groups=df_eval[params["groups"]]).fit(
                    reml=False, method='lbfgs', maxiter=500
                )
            fit_results.append({'Model': name, 'AIC': model.aic, 'BIC': model.bic, 'LogLik': model.llf})
        except Exception:
            fit_results.append({'Model': name, 'AIC': np.inf, 'BIC': np.inf, 'LogLik': np.nan})
            
    df_fits = pd.DataFrame(fit_results).sort_values(by='AIC').reset_index(drop=True)
    best_config = models_to_test[df_fits.iloc[0]['Model']]
    print(f"Winning Structure: {df_fits.iloc[0]['Model']}")
    
    # Final Fit
    formula = best_config['formula']
    if best_config['type'] == 'mixedlm':
        final_model = smf.mixedlm(formula, data=df_eval, groups=df_eval[best_config['groups']]).fit(
            reml=True, method='lbfgs', maxiter=500
        )
    else:
        final_model = smf.ols(formula, data=df_eval).fit()
        
    res_fixed = pd.DataFrame({
        'Predictor': final_model.params.index, 'Estimate': final_model.params.values,
        'SE': final_model.bse.values, 't_z_value': final_model.tvalues.values,
        'p_value': final_model.pvalues.values,
        'CI_Lower': final_model.conf_int()[0].values, 'CI_Upper': final_model.conf_int()[1].values
    })
    res_fixed = res_fixed[~res_fixed['Predictor'].str.contains('Var')]
    
    # EMMs and Pairwise Contrast via Patsy Matrix
    df_pred = pd.DataFrame({'Time_Bin': ['First_50', 'Last_50']})
    X = dmatrix(formula.split('~')[1], df_pred, return_type='dataframe')
    fe_cols = [col for col in final_model.params.index if col in X.columns]
    X_mat, params, cov = X[fe_cols].values, final_model.params[fe_cols].values, final_model.cov_params().loc[fe_cols, fe_cols].values
    
    resid_sd = np.sqrt(final_model.scale if hasattr(final_model, 'scale') else final_model.mse_resid)
    emms = X_mat.dot(params)
    se_emms = np.sqrt(np.diag(X_mat.dot(cov).dot(X_mat.T)))
    df_emms = pd.DataFrame({'Time_Bin': df_pred['Time_Bin'], 'EMM': emms, 'SE': se_emms})
    
    # Contrast: Last_50 - First_50
    c_vec = X_mat[1] - X_mat[0]
    diff = c_vec.dot(params)
    se_diff = np.sqrt(c_vec.dot(cov).dot(c_vec.T))
    z_stat = diff / se_diff
    p_raw = 2 * stats.norm.sf(np.abs(z_stat))
    
    df_contrast = pd.DataFrame([{
        'Comparison': 'Last_50 vs First_50', 'Estimate': diff, 'SE': se_diff, 
        'z_stat': z_stat, 'd': diff / resid_sd, 'p_value': p_raw
    }])
    
    # Export Outputs
    dtype_prefix = "Zscored" if USE_Z_SCORES else "Raw"
    safe_cond_name = condition_label.replace(' ', '_').replace('-', '')
    prefix = f"SF_{safe_cond_name}_5050_{dtype_prefix}"
    
    res_fixed.to_csv(TEMPORAL_DIR / f'FixedEffects_{prefix}.csv', index=False)
    df_emms.to_csv(TEMPORAL_DIR / f'EMMs_{prefix}.csv', index=False)
    df_contrast.to_csv(TEMPORAL_DIR / f'Contrast_{prefix}.csv', index=False)
    
    print(df_contrast.to_string(index=False))
    return final_model, df_emms, df_contrast

# 3. Execute Analysis Loop for all 4 Conditions
conditions_to_test = [
    'Congruent In-Party', 
    'Incongruent In-Party', 
    'Congruent Out-Party', 
    'Incongruent Out-Party'
]

results_dict = {}
for cond in conditions_to_test:
    # Double check if the condition actually exists in the data
    if cond in df_sf_cond_binned['Condition'].unique():
        results_dict[cond] = run_condition_temporal_analysis_5050(df_sf_cond_binned, cond)
    else:
        print(f"\nSkipping {cond}: Not found in data.")

# %%
