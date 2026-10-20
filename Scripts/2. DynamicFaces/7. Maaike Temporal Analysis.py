# %% #######################################################################################################################
# The goal of this script is to run temporal bin analysis, where we test whether FAA has changed from beginning to end of tasks
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Setup, Single-Trial Phase Extraction & Temporal Binning
import os
import mne
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import statsmodels.formula.api as smf
import scipy.stats as stats
from patsy import dmatrix

warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
FAA_METHOD = "diff"
EXCLUSION_CRITERION = "Strict"  # Options: "Strict", "Lenient", or "None"

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives'
PROCESSED_DIR = DERIVATIVES_DIR / 'processed' / BATCH_NAME
QC_DIR = DERIVATIVES_DIR / 'quality_check' / BATCH_NAME

# Temporal Directory for simplified Emotion analysis
TEMPORAL_DIR = DERIVATIVES_DIR / 'temporal' / BATCH_NAME / FAA_METHOD
TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Exclusions
# ==============================================================================
EXCLUDE_SUBJECTS = {""}
flag_file = QC_DIR / 'Comprehensive_Exclusions_List.csv'

if flag_file.exists() and EXCLUSION_CRITERION in ["Strict", "Lenient"]:
    df_flags = pd.read_csv(flag_file)
    target_col = f'Exclude_{EXCLUSION_CRITERION}'
    if not df_flags.empty and target_col in df_flags.columns:
        flagged_subs = df_flags[df_flags[target_col] == True]['Subject'].unique()
        EXCLUDE_SUBJECTS.update(flagged_subs)
        
print(f"Applied Exclusion Criterion: {EXCLUSION_CRITERION}")
print(f"Final Exclusion List: {EXCLUDE_SUBJECTS}")

# ==============================================================================
# Event Dictionaries & Mapping
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
ID_TO_EVENT = {v: k for k, v in EVENTS_EMOTION.items()}

print("==================================================")
print(f"Extracting Single-Trial Phase Data for Temporal Analysis...")
print("==================================================")

subject_folders = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
trials_list = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    epo_file = subj_dir / 'eeg' / f"{subject}_task-emotion_faa_{FAA_METHOD}-epo.fif" 
    if not epo_file.exists(): continue
        
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Target Time Masks
    t_static = (epochs.times >= 0.800) & (epochs.times <= 1.900)
    t_dynamic = (epochs.times >= 1.950) & (epochs.times <= 4.000)

    faa_data = epochs.get_data(picks='FAA')[:, 0, :]
    
    for i, event_id in enumerate(epochs.events[:, 2]):
        event_name = ID_TO_EVENT.get(event_id)
        if not event_name: continue
            
        trial_data = faa_data[i, :]
        val_static = np.mean(trial_data[t_static])
        val_dynamic = np.mean(trial_data[t_dynamic])
        
        # Absolute timeline position tracking
        absolute_trial_idx = epochs.selection[i] + 1
        
        trials_list.append({
            'Subject': subject, 'Trial': absolute_trial_idx,
            'Static_Value': val_static, 'Dynamic_Value': val_dynamic
        })

df_master = pd.DataFrame(trials_list)

# ==============================================================================
# Define Time Bins per Subject (Last_50 vs First_50 Timeline)
# ==============================================================================
def assign_custom_time_bins(df):
    df_out = []
    for subj, grp in df.groupby('Subject'):
        median = np.percentile(grp['Trial'], 50)
        
        grp = grp.copy()
        grp['Time_Bin'] = 'Last_50' # Default to Last_50
        grp.loc[grp['Trial'] <= median, 'Time_Bin'] = 'First_50'
        df_out.append(grp)
    return pd.concat(df_out, ignore_index=True)

df_master = assign_custom_time_bins(df_master)
print(f"Master Dataset Prepared: {len(df_master)} total single trials loaded.")

# %% Temporal Analysis Handler Logic
def run_temporal_analysis(df, phase_col, task_label):
    """Evaluates OLS vs LMM structures and saves outputs for a specified target phase."""
    print(f"\n--- Temporal Analysis: Overall {task_label} | Phase: {phase_col} ---")
    
    # Isolate comparison bins
    df_eval = df[df['Time_Bin'].isin(['First_50', 'Last_50'])].copy()
    
    if len(df_eval) == 0:
        print("No trials available in chosen bins for this selection.")
        return None, None, None

    models_to_test = {
        "GLM": {"type": "ols", "formula": f"{phase_col} ~ C(Time_Bin, Treatment('First_50'))"},
        "LMM": {"type": "mixedlm", "formula": f"{phase_col} ~ C(Time_Bin, Treatment('First_50'))", "groups": "Subject"}
    }
    
    # 1. Selection Loop
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
    
    # 2. Final Fit
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
    
    # 3. EMMs and Pairwise Contrast via Patsy Matrix
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
    
    # 4. Export Outputs (CSV and LaTeX)
    prefix = f"{task_label}_{phase_col}_Overall"
    
    res_fixed.to_csv(TEMPORAL_DIR / f'FixedEffects_{prefix}.csv', index=False)
    df_emms.to_csv(TEMPORAL_DIR / f'EMMs_{prefix}.csv', index=False)
    df_contrast.to_csv(TEMPORAL_DIR / f'Contrast_{prefix}.csv', index=False)
    
    # Generate LaTeX table string
    tex_rows = ""
    for _, row in res_fixed.iterrows():
        p = row['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.10 else ''
        pred = row['Predictor'].replace('_', '\\_').replace('C(', '').replace(')', '').replace('[T.', ' [').replace(']', ']')
        tex_rows += f"{pred} & {row['Estimate']:.3f} & {row['SE']:.3f} & {row['t_z_value']:.2f} & {p:.3f} {sig} \\\\\n"

    tex_table = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Temporal Fixed Effects: Overall {task_label}}}\n"
        "\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.3] S[table-format=2.2] l@{}}\n"
        "\\toprule\n\\textbf{Predictor} & \\textbf{$b$} & \\textbf{SE} & \\textbf{$t/z$} & \\textbf{$p$} \\\\\n\\midrule\n"
        f"{tex_rows}\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"
    )
    with open(TEMPORAL_DIR / f'FixedEffects_{prefix}.tex', 'w') as f: f.write(tex_table)
    
    print(df_contrast.to_string(index=False))
    return final_model, df_emms, df_contrast

# %% Execution: Overall Analyses Only
print("\n=============================================================================================")
print("RUNNING MAIN TEMPORAL ANALYSIS STREAMS (OVERALL ONLY)")
print("=============================================================================================")

# 1. Static Phase Analysis
mod_static, emms_static, cont_static = run_temporal_analysis(df_master, 'Static_Value', 'StaticPhase')

# 2. Dynamic Phase Analysis
mod_dynamic, emms_dynamic, cont_dynamic = run_temporal_analysis(df_master, 'Dynamic_Value', 'DynamicPhase')
