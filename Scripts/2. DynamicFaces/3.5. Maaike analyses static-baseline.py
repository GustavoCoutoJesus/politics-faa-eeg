# %% #######################################################################################################################
# The goal of this script is to run FAA regressions for Static Faces and its baseline period
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Cell 1: Single-Trial Data Extraction & Trial-Level Baselining
import os
import time
import mne
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
import scipy.stats as stats
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix
from pathlib import Path

# Suppress convergence warnings for clean output
warnings.filterwarnings("ignore", module="statsmodels")

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
FAA_METHOD = "static_baseline"  # Changed to route outputs to new static_baseline folders
EXCLUSION_CRITERION = "Strict"  

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives'
PROCESSED_DIR = DERIVATIVES_DIR / 'processed' / BATCH_NAME

# Output Directories (Automatically uses the new FAA_METHOD folder name)
REGRESSION_DIR = DERIVATIVES_DIR / 'regression' / BATCH_NAME / FAA_METHOD
PAIRWISE_DIR = DERIVATIVES_DIR / 'pairwise' / BATCH_NAME / FAA_METHOD
FIGURES_DIR = DERIVATIVES_DIR / 'figures' / BATCH_NAME / FAA_METHOD  

REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR = DERIVATIVES_DIR / 'quality_check' / BATCH_NAME

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

# Base Event Dictionaries
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

def parse_event_string(ev_str):
    group = "Non-Politician" if "nonpol" in ev_str else "In-Party" if "inparty" in ev_str else "Out-Party"
    emotion = "Angry" if "angry" in ev_str else "Happy" if "happy" in ev_str else "Neutral"
    return group, emotion

print("==================================================")
print(f"Extracting Single-Trial Data from {BATCH_NAME}...")
print("==================================================")

subject_folders = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
all_trials = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    eeg_dir = subj_dir / 'eeg'
    # Use diff file if the raw epo is saved under diff processing, adjust if needed
    epo_file = eeg_dir / f"{subject}_task-emotion_faa_diff-epo.fif" 
    
    if not epo_file.exists():
        # Fallback to check if a specific static named epo exists, otherwise skip
        epo_file = eeg_dir / f"{subject}_task-emotion_faa_static_baseline-epo.fif"
        if not epo_file.exists():
            continue
        
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Define Target Time Masks for Baseline (
    t_baseline = (epochs.times >= 0.0) & (epochs.times <= 0.8)
    t_static_face = (epochs.times >= 0.800) & (epochs.times <= 1.900)

    faa_data = epochs.get_data(picks='FAA')[:, 0, :]
    
    for i, event_id in enumerate(epochs.events[:, 2]):
        event_name = ID_TO_EVENT.get(event_id)
        if not event_name: continue
            
        group, emotion = parse_event_string(event_name)
        
        # Create the binary grouping variable (Politician vs Non-Politician)
        pol_category = "Non-Politician" if group == "Non-Politician" else "Politician"
        
        trial_data = faa_data[i, :]
        
        # Calculate Phase Means and Difference Score
        val_baseline = np.mean(trial_data[t_baseline])
        val_static_face = np.mean(trial_data[t_static_face])
        diff_val = val_static_face - val_baseline
        
        all_trials.append({
            'Subject': subject,
            'Trial': i + 1,
            'Group': group,
            'Pol_Category': pol_category,
            'Emotion': emotion,
            'FAA_Diff': diff_val
        })

df_trials = pd.DataFrame(all_trials)
print(f"Extraction complete: {len(df_trials)} total single trials formatted.")


# %% Cell 2: Model Testing & Pairwise Tools
print("\n=============================================================================================")

# TOGGLE STANDARDIZATION 
USE_Z_SCORES = False

metric_name = "Standardized" if USE_Z_SCORES else "Unstandardized"
coef_sym = "$\\beta$" if USE_Z_SCORES else "$b$"

print(f"EVALUATING AND FITTING {metric_name.upper()} MODELS")
print("=============================================================================================")

# ==============================================================================
# Formatting & Naming Helpers
# ==============================================================================
def fmt2(x): return f"{float(x):.2f}" if pd.notna(x) else ""
def fmt3(x): return f"{float(x):.3f}" if pd.notna(x) else ""

def fmt_p_tex(p):
    """Formats p-values for LaTeX, using scientific notation for p < .001"""
    if pd.isna(p): return ""
    if p < 0.001:
        base, exp = f"{p:.2e}".split('e')
        return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"
    return f"{p:.3f}"

def ci_str(lo, hi, decimals=2):
    if pd.isna(lo) or pd.isna(hi): return ""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"

def pretty_term(term):
    """Cleans up statsmodels predictors for reporting, placing reference first."""
    t = str(term)
    t = t.replace("Intercept", "(Intercept)")
    t = t.replace("Trial_Target", "Trial")
    
    # Binary Politician vs Non-Politician
    t = t.replace("C(Pol_Category, Treatment('Non-Politician'))[T.Politician]", "Face (Non-Politician vs. Politician)")
    
    # 3-Level Group 
    t = t.replace("C(Group, Treatment('Non-Politician'))[T.In-Party]", "Face (Non-Politician vs. In-Party)")
    t = t.replace("C(Group, Treatment('Non-Politician'))[T.Out-Party]", "Face (Non-Politician vs. Out-Party)")
    t = t.replace("C(Group, Treatment('In-Party'))[T.Non-Politician]", "Face (In-Party vs. Non-Politician)")
    t = t.replace("C(Group, Treatment('In-Party'))[T.Out-Party]", "Face (In-Party vs. Out-Party)")
    
    # Emotion Main Effects
    t = t.replace("C(Emotion, Treatment('Neutral'))[T.Angry]", "Emotion (Neutral vs. Angry)")
    t = t.replace("C(Emotion, Treatment('Neutral'))[T.Happy]", "Emotion (Neutral vs. Happy)")
    t = t.replace("C(Emotion, Treatment('Angry'))[T.Neutral]", "Emotion (Angry vs. Neutral)")
    t = t.replace("C(Emotion, Treatment('Angry'))[T.Happy]", "Emotion (Angry vs. Happy)")
    
    return t

def get_model_context(prefix):
    """Generates clear, publication-ready descriptions of the analytical subset."""
    prefix_map = {
        'Static_3Level': '3-level group effect (Difference Score)',
        'Static_2Level_PolVsNonPol': '2-level group effect (Difference Score)',
        'StaticOnly_3Level': '3-level group effect (Static Period)',
        'StaticOnly_2Level_PolVsNonPol': '2-level group effect (Static Period)',
        'Baseline_3Level': '3-level group effect (Baseline Period)',
        'Baseline_2Level_PolVsNonPol': '2-level group effect (Baseline Period)'
    }
    for k, v in prefix_map.items():
        if prefix.startswith(k): return v
    return prefix.replace('_', ' ')


# --- Shared Model Testing Helper ---
def evaluate_and_fit_best_model(df, target_var, formula_base, model_prefix, subfolder="general"):
    """Evaluates 4 model structures, exports fits, and returns the best fitted unstandardized and standardized models."""
    out_dir = REGRESSION_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_test = {
        f"M1 (GLM Base)": {"type": "ols", "formula": f"{target_var} ~ {formula_base}"},
        f"M2 (GLM + Trial)": {"type": "ols", "formula": f"{target_var} ~ {formula_base} + Trial_Target"},
        f"M3 (LMM Base)": {"type": "mixedlm", "formula": f"{target_var} ~ {formula_base}", "groups": "Subject"},
        f"M4 (LMM + Trial)": {"type": "mixedlm", "formula": f"{target_var} ~ {formula_base} + Trial_Target", "groups": "Subject"}
    }

    fit_results = []
    for name, params in models_to_test.items():
        try:
            if params["type"] == "ols":
                mod = smf.ols(params["formula"], data=df).fit()
            else:
                mod = smf.mixedlm(params["formula"], data=df, groups=df[params["groups"]]).fit(reml=False)
            fit_results.append({'Model': name, 'AIC': mod.aic, 'BIC': mod.bic, 'LogLik': mod.llf})
        except Exception:
            fit_results.append({'Model': name, 'AIC': np.inf, 'BIC': np.inf, 'LogLik': np.nan})

    df_fits = pd.DataFrame(fit_results).sort_values(by='AIC').reset_index(drop=True)
    
    # Export Fits to dynamic subfolder
    df_fits.to_csv(out_dir / f'Model_Fit_Indices_{model_prefix}.csv', index=False)
    name_tex = model_prefix.replace('_', '\\_')
    tex_fits = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Model Fit Comparison ({name_tex})}}\n"
        f"\\label{{tab:model_fits_{model_prefix}}}\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\textbf{Model} & \\textbf{AIC} & \\textbf{BIC} & \\textbf{Log-Likelihood} \\\\\n\\midrule\n"
    )
    for _, row in df_fits.iterrows():
        tex_fits += f"{row['Model']} & {row['AIC']:.2f} & {row['BIC']:.2f} & {row['LogLik']:.2f} \\\\\n"
    tex_fits += "\\bottomrule\n\\end{tabular}\n\\end{table}"
    with open(out_dir / f'Model_Fit_Indices_{model_prefix}.tex', 'w') as f:
        f.write(tex_fits)

    best_model_name = df_fits.iloc[0]['Model']
    config = models_to_test[best_model_name]
    print(f"  [{model_prefix}] Best Fit: {best_model_name}")
    
    # Create Z-scored dataframe for standardized betas
    df_z = df.copy()
    df_z[target_var] = (df_z[target_var] - df_z[target_var].mean()) / df_z[target_var].std()
    if 'Trial_Target' in config['formula'] and 'Trial_Target' in df_z.columns:
        df_z['Trial_Target'] = (df_z['Trial_Target'] - df_z['Trial_Target'].mean()) / df_z['Trial_Target'].std()
    
    # Fit Final Models
    if config['type'] == 'mixedlm':
        final_mod = smf.mixedlm(config["formula"], data=df, groups=df[config["groups"]]).fit(reml=True)
        final_mod_z = smf.mixedlm(config["formula"], data=df_z, groups=df_z[config["groups"]]).fit(reml=True)
    else:
        final_mod = smf.ols(config["formula"], data=df).fit()
        final_mod_z = smf.ols(config["formula"], data=df_z).fit()
        
    return final_mod, final_mod_z, config["formula"]

def _get_fe_df(model, model_z, model_prefix):
    """Internal helper to extract and format a unified dataframe of fixed effects."""
    res = pd.DataFrame({
        'Model_Type': model_prefix,
        'Predictor': model.params.index,
        'Estimate': model.params.values,
        'Beta': model_z.params.values, # Grab Beta from the Z-scored model
        'SE': model.bse.values,
        't_z_value': model.tvalues.values if hasattr(model, 'tvalues') else model.params.values / model.bse.values,
        'p_value': model.pvalues.values,
        'CI_Lower': model.conf_int()[0].values,
        'CI_Upper': model.conf_int()[1].values
    })
    res_clean = res[~res['Predictor'].str.contains('Var')].copy()
    res_clean['Predictor_Clean'] = res_clean['Predictor'].apply(pretty_term)
    res_clean['b'] = res_clean['Estimate'].apply(fmt3)
    res_clean['beta'] = res_clean['Beta'].apply(fmt2) 
    res_clean['SE_fmt'] = res_clean['SE'].apply(fmt3)
    res_clean['95% CI'] = res_clean.apply(lambda r: ci_str(r['CI_Lower'], r['CI_Upper'], 2), axis=1)
    res_clean['p_fmt'] = res_clean['p_value'].apply(fmt_p_tex)
    return res_clean

def export_fe_df(res_clean, model_prefix, subfolder="general"):
    """Internal helper to print, format to CSV, and export to TeX."""
    out_dir = REGRESSION_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_export = res_clean[['Predictor_Clean', 'b', 'SE_fmt', 't_z_value', 'p_value', 'beta', '95% CI']].rename(
        columns={'Predictor_Clean': 'Predictor', 'SE_fmt': 'SE', 't_z_value': 't', 'p_value': 'p'}
    )
    
    print(f"\n[Fixed Effects: {model_prefix}]")
    print(csv_export[['Predictor', 'b', 'SE', 't', 'p', 'beta']].to_string(index=False))

    csv_export.to_csv(out_dir / f'Fixed_Effects_{model_prefix}.csv', index=False)
    
    tex_rows = ""
    for _, row in res_clean.iterrows():
        t_val = fmt2(row['t_z_value'])
        tex_rows += f"{row['Predictor_Clean']} & {row['b']} & {row['SE_fmt']} & {t_val} & {row['p_fmt']} & {row['beta']} & {row['95% CI']} \\\\\n"

    model_context = get_model_context(model_prefix)
    tex_fixed = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Fixed effects from the final model predicting FAA ({model_context})}}\n"
        f"\\label{{tab:fixed_effects_{model_prefix}}}\n"
        "\\begin{threeparttable}\n"
        "\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.3] S[table-format=2.2] l S[table-format=1.2] l@{}}\n\\toprule\n"
        f"\\textbf{{Predictor}} & \\textbf{{$b$}} & \\textbf{{SE}} & \\textbf{{$t$}} & \\textbf{{$p$}} & \\textbf{{$\\beta$}} & \\textbf{{95\\% CI}} \\\\\n\\midrule\n"
        f"{tex_rows}"
        "\\bottomrule\n\\end{tabular}\n"
        f"\\begin{{tablenotes}}\n\\footnotesize\n\\item \\textit{{Note.}} Unstandardized ($b$) and standardized ($\\beta$) coefficients shown with 95\\% CIs.\n\\end{{tablenotes}}\n"
        "\\end{threeparttable}\n\\end{table}"
    )
    with open(out_dir / f'Fixed_Effects_{model_prefix}.tex', 'w') as f:
        f.write(tex_fixed)
        
    return res_clean

def extract_fixed_effects(model, model_z, model_prefix, subfolder="general"):
    """Standard single-model fixed effects extraction."""
    return export_fe_df(_get_fe_df(model, model_z, model_prefix), model_prefix, subfolder)

def extract_combined_fixed_effects(mod_base, mod_base_z, mod_alt, mod_alt_z, term_alt_raw, model_prefix, subfolder="general"):
    """Extracts FE from a base model and cleanly appends a specific missing term from an alternative model."""
    df_base = _get_fe_df(mod_base, mod_base_z, model_prefix)
    df_alt = _get_fe_df(mod_alt, mod_alt_z, model_prefix)
    
    alt_row = df_alt[df_alt['Predictor'] == term_alt_raw]
    combined = pd.concat([df_base, alt_row], ignore_index=True)
    return export_fe_df(combined, model_prefix, subfolder)

def compute_and_print_emms_contrasts(model, df_master, formula_str, pred_dict, model_name, subfolder="general"):
    """Calculates EMMs and Pairwise Contrasts via matrix multiplication, exports to CSV/TeX."""
    out_dir = PAIRWISE_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    combinations = list(itertools.product(*pred_dict.values()))
    df_pred = pd.DataFrame(combinations, columns=pred_dict.keys())
    
    if 'Trial_Target' in formula_str:
        df_pred['Trial_Target'] = [df_master['Trial_Target'].mean()] * len(df_pred)
        
    formula_rhs = formula_str.split('~')[1]
    X = dmatrix(formula_rhs, df_pred, return_type='dataframe')
    
    fe_cols = [col for col in model.params.index if col in X.columns]
    X, params, cov_matrix = X[fe_cols], model.params[fe_cols].values, model.cov_params().loc[fe_cols, fe_cols].values
    
    resid_var = model.scale if hasattr(model, 'scale') else model.mse_resid
    resid_sd = np.sqrt(resid_var)

    X_vals = X.values
    emms = X_vals.dot(params)
    se_emms = np.sqrt(np.diag(X_vals.dot(cov_matrix).dot(X_vals.T)))

    emms_export = df_pred.copy()
    if 'Trial_Target' in emms_export: emms_export.drop(columns=['Trial_Target'], inplace=True)
    emms_export['EMM_Val'] = emms
    emms_export['SE'] = se_emms

    # Export to dynamic subfolder
    emms_export.to_csv(out_dir / f'EMMs_{model_name}.csv', index=False)
    
    contrast_results = []
    n_conds = len(df_pred)
    
    for i, j in itertools.combinations(range(n_conds), 2):
        cond_cols = [col for col in pred_dict.keys() if col != 'Trial_Target']
        cond_a = "_".join([str(df_pred.iloc[i][col]) for col in cond_cols])
        cond_b = "_".join([str(df_pred.iloc[j][col]) for col in cond_cols])
        
        c_vec = X_vals[i] - X_vals[j]
        diff = c_vec.dot(params)
        se_diff = np.sqrt(c_vec.dot(cov_matrix).dot(c_vec.T))
        
        z_stat = diff / se_diff
        p_raw = 2 * stats.norm.sf(np.abs(z_stat))
        ci_low, ci_high = diff - (1.96 * se_diff), diff + (1.96 * se_diff)
        
        contrast_results.append({
            'Model': model_name, 'Cond_A': cond_a, 'Cond_B': cond_b, 
            'Estimate': diff, 'SE': se_diff, 'z_stat': z_stat, 'd': diff / resid_sd, 
            'CI_low': ci_low, 'CI_high': ci_high, 'p_raw': p_raw
        })

    df_contrasts = pd.DataFrame(contrast_results)
    _, p_holm, _, _ = multipletests(df_contrasts['p_raw'], method='holm')
    df_contrasts['p_holm'] = p_holm
    
    # Export to dynamic subfolder
    df_contrasts.to_csv(out_dir / f'Contrasts_{model_name}.csv', index=False)
    
    print(f"\n--- Pairwise Contrasts: {model_name} ---")
    print(f"{'Cond A':<15} | {'Cond B':<15} | {'Delta':<8} | {'Cohen d':<8} | {'z-stat':<8} | {'p_raw':<8} | {'p_Holm':<8}")
    print("-" * 90)
    for _, row in df_contrasts.iterrows():
        sig = "***" if row['p_holm'] < 0.001 else "**" if row['p_holm'] < 0.01 else "*" if row['p_holm'] < 0.05 else "." if row['p_holm'] < 0.10 else ""
        print(f"{row['Cond_A']:<15} | {row['Cond_B']:<15} | {row['Estimate']:>8.3f} | {row['d']:>8.2f} | {row['z_stat']:>8.2f} | {row['p_raw']:>8.3f} | {row['p_holm']:>8.3f} {sig}")
            
    return df_contrasts


# %% Cell 3: Targeted Analyses (Static Face - Baseline Score | Emotions Combined)
print("\n=============================================================================================")
print("TARGETED ANALYSES: Difference Score (Static Face - Baseline) | All Emotions")
print("=============================================================================================")

# 1. Prepare Base Data for the Target Metric
df_target = df_trials.copy()

if USE_Z_SCORES:
    df_target['FAA_Target'] = (df_target['FAA_Diff'] - df_target['FAA_Diff'].mean()) / df_target['FAA_Diff'].std()
    df_target['Trial_Target'] = (df_target['Trial'] - df_target['Trial'].mean()) / df_target['Trial'].std()
else:
    df_target['FAA_Target'] = df_target['FAA_Diff']
    df_target['Trial_Target'] = df_target['Trial']

# -----------------------------------------------------------------------------
# 1) 3-Level Group Analysis (In-Party vs. Out-Party vs. Non-Politician)
# -----------------------------------------------------------------------------
print("\n[MODEL 1] 3-Level Group Effect (All Emotions Combined)")

factors_group = {'Group': ['In-Party', 'Out-Party', 'Non-Politician']}
form_ref_np = "C(Group, Treatment('Non-Politician'))"
form_ref_ip = "C(Group, Treatment('In-Party'))"

# Fit Models (Unpacking 3 values and adding subfolder)
mod_np, mod_np_z, f_np = evaluate_and_fit_best_model(df_target, 'FAA_Target', form_ref_np, "Static_3Level", subfolder="groupwise")
mod_ip, mod_ip_z, f_ip = evaluate_and_fit_best_model(df_target, 'FAA_Target', form_ref_ip, "Static_3Level_RefIP", subfolder="groupwise")

# Extract Combined Fixed Effects (Passing the z-scored models)
term_alt_raw = "C(Group, Treatment('In-Party'))[T.Out-Party]"
extract_combined_fixed_effects(mod_np, mod_np_z, mod_ip, mod_ip_z, term_alt_raw, "Static_3Level", subfolder="groupwise")

# Print & Export Pairwise Contrasts
compute_and_print_emms_contrasts(mod_np, df_target, f_np, factors_group, "Static_3Level_Contrasts", subfolder="groupwise")

# -----------------------------------------------------------------------------
# 2) 2-Level Group Analysis (Politician vs. Non-Politician)
# -----------------------------------------------------------------------------
print("\n[MODEL 2] 2-Level Politician vs Non-Politician (All Emotions Combined)")

formula_pol = "C(Pol_Category, Treatment('Non-Politician'))"
factors_pol = {'Pol_Category': ['Non-Politician', 'Politician']}

# Fit Models (Unpacking 3 values and adding subfolder)
mod_pol, mod_pol_z, form_pol = evaluate_and_fit_best_model(df_target, 'FAA_Target', formula_pol, 'Static_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")
extract_fixed_effects(mod_pol, mod_pol_z, 'Static_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")
compute_and_print_emms_contrasts(mod_pol, df_target, form_pol, factors_pol, 'Static_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")


# analysing static period by itself without baselining
############################################################################################
# %% Cell 1: Single-Trial Data Extraction & Trial-Level Baselining
import os
import time
import mne
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
import scipy.stats as stats
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix
from pathlib import Path

# Suppress convergence warnings for clean output
warnings.filterwarnings("ignore", module="statsmodels")

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
FAA_METHOD = "static"  # Changed to route outputs to new 'static' folder
EXCLUSION_CRITERION = "Strict"  

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives'
PROCESSED_DIR = DERIVATIVES_DIR / 'processed' / BATCH_NAME

# Output Directories (Automatically uses the new FAA_METHOD folder name)
REGRESSION_DIR = DERIVATIVES_DIR / 'regression' / BATCH_NAME / FAA_METHOD
PAIRWISE_DIR = DERIVATIVES_DIR / 'pairwise' / BATCH_NAME / FAA_METHOD
FIGURES_DIR = DERIVATIVES_DIR / 'figures' / BATCH_NAME / FAA_METHOD  

REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR = DERIVATIVES_DIR / 'quality_check' / BATCH_NAME

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

# Base Event Dictionaries
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

def parse_event_string(ev_str):
    group = "Non-Politician" if "nonpol" in ev_str else "In-Party" if "inparty" in ev_str else "Out-Party"
    emotion = "Angry" if "angry" in ev_str else "Happy" if "happy" in ev_str else "Neutral"
    return group, emotion

print("==================================================")
print(f"Extracting Single-Trial Data from {BATCH_NAME}...")
print("==================================================")

subject_folders = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
all_trials = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    eeg_dir = subj_dir / 'eeg'
    
    # Try diff or fallback static files to load epochs
    epo_file = eeg_dir / f"{subject}_task-emotion_faa_diff-epo.fif" 
    if not epo_file.exists():
        epo_file = eeg_dir / f"{subject}_task-emotion_faa_static_baseline-epo.fif"
        if not epo_file.exists():
            continue
        
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Define Target Time Mask for Static Face 
    t_static_face = (epochs.times >= 0.800) & (epochs.times <= 1.900)

    faa_data = epochs.get_data(picks='FAA')[:, 0, :]
    
    for i, event_id in enumerate(epochs.events[:, 2]):
        event_name = ID_TO_EVENT.get(event_id)
        if not event_name: continue
            
        group, emotion = parse_event_string(event_name)
        
        # Create the binary grouping variable (Politician vs Non-Politician)
        pol_category = "Non-Politician" if group == "Non-Politician" else "Politician"
        
        trial_data = faa_data[i, :]
        
        # Calculate Phase Mean for Static Face Window (No baseline subtraction)
        val_static = np.mean(trial_data[t_static_face])
        
        all_trials.append({
            'Subject': subject,
            'Trial': i + 1,
            'Group': group,
            'Pol_Category': pol_category,
            'Emotion': emotion,
            'Phase_Static': val_static
        })

df_trials = pd.DataFrame(all_trials)
print(f"Extraction complete: {len(df_trials)} total single trials formatted.")


# %% Cell 2: Model Testing & Pairwise Tools
print("\n=============================================================================================")

# TOGGLE STANDARDIZATION 
USE_Z_SCORES = False



# %% Cell 3: Targeted Analyses (Static Face Period | Emotions Combined)
print("\n=============================================================================================")
print("TARGETED ANALYSES: Static Face Period | All Emotions")
print("=============================================================================================")

# 1. Prepare Base Data for the Target Metric
df_target = df_trials.copy()

if USE_Z_SCORES:
    df_target['FAA_Target'] = (df_target['Phase_Static'] - df_target['Phase_Static'].mean()) / df_target['Phase_Static'].std()
    df_target['Trial_Target'] = (df_target['Trial'] - df_target['Trial'].mean()) / df_target['Trial'].std()
else:
    df_target['FAA_Target'] = df_target['Phase_Static']
    df_target['Trial_Target'] = df_target['Trial']

# -----------------------------------------------------------------------------
# 1) 3-Level Group Analysis (In-Party vs. Out-Party vs. Non-Politician)
# -----------------------------------------------------------------------------
print("\n[MODEL 1] 3-Level Group Effect (All Emotions Combined)")

factors_group = {'Group': ['In-Party', 'Out-Party', 'Non-Politician']}
form_ref_np = "C(Group, Treatment('Non-Politician'))"
form_ref_ip = "C(Group, Treatment('In-Party'))"

# Fit Models
mod_np, mod_np_z, f_np = evaluate_and_fit_best_model(df_target, 'FAA_Target', form_ref_np, "StaticOnly_3Level", subfolder="groupwise")
mod_ip, mod_ip_z, f_ip = evaluate_and_fit_best_model(df_target, 'FAA_Target', form_ref_ip, "StaticOnly_3Level_RefIP", subfolder="groupwise")

# Extract Combined Fixed Effects
term_alt_raw = "C(Group, Treatment('In-Party'))[T.Out-Party]"
extract_combined_fixed_effects(mod_np, mod_np_z, mod_ip, mod_ip_z, term_alt_raw, "StaticOnly_3Level", subfolder="groupwise")

# Print & Export Pairwise Contrasts
compute_and_print_emms_contrasts(mod_np, df_target, f_np, factors_group, "StaticOnly_3Level_Contrasts", subfolder="groupwise")

# -----------------------------------------------------------------------------
# 2) 2-Level Group Analysis (Politician vs. Non-Politician)
# -----------------------------------------------------------------------------
print("\n[MODEL 2] 2-Level Politician vs Non-Politician (All Emotions Combined)")

formula_pol = "C(Pol_Category, Treatment('Non-Politician'))"
factors_pol = {'Pol_Category': ['Non-Politician', 'Politician']}

mod_pol, mod_pol_z, form_pol = evaluate_and_fit_best_model(df_target, 'FAA_Target', formula_pol, 'StaticOnly_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")
extract_fixed_effects(mod_pol, mod_pol_z, 'StaticOnly_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")
compute_and_print_emms_contrasts(mod_pol, df_target, form_pol, factors_pol, 'StaticOnly_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")

# analysing baseline period by itself
############################################################################################
# %% Cell 1: Single-Trial Data Extraction (Baseline 0.0 to 0.8s)
import os
import time
import mne
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
import scipy.stats as stats
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix
from pathlib import Path

# Suppress convergence warnings for clean output
warnings.filterwarnings("ignore", module="statsmodels")

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
FAA_METHOD = "baseline"  # Changed to route outputs to new 'baseline' folder
EXCLUSION_CRITERION = "Strict"  

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives'
PROCESSED_DIR = DERIVATIVES_DIR / 'processed' / BATCH_NAME

# Output Directories (Automatically uses the new FAA_METHOD folder name)
REGRESSION_DIR = DERIVATIVES_DIR / 'regression' / BATCH_NAME / FAA_METHOD
PAIRWISE_DIR = DERIVATIVES_DIR / 'pairwise' / BATCH_NAME / FAA_METHOD
FIGURES_DIR = DERIVATIVES_DIR / 'figures' / BATCH_NAME / FAA_METHOD  

REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR = DERIVATIVES_DIR / 'quality_check' / BATCH_NAME

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

# Base Event Dictionaries
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

def parse_event_string(ev_str):
    group = "Non-Politician" if "nonpol" in ev_str else "In-Party" if "inparty" in ev_str else "Out-Party"
    emotion = "Angry" if "angry" in ev_str else "Happy" if "happy" in ev_str else "Neutral"
    return group, emotion

print("==================================================")
print(f"Extracting Single-Trial Data from {BATCH_NAME} (Baseline)...")
print("==================================================")

subject_folders = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
all_trials = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    eeg_dir = subj_dir / 'eeg'
    
    # Try diff or fallback static files to load epochs
    epo_file = eeg_dir / f"{subject}_task-emotion_faa_diff-epo.fif" 
    if not epo_file.exists():
        epo_file = eeg_dir / f"{subject}_task-emotion_faa_static_baseline-epo.fif"
        if not epo_file.exists():
            continue
        
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Define Target Time Mask for Baseline (0.0 to 0.8s)
    t_baseline = (epochs.times >= 0.0) & (epochs.times <= 0.800)

    faa_data = epochs.get_data(picks='FAA')[:, 0, :]
    
    for i, event_id in enumerate(epochs.events[:, 2]):
        event_name = ID_TO_EVENT.get(event_id)
        if not event_name: continue
            
        group, emotion = parse_event_string(event_name)
        
        # Create the binary grouping variable (Politician vs Non-Politician)
        pol_category = "Non-Politician" if group == "Non-Politician" else "Politician"
        
        trial_data = faa_data[i, :]
        
        # Calculate Phase Mean for Baseline Window (No subtraction)
        val_baseline = np.mean(trial_data[t_baseline])
        
        all_trials.append({
            'Subject': subject,
            'Trial': i + 1,
            'Group': group,
            'Pol_Category': pol_category,
            'Emotion': emotion,
            'Phase_Baseline': val_baseline
        })

df_trials = pd.DataFrame(all_trials)
print(f"Extraction complete: {len(df_trials)} total single trials formatted.")


# %% Cell 2: Model Testing & Pairwise Tools
print("\n=============================================================================================")

# TOGGLE STANDARDIZATION 
USE_Z_SCORES = False



# %% Cell 3: Targeted Analyses (Baseline Period | Emotions Combined)
print("\n=============================================================================================")
print("TARGETED ANALYSES: Baseline Period (0.0 - 0.8s) | All Emotions")
print("=============================================================================================")

# 1. Prepare Base Data for the Target Metric
df_target = df_trials.copy()

if USE_Z_SCORES:
    df_target['FAA_Target'] = (df_target['Phase_Baseline'] - df_target['Phase_Baseline'].mean()) / df_target['Phase_Baseline'].std()
    df_target['Trial_Target'] = (df_target['Trial'] - df_target['Trial'].mean()) / df_target['Trial'].std()
else:
    df_target['FAA_Target'] = df_target['Phase_Baseline']
    df_target['Trial_Target'] = df_target['Trial']

# -----------------------------------------------------------------------------
# 1) 3-Level Group Analysis (In-Party vs. Out-Party vs. Non-Politician)
# -----------------------------------------------------------------------------
print("\n[MODEL 1] 3-Level Group Effect (All Emotions Combined)")

factors_group = {'Group': ['In-Party', 'Out-Party', 'Non-Politician']}
form_ref_np = "C(Group, Treatment('Non-Politician'))"
form_ref_ip = "C(Group, Treatment('In-Party'))"

# Fit Models
mod_np, mod_np_z, f_np = evaluate_and_fit_best_model(df_target, 'FAA_Target', form_ref_np, "Baseline_3Level", subfolder="groupwise")
mod_ip, mod_ip_z, f_ip = evaluate_and_fit_best_model(df_target, 'FAA_Target', form_ref_ip, "Baseline_3Level_RefIP", subfolder="groupwise")

# Extract Combined Fixed Effects
term_alt_raw = "C(Group, Treatment('In-Party'))[T.Out-Party]"
extract_combined_fixed_effects(mod_np, mod_np_z, mod_ip, mod_ip_z, term_alt_raw, "Baseline_3Level", subfolder="groupwise")

# Print & Export Pairwise Contrasts
compute_and_print_emms_contrasts(mod_np, df_target, f_np, factors_group, "Baseline_3Level_Contrasts", subfolder="groupwise")

# -----------------------------------------------------------------------------
# 2) 2-Level Group Analysis (Politician vs. Non-Politician)
# -----------------------------------------------------------------------------
print("\n[MODEL 2] 2-Level Politician vs Non-Politician (All Emotions Combined)")

formula_pol = "C(Pol_Category, Treatment('Non-Politician'))"
factors_pol = {'Pol_Category': ['Non-Politician', 'Politician']}

mod_pol, mod_pol_z, form_pol = evaluate_and_fit_best_model(df_target, 'FAA_Target', formula_pol, 'Baseline_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")
extract_fixed_effects(mod_pol, mod_pol_z, 'Baseline_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")
compute_and_print_emms_contrasts(mod_pol, df_target, form_pol, factors_pol, 'Baseline_2Level_PolVsNonPol', subfolder="nonpol_vs_pol")
#%%