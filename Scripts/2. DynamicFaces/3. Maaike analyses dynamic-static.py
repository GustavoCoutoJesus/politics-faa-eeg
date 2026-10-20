# %% #######################################################################################################################
# The goal of this script is to run core FAA regressions: Dynamic-Static
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
FAA_METHOD = "diff"
EXCLUSION_CRITERION = "Strict"  # USER SELECTABLE: "Strict", "Lenient", or "None"

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives'
PROCESSED_DIR = DERIVATIVES_DIR / 'processed' / BATCH_NAME

# Output Directories
REGRESSION_DIR = DERIVATIVES_DIR / 'regression' / BATCH_NAME / FAA_METHOD
PAIRWISE_DIR = DERIVATIVES_DIR / 'pairwise' / BATCH_NAME / FAA_METHOD
FIGURES_DIR = DERIVATIVES_DIR / 'figures' / BATCH_NAME / FAA_METHOD  # New Figures Directory

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
    epo_file = eeg_dir / f"{subject}_task-emotion_faa_{FAA_METHOD}-epo.fif" 

    if not epo_file.exists():
        continue
        
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Define Target Time Masks
    t_static = (epochs.times >= 0.800) & (epochs.times <= 1.900)
    t_dynamic = (epochs.times >= 1.950) & (epochs.times <= 4.000)

    # For a later phase-based analysis
    t_baseline = (epochs.times >= 0.000) & (epochs.times <= 0.800)

    faa_data = epochs.get_data(picks='FAA')[:, 0, :]
    
    for i, event_id in enumerate(epochs.events[:, 2]):
        event_name = ID_TO_EVENT.get(event_id)
        if not event_name: continue
            
        group, emotion = parse_event_string(event_name)
        
        # Create the binary grouping variable (Politician vs Non-Politician)
        pol_category = "Non-Politician" if group == "Non-Politician" else "Politician"
        
        trial_data = faa_data[i, :]
        
        # Calculate Phase Means
        val_static = np.mean(trial_data[t_static])
        val_dynamic = np.mean(trial_data[t_dynamic])

        # For later analysis 
        val_baseline = np.mean(trial_data[t_baseline])
        
        # Difference score (Dynamic - Static)
        diff_val = val_dynamic - val_static
        
        all_trials.append({
            'Subject': subject,
            'Trial': i + 1,
            'Group': group,
            'Pol_Category': pol_category,
            'Emotion': emotion,
            'Condition': f"{emotion} {group}",
            'Phase_Static': val_static,
            'Phase_Dynamic': val_dynamic,
            'Phase_Baseline': val_baseline, # For later analysis
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
    
    # Binary Politician vs Non-Politician #JUST FOR EXPLORATORY PURPOSES, NOT USED
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
    
    # Phase Main Effects
    t = t.replace("C(Phase, Treatment('Static'))[T.Dynamic]", "Phase (Static vs. Dynamic)")
    t = t.replace("C(Phase, Treatment('Baseline'))[T.Dynamic]", "Phase (Baseline vs. Dynamic)")
    
    return t

def get_model_context(prefix):
    """Generates clear descriptions of the analytical subset."""
    prefix_map = {
        'Mod1_AllFaces': 'Overall analysis combining all emotion conditions',
        'Mod2_AngryOnly': 'Subset analysis restricted to angry faces',
        'Mod3_NeutralOnly': 'Subset analysis restricted to neutral faces',
        'Mod4_HappyOnly': 'Subset analysis restricted to happy faces',
        'Overall_AllEmotions': 'Overall analysis combining all emotions across political groups',
        'Angry_Only': 'Subset analysis restricted to angry faces across political groups',
        'Neutral_Only': 'Subset analysis restricted to neutral faces across political groups',
        'Happy_Only': 'Subset analysis restricted to happy faces across political groups',
        'Overall_Emotion': 'Analysis of emotion effects restricted to Non-politician faces',
        'StaticDynamic_MainEffect': 'Main effect of task phase (Static vs. Dynamic)',
        'BaselineDynamic_MainEffect': 'Main effect of task phase (Baseline vs. Dynamic)'
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
        't_z_value': model.tvalues.values,
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
    
    cond_cols = [col for col in pred_dict.keys() if col != 'Trial_Target']
    emms_export['Condition'] = emms_export[cond_cols].astype(str).agg(' / '.join, axis=1).str.replace('_', ' ')

    csv_emm = pd.DataFrame({
        'Condition': emms_export['Condition'],
        'M': emms_export['EMM_Val'].apply(fmt2),
        'SE': emms_export['SE'].apply(fmt3)
    })

    # Export to dynamic subfolder
    emms_export.to_csv(out_dir / f'EMMs_{model_name}.csv', index=False)
    
    model_context = get_model_context(model_name)
    tex_emms = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Estimated marginal means (EMMs) and SEs for FAA ({model_context})}}\n"
        f"\\label{{tab:emms_{model_name}}}\n"
        "\\begin{threeparttable}\n"
        "\\begin{tabular}{@{}l S[table-format=1.2] S[table-format=1.3]@{}}\n\\toprule\n"
        "\\textbf{Condition} & \\textbf{M} & \\textbf{SE} \\\\\n\\midrule\n"
    )
    for _, row in csv_emm.iterrows():
        tex_emms += f"{row['Condition']} & {row['M']} & {row['SE']} \\\\\n"
    tex_emms += (
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n\\item \\textit{Note.} Values are Estimated Marginal Means (EMMs) with Standard Errors (SEs), taken from the best-fitting model.\n\\end{tablenotes}\n"
        "\\end{threeparttable}\n\\end{table}"
    )
    with open(out_dir / f'EMMs_{model_name}.tex', 'w') as f: f.write(tex_emms)

    # Calculate Contrasts
    contrast_results = []
    n_conds = len(df_pred)
    
    for i, j in itertools.combinations(range(n_conds), 2):
        cond_a = " ".join([str(df_pred.iloc[i][col]) for col in cond_cols]).replace('_', ' ')
        cond_b = " ".join([str(df_pred.iloc[j][col]) for col in cond_cols]).replace('_', ' ')
        
        c_vec = X_vals[i] - X_vals[j]
        diff = c_vec.dot(params)
        se_diff = np.sqrt(c_vec.dot(cov_matrix).dot(c_vec.T))
        
        z_stat = diff / se_diff
        p_raw = 2 * stats.norm.sf(np.abs(z_stat))
        ci_low, ci_high = diff - (1.96 * se_diff), diff + (1.96 * se_diff)
        
        contrast_results.append({
            'Model': model_name, 'Cond A': cond_a, 'Cond B': cond_b, 
            'Estimate': diff, 'SE': se_diff, 'd': diff / resid_sd, 
            'CI_low': ci_low, 'CI_high': ci_high, 'p_raw': p_raw
        })

    df_contrasts = pd.DataFrame(contrast_results)
    _, p_holm, _, _ = multipletests(df_contrasts['p_raw'], method='holm')
    df_contrasts['p_holm'] = p_holm
    
    csv_contrasts = pd.DataFrame({
        'Cond A': df_contrasts['Cond A'],
        'Cond B': df_contrasts['Cond B'],
        'Delta': df_contrasts['Estimate'].apply(fmt2),
        'd': df_contrasts['d'].apply(fmt2),
        '95% CI': df_contrasts.apply(lambda r: ci_str(r['CI_low'], r['CI_high'], 2), axis=1),
        'p_raw': df_contrasts['p_raw'].apply(fmt_p_tex),
        'p_Holm': df_contrasts['p_holm'].apply(fmt_p_tex)
    })

    # Export to dynamic subfolder
    df_contrasts.to_csv(out_dir / f'Contrasts_{model_name}.csv', index=False)
    tex_contrasts = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Pairwise contrasts of FAA (Holm-corrected) ({model_context})}}\n"
        f"\\label{{tab:contrasts_{model_name}}}\n"
        "\\begin{threeparttable}\n"
        "\\begin{tabular}{@{}ll S[table-format=1.2] S[table-format=1.2] l l l@{}}\n\\toprule\n"
        "\\textbf{Cond A} & \\textbf{Cond B} & \\textbf{$\\Delta$} & \\textbf{d} & \\textbf{95\\% CI} & \\textbf{$p_\\text{raw}$} & \\textbf{$p_\\text{Holm}$} \\\\\n\\midrule\n"
    )
    for _, row in csv_contrasts.iterrows():
        tex_contrasts += f"{row['Cond A']} & {row['Cond B']} & {row['Delta']} & {row['d']} & {row['95% CI']} & {row['p_raw']} & {row['p_Holm']} \\\\\n"
    tex_contrasts += (
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n\\item \\textit{Note.} Estimates are provided as mean differences ($\\Delta$) and standardized Cohen's d.\n\\end{tablenotes}\n"
        "\\end{threeparttable}\n\\end{table}"
    )
    with open(out_dir / f'Contrasts_{model_name}.tex', 'w') as f: f.write(tex_contrasts)
    
    print(f"\n--- Pairwise Contrasts: {model_name} ---")
    print(f"{'Cond A':<15} | {'Cond B':<15} | {'Delta':<8} | {'Cohen d':<8} | {'p_raw':<8} | {'p_Holm':<8}")
    print("-" * 80)
    for _, row in df_contrasts.iterrows():
        sig = "***" if row['p_holm'] < 0.001 else "**" if row['p_holm'] < 0.01 else "*" if row['p_holm'] < 0.05 else "." if row['p_holm'] < 0.10 else ""
        print(f"{row['Cond A']:<15} | {row['Cond B']:<15} | {row['Estimate']:>8.3f} | {row['d']:>8.2f} | {row['p_raw']:>8.3f} | {row['p_holm']:>8.3f} {sig}")
            
    return df_contrasts

#%% Absolute EMM FAA values from dynamic phase (added later)
print("\n=============================================================================================")
print("TARGETED ANALYSES: Dynamic Phase Only (No Baselining)")
print("=============================================================================================")

# 1. Copy the trial data and set the raw dynamic phase as the target
df_dyn = df_trials.copy()

# Toggle standardization (matches your existing script logic)
if USE_Z_SCORES:
    df_dyn['FAA_Target'] = (df_dyn['Phase_Dynamic'] - df_dyn['Phase_Dynamic'].mean()) / df_dyn['Phase_Dynamic'].std()
else:
    df_dyn['FAA_Target'] = df_dyn['Phase_Dynamic']

# 2. Define the factors and formula for the full model
factors_all = {
    'Group': ['In-Party', 'Out-Party', 'Non-Politician'],
    'Emotion': ['Angry', 'Neutral', 'Happy']
}

# Base formula examining the interaction of Group and Emotion
formula_all = "C(Group, Treatment('Non-Politician')) * C(Emotion, Treatment('Neutral'))"

# 3. Fit the model using your existing evaluation function
mod_dyn, mod_dyn_z, form_dyn = evaluate_and_fit_best_model(
    df_dyn, 
    'FAA_Target', 
    formula_all, 
    'Dynamic_Only_Overall', 
    subfolder="dynamic_only"
)

# 4. Extract and export the EMMs (and contrasts) using your existing function
compute_and_print_emms_contrasts(
    mod_dyn, 
    df_dyn, 
    form_dyn, 
    factors_all, 
    'Dynamic_Only_Overall', 
    subfolder="dynamic_only"
)


# %% Cell 3: Targeted Non-Pol vs Pol Analyses (Difference Scores) #DEPRECATED 
print("\n=============================================================================================")
print("TARGETED ANALYSES: Non-Politician vs. Politician (Difference Score)")
print("=============================================================================================")

# 1. Prepare Base Data
df_diff = df_trials.copy()

if USE_Z_SCORES:
    df_diff['FAA_Target'] = (df_diff['FAA_Diff'] - df_diff['FAA_Diff'].mean()) / df_diff['FAA_Diff'].std()
    df_diff['Trial_Target'] = (df_diff['Trial'] - df_diff['Trial'].mean()) / df_diff['Trial'].std()
else:
    df_diff['FAA_Target'] = df_diff['FAA_Diff']
    df_diff['Trial_Target'] = df_diff['Trial']

# Common model definitions for these tasks
formula_target = "C(Pol_Category, Treatment('Non-Politician'))"
factors_target = {'Pol_Category': ['Non-Politician', 'Politician']}

# -----------------------------------------------------------------------------
# 1) Non-Pol vs Pol (Across All Faces)
# -----------------------------------------------------------------------------
print("\n[1] Overall: Non-Pol vs Pol (All Emotions)")
mod_1, mod_1_z, form_1 = evaluate_and_fit_best_model(df_diff, 'FAA_Target', formula_target, 'Mod1_AllFaces', subfolder="nonpol_vs_pol")
extract_fixed_effects(mod_1, mod_1_z, 'Mod1_AllFaces', subfolder="nonpol_vs_pol")
compute_and_print_emms_contrasts(mod_1, df_diff, form_1, factors_target, 'Mod1_AllFaces', subfolder="nonpol_vs_pol")

# -----------------------------------------------------------------------------
# 2) Non-Pol vs Pol (Angry Faces Only)
# -----------------------------------------------------------------------------
print("\n[2] Angry Subset: Non-Pol vs Pol")
df_angry = df_diff[df_diff['Emotion'] == 'Angry'].copy()
mod_2, mod_2_z, form_2 = evaluate_and_fit_best_model(df_angry, 'FAA_Target', formula_target, 'Mod2_AngryOnly', subfolder="nonpol_vs_pol")
extract_fixed_effects(mod_2, mod_2_z, 'Mod2_AngryOnly', subfolder="nonpol_vs_pol")
compute_and_print_emms_contrasts(mod_2, df_angry, form_2, factors_target, 'Mod2_AngryOnly', subfolder="nonpol_vs_pol")

# -----------------------------------------------------------------------------
# 3) Non-Pol vs Pol (Neutral Faces Only)
# -----------------------------------------------------------------------------
print("\n[3] Neutral Subset: Non-Pol vs Pol")
df_neu = df_diff[df_diff['Emotion'] == 'Neutral'].copy()
mod_3, mod_3_z, form_3 = evaluate_and_fit_best_model(df_neu, 'FAA_Target', formula_target, 'Mod3_NeutralOnly', subfolder="nonpol_vs_pol")
extract_fixed_effects(mod_3, mod_3_z, 'Mod3_NeutralOnly', subfolder="nonpol_vs_pol")
compute_and_print_emms_contrasts(mod_3, df_neu, form_3, factors_target, 'Mod3_NeutralOnly', subfolder="nonpol_vs_pol")

# -----------------------------------------------------------------------------
# 4) Non-Pol vs Pol (Happy Faces Only)
# -----------------------------------------------------------------------------
print("\n[4] Happy Subset: Non-Pol vs Pol")
df_hap = df_diff[df_diff['Emotion'] == 'Happy'].copy()
mod_4, mod_4_z, form_4 = evaluate_and_fit_best_model(df_hap, 'FAA_Target', formula_target, 'Mod4_HappyOnly', subfolder="nonpol_vs_pol")
extract_fixed_effects(mod_4, mod_4_z, 'Mod4_HappyOnly', subfolder="nonpol_vs_pol")
compute_and_print_emms_contrasts(mod_4, df_hap, form_4, factors_target, 'Mod4_HappyOnly', subfolder="nonpol_vs_pol")


# %% Cell 4: Targeted 3-Level Group Analyses (Difference Scores) #CORE ANALYSES
print("\n=============================================================================================")
print("TARGETED ANALYSES: In-Party vs. Out-Party vs. Non-Politician (Difference Score)")
print("=============================================================================================")

factors_group = {'Group': ['In-Party', 'Out-Party', 'Non-Politician']}
form_ref_np = "C(Group, Treatment('Non-Politician'))"
form_ref_ip = "C(Group, Treatment('In-Party'))"

def run_3level_analysis(df_subset, subset_name):
    """Fits both reference models, neatly extracts one combined fixed effects table, and outputs contrasts."""
    print(f"\n\n#############################################################################")
    print(f"ANALYSIS SUBSET: {subset_name}")
    print(f"#############################################################################")

    mod_np, mod_np_z, f_np = evaluate_and_fit_best_model(df_subset, 'FAA_Target', form_ref_np, f"{subset_name}", subfolder="groupwise")
    mod_ip, mod_ip_z, f_ip = evaluate_and_fit_best_model(df_subset, 'FAA_Target', form_ref_ip, f"{subset_name}_RefIP", subfolder="groupwise")

    term_alt_raw = "C(Group, Treatment('In-Party'))[T.Out-Party]"
    extract_combined_fixed_effects(mod_np, mod_np_z, mod_ip, mod_ip_z, term_alt_raw, f"{subset_name}", subfolder="groupwise")

    compute_and_print_emms_contrasts(mod_np, df_subset, f_np, factors_group, f"{subset_name}_Contrasts", subfolder="groupwise")

# Execution Block
run_3level_analysis(df_diff, "Overall_AllEmotions")
run_3level_analysis(df_diff[df_diff['Emotion'] == 'Angry'].copy(), "Angry_Only")
run_3level_analysis(df_diff[df_diff['Emotion'] == 'Neutral'].copy(), "Neutral_Only")
run_3level_analysis(df_diff[df_diff['Emotion'] == 'Happy'].copy(), "Happy_Only")


# %% Cell 5: Emotion Analyses (Restricted to Non-Politicians) #CORE ANALYSIS
print("\n=============================================================================================")
print("TARGETED ANALYSES: Emotion Effects in Non-Politicians (Angry vs. Happy vs. Neutral)")
print("=============================================================================================")

factors_emotion = {'Emotion': ['Neutral', 'Angry', 'Happy']}
form_ref_neutral = "C(Emotion, Treatment('Neutral'))"
form_ref_angry = "C(Emotion, Treatment('Angry'))"

print(f"\n\n#############################################################################")
print(f"ANALYSIS SUBSET: Emotion Effects (Non-Politicians Only)")
print(f"#############################################################################")

# Filter dataset to only non-politician trials
df_diff_nonpol = df_diff[df_diff['Group'] == 'Non-Politician'].copy()

# 1. Fit Models (saving names preserved to prevent breaking imports)
mod_neu, mod_neu_z, f_neu = evaluate_and_fit_best_model(
    df_diff_nonpol, 'FAA_Target', form_ref_neutral, "Overall_Emotion", subfolder="emotions"
)
mod_ang, mod_ang_z, f_ang = evaluate_and_fit_best_model(
    df_diff_nonpol, 'FAA_Target', form_ref_angry, "Overall_Emotion_RefAngry", subfolder="emotions"
)

# 2. Extract Combined Fixed Effects 
term_alt_raw = "C(Emotion, Treatment('Angry'))[T.Happy]"
extract_combined_fixed_effects(
    mod_neu, mod_neu_z, mod_ang, mod_ang_z, term_alt_raw, "Overall_Emotion", subfolder="emotions"
)

# 3. Print & Export Pairwise Contrasts
compute_and_print_emms_contrasts(
    mod_neu, df_diff_nonpol, f_neu, factors_emotion, "Overall_Emotion_Contrasts", subfolder="emotions"
)

# %% Cell 6: Main Effect of Phase (Static vs. Dynamic) #MINOR CORE ANAYSIS
print("\n=============================================================================================")
print(f"TARGETED ANALYSIS: Static vs. Dynamic - {metric_name.upper()}")
print("=============================================================================================")

# Reshape
df_sd = df_trials.melt(
    id_vars=['Subject', 'Trial', 'Group', 'Pol_Category', 'Emotion'],
    value_vars=['Phase_Static', 'Phase_Dynamic'],
    var_name='Phase', value_name='FAA_Val'
)
df_sd['Phase'] = df_sd['Phase'].str.replace('Phase_', '') 

# Standardize
df_sd['FAA_Target'] = (df_sd['FAA_Val'] - df_sd['FAA_Val'].mean()) / df_sd['FAA_Val'].std() if USE_Z_SCORES else df_sd['FAA_Val']

# Fit and Export with 'StaticDynamic' naming convention
mod_sd, mod_sd_z, form_sd = evaluate_and_fit_best_model(df_sd, 'FAA_Target', "C(Phase, Treatment('Static'))", 'StaticDynamic_MainEffect', subfolder="phase")
extract_fixed_effects(mod_sd, mod_sd_z, 'StaticDynamic_MainEffect', subfolder="phase")
compute_and_print_emms_contrasts(mod_sd, df_sd, form_sd, {'Phase': ['Static', 'Dynamic']}, 'StaticDynamic_MainEffect', subfolder="phase")

# %% Cell 7: Main Effect of Phase (Baseline vs. Dynamic) #EXPLORATORY, NOT IN USE
print("\n=============================================================================================")
print(f"TARGETED ANALYSIS: Baseline vs. Dynamic - {metric_name.upper()}")
print("=============================================================================================")

# Reshape
df_bd = df_trials.melt(
    id_vars=['Subject', 'Trial', 'Group', 'Pol_Category', 'Emotion'],
    value_vars=['Phase_Baseline', 'Phase_Dynamic'],
    var_name='Phase', value_name='FAA_Val'
)
df_bd['Phase'] = df_bd['Phase'].str.replace('Phase_', '') 

# Standardize
df_bd['FAA_Target'] = (df_bd['FAA_Val'] - df_bd['FAA_Val'].mean()) / df_bd['FAA_Val'].std() if USE_Z_SCORES else df_bd['FAA_Val']

# Fit and Export with 'BaselineDynamic' naming convention
mod_bd, mod_bd_z, form_bd = evaluate_and_fit_best_model(df_bd, 'FAA_Target', "C(Phase, Treatment('Baseline'))", 'BaselineDynamic_MainEffect', subfolder="phase")
extract_fixed_effects(mod_bd, mod_bd_z, 'BaselineDynamic_MainEffect', subfolder="phase")
compute_and_print_emms_contrasts(mod_bd, df_bd, form_bd, {'Phase': ['Baseline', 'Dynamic']}, 'BaselineDynamic_MainEffect', subfolder="phase")

#%%
