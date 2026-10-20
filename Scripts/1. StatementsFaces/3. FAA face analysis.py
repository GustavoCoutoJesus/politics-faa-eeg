# %% #######################################################################################################################
# The goal of this script is to run core FAA regressions
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Cell 1: Single-Trial Data Extraction & JF Baseline Stats
import os
import time
import mne
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
from pathlib import Path

# Suppress convergence warnings for clean output
warnings.filterwarnings("ignore", module="statsmodels")

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

def get_party_labels(face_raw, stmt_raw):
    """Maps GL/PVV and left/right to In-party/Out-party"""
    face_party = 'In-party' if face_raw == 'GL' else 'Out-party'
    stmt_party = 'In-party' if stmt_raw == 'left' else 'Out-party'
    return face_party, stmt_party

def face_pairing_full(face_party, stmt_party):
    congr = 'Congruent' if face_party == stmt_party else 'Incongruent'
    return f"{congr} {face_party.lower()} face"

def face_pairing_abbr(face_party, stmt_party):
    congr = 'Cong.' if face_party == stmt_party else 'Incong.'
    party_abbr = 'in-' if face_party == 'In-party' else 'out-'
    return f"{congr} {party_abbr} face"

def pretty_term(term):
    """Cleans up statsmodels predictors for reporting"""
    t = str(term).replace('C(Face)[T.PVV]', 'Face (In- vs Out-party)')
    t = t.replace('C(Statement)[T.right]', 'Statement (In- vs Out-party)')
    t = t.replace('C(Face)[T.PVV]:C(Statement)[T.right]', 'Face \\times Statement')
    t = t.replace('Intercept', '(Intercept)')
    t = t.replace('Trial_Target', 'Trial')
    return t

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"
FAA_METHOD = "diff"

PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids' / 'derivatives'
REGRESSION_DIR = DERIVATIVES_DIR / 'regression' / BATCH_NAME / FAA_METHOD
REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR = DERIVATIVES_DIR / 'quality_check' / BATCH_NAME

EXCLUDE_SUBJECTS = {"sub-P03"}  

# Auto-Exclude Flagged Participants (Bad F3/F4)
flag_file = QC_DIR / 'Flagged_Bad_F3_F4_Participants.csv'
if flag_file.exists():
    df_flags = pd.read_csv(flag_file)
    if not df_flags.empty and 'Subject' in df_flags.columns:
        EXCLUDE_SUBJECTS.update(df_flags['Subject'].unique())
print(f"Final Exclusion List: {EXCLUDE_SUBJECTS}")

# Event Dictionary
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

# Mapping JF Event IDs to Party and Politician for baseline calculation
JF_EVENT_MAP = {
    201: ('PVV', 'Agema'), 
    202: ('PVV', 'Wilders'), 
    211: ('GL', 'Klaver'), 
    212: ('GL', 'Timmermans')
}

print("==================================================")
print(f"Extracting Single-Trial Data from {BATCH_NAME}...")
print("==================================================")

eeg_processed_dir = DERIVATIVES_DIR / 'processed' / BATCH_NAME
subject_folders = sorted([d for d in eeg_processed_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
all_trials = []
jf_baselines = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    eeg_dir = subj_dir / 'eeg'
    sf_file = eeg_dir / f"{subject}_task-StatementsFaces_faa_{FAA_METHOD}-epo.fif" 
    jf_file = eeg_dir / f"{subject}_task-JustFaces_faa_{FAA_METHOD}-epo.fif"

    if sf_file.exists() and jf_file.exists():
        
        # Calculate Participant-Level JF Baselines
        epochs_jf = mne.read_epochs(jf_file, preload=True, verbose=False)
        time_mask_jf = (epochs_jf.times >= 0.0) & (epochs_jf.times <= 1.2)
        faa_data_jf = epochs_jf.get_data(picks='FAA')[:, 0, :]
        mean_faa_jf_trials = np.mean(faa_data_jf[:, time_mask_jf], axis=-1)
        
        party_vals = {'GL': [], 'PVV': []}
        pol_vals = {'Klaver': [], 'Timmermans': [], 'Agema': [], 'Wilders': []}
        
        for idx, ev_id in enumerate(epochs_jf.events[:, 2]):
            if ev_id in JF_EVENT_MAP:
                party, pol = JF_EVENT_MAP[ev_id]
                val = mean_faa_jf_trials[idx]
                party_vals[party].append(val)
                pol_vals[pol].append(val)
                
        # Calculate means 
        overall_mean = np.mean(mean_faa_jf_trials)
        party_means = {k: np.mean(v) if len(v) > 0 else overall_mean for k, v in party_vals.items()}
        pol_means = {k: np.mean(v) if len(v) > 0 else party_means.get('GL' if k in ['Klaver','Timmermans'] else 'PVV', overall_mean) for k, v in pol_vals.items()}
        
        # Add to baselines dataframe for tracking
        jf_baselines.append({
            'Subject': subject, 'Overall_Mean': overall_mean,
            'GL_Mean': party_means['GL'], 'PVV_Mean': party_means['PVV']
        })

        # SF Processing 
        epochs = mne.read_epochs(sf_file, preload=True, verbose=False)
        time_mask = (epochs.times >= 0.0) & (epochs.times <= 1.2)
        faa_data = epochs.get_data(picks='FAA')[:, 0, :]
        
        mean_faa_per_trial = np.mean(faa_data[:, time_mask], axis=-1)
        
        for i, event_id in enumerate(epochs.events[:, 2]):
            event_name = ID_TO_EVENT.get(event_id)
            if not event_name: continue
                
            parts = event_name.split('/')
            party = parts[0]
            pol = parts[1]
            statement = parts[2]
            
            raw_val = mean_faa_per_trial[i]
            base_party = party_means.get(party, overall_mean)
            base_pol = pol_means.get(pol, base_party)
            
            all_trials.append({
                'Subject': subject,
                'Trial': i + 1,
                'Face': party,
                'Politician': pol,
                'Statement': statement,
                'FAA_Raw': raw_val,
                'FAA_Party_Corrected': raw_val - base_party,
                'FAA_Pol_Corrected': raw_val - base_pol
            })

df_trials = pd.DataFrame(all_trials)
df_baselines = pd.DataFrame(jf_baselines)

print(f"Extraction complete: {len(df_trials)} total single trials.")
print("\n--- JustFaces (JF) Baseline Descriptive Statistics ---")
print(f"Participants Processed: {len(df_baselines)}")
print(f"Grand Mean FAA (JF):  {df_baselines['Overall_Mean'].mean():.4f}")
print(f"Std Dev FAA (JF):     {df_baselines['Overall_Mean'].std():.4f}\n")


# %% Cell 2: Standardized Multi-Model Fitting & Model-Specific Exports
print("=============================================================================================")

# TOGGLE STANDARDIZATION 
USE_Z_SCORES = False


metric_name = "Standardized" if USE_Z_SCORES else "Unstandardized"
coef_sym = "$\\beta$" if USE_Z_SCORES else "$b$"

print(f"EVALUATING AND FITTING {metric_name.upper()} MODELS")
print("=============================================================================================")

# 1. Prepare Variables
df_proc = df_trials.copy()

# Unstandardized targets (for b, SE, p)
df_proc['FAA_Raw_Target'] = df_trials['FAA_Raw']
df_proc['FAA_Party_Target'] = df_trials['FAA_Party_Corrected']
df_proc['FAA_Pol_Target'] = df_trials['FAA_Pol_Corrected']
df_proc['Trial_Target'] = df_trials['Trial']

# Standardized targets (for beta)
df_proc['FAA_Raw_Target_Z'] = (df_trials['FAA_Raw'] - df_trials['FAA_Raw'].mean()) / df_trials['FAA_Raw'].std()
df_proc['FAA_Party_Target_Z'] = (df_trials['FAA_Party_Corrected'] - df_trials['FAA_Party_Corrected'].mean()) / df_trials['FAA_Party_Corrected'].std()
df_proc['FAA_Pol_Target_Z'] = (df_trials['FAA_Pol_Corrected'] - df_trials['FAA_Pol_Corrected'].mean()) / df_trials['FAA_Pol_Corrected'].std()
df_proc['Trial_Target_Z'] = (df_trials['Trial'] - df_trials['Trial'].mean()) / df_trials['Trial'].std()

# 2. Find Winning Model Structure using Target Data
models_to_test = {
    "Model 1 (GLM Base)": {"type": "ols", "formula": "FAA_Raw_Target ~ C(Face) * C(Statement)"},
    "Model 2 (GLM + Trial)": {"type": "ols", "formula": "FAA_Raw_Target ~ C(Face) * C(Statement) + Trial_Target"},
    "Model 3 (LMM Base)": {"type": "mixedlm", "formula": "FAA_Raw_Target ~ C(Face) * C(Statement)", "groups": "Subject"},
    "Model 4 (LMM + Trial)": {"type": "mixedlm", "formula": "FAA_Raw_Target ~ C(Face) * C(Statement) + Trial_Target", "groups": "Subject"}
}

fit_results = []
for name, params in models_to_test.items():
    try:
        if params["type"] == "ols":
            model = smf.ols(params["formula"], data=df_proc).fit()
        else:
            model = smf.mixedlm(params["formula"], data=df_proc, groups=df_proc[params["groups"]]).fit(reml=False)
        fit_results.append({'Model': name, 'AIC': model.aic, 'BIC': model.bic, 'LogLik': model.llf})
    except Exception as e:
        fit_results.append({'Model': name, 'AIC': np.inf, 'BIC': np.inf, 'LogLik': np.nan})

df_fits = pd.DataFrame(fit_results).sort_values(by='AIC').reset_index(drop=True)
best_model_name = df_fits.iloc[0]['Model']
BEST_MODEL_CONFIG = models_to_test[best_model_name]
print(f"Winning Structure: **{best_model_name}**\n")

# Save Fit Indices
df_fits.to_csv(REGRESSION_DIR / 'Model_Fit_Indices.csv', index=False)
tex_fits = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Model Fit Comparison (Sorted by AIC)}\n"
    "\\label{tab:model_fits}\n"
    "\\begin{tabular}{@{}lccc@{}}\n"
    "\\toprule\n"
    "\\textbf{Model} & \\textbf{AIC} & \\textbf{BIC} & \\textbf{Log-Likelihood} \\\\\n"
    "\\midrule\n"
)
for _, row in df_fits.iterrows():
    tex_fits += f"{row['Model']} & {row['AIC']:.2f} & {row['BIC']:.2f} & {row['LogLik']:.2f} \\\\\n"
tex_fits += "\\bottomrule\n\\end{tabular}\n\\end{table}"
with open(REGRESSION_DIR / 'Model_Fit_Indices.tex', 'w') as f:
    f.write(tex_fits)

# 3. Fit Winning Structure to Raw, Party-Corrected, and Politician-Corrected
formula_raw = BEST_MODEL_CONFIG['formula']
formula_raw_z = formula_raw.replace('FAA_Raw_Target', 'FAA_Raw_Target_Z').replace('Trial_Target', 'Trial_Target_Z')

formula_party = formula_raw.replace('FAA_Raw_Target', 'FAA_Party_Target')
formula_party_z = formula_raw_z.replace('FAA_Raw_Target_Z', 'FAA_Party_Target_Z')

formula_pol = formula_raw.replace('FAA_Raw_Target', 'FAA_Pol_Target')
formula_pol_z = formula_raw_z.replace('FAA_Raw_Target_Z', 'FAA_Pol_Target_Z')

def get_model_results(form_unstd, form_std, df, config, name):
    if config['type'] == 'mixedlm':
        model = smf.mixedlm(form_unstd, data=df, groups=df[config['groups']]).fit(reml=True)
        model_z = smf.mixedlm(form_std, data=df, groups=df[config['groups']]).fit(reml=True)
    else:
        model = smf.ols(form_unstd, data=df).fit()
        model_z = smf.ols(form_std, data=df).fit()
        
    res = pd.DataFrame({
        'Model_Type': name,
        'Predictor': model.params.index,
        'Estimate': model.params.values,
        'Beta': model_z.params.values,
        'SE': model.bse.values,
        't_z_value': model.tvalues.values,
        'p_value': model.pvalues.values,
        'CI_Lower': model.conf_int()[0].values,
        'CI_Upper': model.conf_int()[1].values
    })
    return model, res[~res['Predictor'].str.contains('Var')]

# Different models, based on which type of face is used as baseline. Politician_Corrected_SF is used later in analyses.
model_raw, res_raw = get_model_results(formula_raw, formula_raw_z, df_proc, BEST_MODEL_CONFIG, 'Raw_SF')
model_party, res_party = get_model_results(formula_party, formula_party_z, df_proc, BEST_MODEL_CONFIG, 'Party_Corrected_SF')
model_pol, res_pol = get_model_results(formula_pol, formula_pol_z, df_proc, BEST_MODEL_CONFIG, 'Politician_Corrected_SF')

df_all_results = pd.concat([res_raw, res_party, res_pol], ignore_index=True)
df_all_results.to_csv(REGRESSION_DIR / f'{metric_name}_Fixed_Effects_Comparison.csv', index=False)

# Print and Export Model-Specific Results
models_data = [
    ('Raw_SF', res_raw),
    ('Party_Corrected_SF', res_party),
    ('Politician_Corrected_SF', res_pol)
]

for name, res_df in models_data:
    print(f"\n--- {metric_name} Fixed Effects ({name}) ---")
    
# 1. Format for CSV 
    csv_out = res_df.copy()
    csv_out['Predictor'] = csv_out['Predictor'].apply(pretty_term)
    
    # Directly populate both from your new dataframe columns
    csv_out['b'] = csv_out['Estimate'].apply(fmt3)
    csv_out['beta'] = csv_out['Beta'].apply(fmt2) 
    
    csv_out['SE'] = csv_out['SE'].apply(fmt3)
    csv_out['95% CI'] = csv_out.apply(lambda r: ci_str(r['CI_Lower'], r['CI_Upper'], 2), axis=1)
    csv_out['p'] = csv_out['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    
    csv_export = csv_out[['Predictor', 'b', 'SE', 't_z_value', 'p', 'beta', '95% CI']].rename(columns={'t_z_value': 't'})
    csv_export.to_csv(REGRESSION_DIR / f'Fixed_Effects_{name}.csv', index=False)
    
    # 2. Format for LaTeX 
    tex_rows = ""
    for _, row in res_df.iterrows():
        pred_clean = pretty_term(row['Predictor'])
        b_val = fmt3(row['Estimate'])
        beta_val = fmt2(row['Beta']) 
        se_val = fmt3(row['SE'])
        t_val = fmt2(row['t_z_value'])
        p_tex = fmt_p_tex(row['p_value'])
        ci_val = ci_str(row['CI_Lower'], row['CI_Upper'], 2)
        
        tex_rows += f"{pred_clean} & {b_val} & {se_val} & {t_val} & {p_tex} & {beta_val} & {ci_val} \\\\\n"

    name_tex = name.replace('_', '\\_')
    tex_fixed = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{Fixed effects from the final model predicting FAA ({name_tex})}}\n"
        f"\\label{{tab:faa_fixed_effects_{name}}}\n"
        "\\begin{threeparttable}\n"
        "\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.3] S[table-format=2.2] l S[table-format=1.2] l@{}}\n"
        "\\toprule\n"
        "\\textbf{Predictor} & \\textbf{$b$} & \\textbf{SE} & \\textbf{$t$} & \\textbf{$p$} & \\textbf{$\\beta$} & \\textbf{95\\% CI} \\\\\n"
        "\\midrule\n"
        f"{tex_rows}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n"
        "\\item \\textit{Note.} Unstandardized ($b$) and standardized ($\\beta$) coefficients shown with 95\\% CIs.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}"
    )
    with open(REGRESSION_DIR / f'Fixed_Effects_{name}.tex', 'w') as f:
        f.write(tex_fixed)
        
# %% Cell 3: Estimated Marginal Means (EMMs) & Pairwise Contrasts (Model-Specific Exports)
import scipy.stats as stats
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix

print("\n=============================================================================================")
print(f"COMPUTING {metric_name.upper()} EMMs & CONTRASTS")
print("=============================================================================================")

PAIRWISE_DIR = DERIVATIVES_DIR / 'pairwise' / BATCH_NAME / FAA_METHOD
PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)

# Define Conditions for Prediction
faces = ['GL', 'PVV']
statements = ['left', 'right']
conditions = list(itertools.product(faces, statements))

pred_dict = {'Face': [c[0] for c in conditions], 'Statement': [c[1] for c in conditions]}
if 'Trial_Target' in formula_raw:
    pred_dict['Trial_Target'] = [df_proc['Trial_Target'].mean()] * 4
df_pred = pd.DataFrame(pred_dict)

def compute_emms_contrasts(model, df_pred, formula_str, model_name):
    formula_rhs = formula_str.split('~')[1]
    X = dmatrix(formula_rhs, df_pred, return_type='dataframe')
    fe_cols = [col for col in model.params.index if col in X.columns]
    X, params, cov_matrix = X[fe_cols], model.params[fe_cols].values, model.cov_params().loc[fe_cols, fe_cols].values
    
    resid_var = model.scale if hasattr(model, 'scale') else model.mse_resid
    resid_sd = np.sqrt(resid_var)

    X_vals = X.values
    emms = X_vals.dot(params)
    se_emms = np.sqrt(np.diag(X_vals.dot(cov_matrix).dot(X_vals.T)))

    df_emms = pd.DataFrame({'Model': model_name, 'Face': df_pred['Face'], 'Statement': df_pred['Statement'], 'EMM_Val': emms, 'SE': se_emms})
    
    contrast_results = []
    for i, j in itertools.combinations(range(4), 2):
        cond_a = f"{df_pred.iloc[i]['Face']}-{df_pred.iloc[i]['Statement']}"
        cond_b = f"{df_pred.iloc[j]['Face']}-{df_pred.iloc[j]['Statement']}"
        
        c_vec = X_vals[i] - X_vals[j]
        diff, se_diff = c_vec.dot(params), np.sqrt(c_vec.dot(cov_matrix).dot(c_vec.T))
        z_stat = diff / se_diff
        p_raw = 2 * stats.norm.sf(np.abs(z_stat))
        ci_low = diff - (1.96 * se_diff)
        ci_high = diff + (1.96 * se_diff)
        
        contrast_results.append({
            'Model': model_name, 'Cond_A': cond_a, 'Cond_B': cond_b, 
            'Estimate': diff, 'SE': se_diff, 'z_stat': z_stat, 'd': diff / resid_sd,
            'CI_low': ci_low, 'CI_high': ci_high, 'p_raw': p_raw
        })

    df_contrasts = pd.DataFrame(contrast_results)
    _, p_holm, _, _ = multipletests(df_contrasts['p_raw'], method='holm')
    df_contrasts['p_holm'] = p_holm
    
    return df_emms, df_contrasts

emms_raw, contr_raw = compute_emms_contrasts(model_raw, df_pred, formula_raw, 'Raw_SF')
emms_party, contr_party = compute_emms_contrasts(model_party, df_pred, formula_party, 'Party_Corrected_SF')
emms_pol, contr_pol = compute_emms_contrasts(model_pol, df_pred, formula_pol, 'Politician_Corrected_SF')

emms_contrasts_data = [
    ('Raw_SF', emms_raw, contr_raw),
    ('Party_Corrected_SF', emms_party, contr_party),
    ('Politician_Corrected_SF', emms_pol, contr_pol)
]

for name, df_e, df_c in emms_contrasts_data:
    name_tex = name.replace('_', '\\_')

    # 1. EMM Exports 
    df_e_out = df_e.copy()
    df_e_out['FaceParty'], df_e_out['StatementParty'] = zip(*df_e_out.apply(lambda r: get_party_labels(r['Face'], r['Statement']), axis=1))
    df_e_out['Face pairing'] = df_e_out.apply(lambda r: face_pairing_full(r['FaceParty'], r['StatementParty']), axis=1)
    df_e_out['Statement'] = df_e_out['StatementParty']
    
    csv_emm = pd.DataFrame({
        'Statement': df_e_out['Statement'],
        'Face pairing': df_e_out['Face pairing'],
        'M': df_e_out['EMM_Val'].apply(fmt2),
        'SE': df_e_out['SE'].apply(fmt3)
    })
    csv_emm.to_csv(PAIRWISE_DIR / f'FAA_EMMs_{name}.csv', index=False)
    
    tex_emms = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{metric_name} Estimated marginal means (EMMs) and SEs by statement and face condition ({name_tex})}}\n"
        f"\\label{{tab:faa_emms_{name}}}\n"
        "\\begin{threeparttable}\n"
        "\\begin{tabular}{@{}ll S[table-format=1.2] S[table-format=1.3]@{}}\n"
        "\\toprule\n"
        "\\textbf{Statement} & \\textbf{Face pairing} & \\textbf{M} & \\textbf{SE} \\\\\n"
        "\\midrule\n"
    )
    for _, row in df_e_out.iterrows():
        tex_emms += f"{row['Statement']} & {row['Face pairing']} & {fmt2(row['EMM_Val'])} & {fmt3(row['SE'])} \\\\\n"
    tex_emms += (
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n"
        "\\item \\textit{Note.} Values are Estimated Marginal Means (EMMs) with Standard Errors (SEs), taken from the best-fitting model.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}"
    )
    with open(PAIRWISE_DIR / f'FAA_EMMs_{name}.tex', 'w') as f: f.write(tex_emms)

    # 2. Contrasts Exports
    def parse_cond(cond_str):
        face_raw, stmt_raw = cond_str.split('-')
        fp, sp = get_party_labels(face_raw, stmt_raw)
        return face_pairing_abbr(fp, sp)

    df_c_out = df_c.copy()
    df_c_out['Cond A'] = df_c_out['Cond_A'].apply(parse_cond)
    df_c_out['Cond B'] = df_c_out['Cond_B'].apply(parse_cond)
    
    csv_contrasts = pd.DataFrame({
        'Cond A': df_c_out['Cond A'],
        'Cond B': df_c_out['Cond B'],
        'Delta': df_c_out['Estimate'].apply(fmt2),
        'd': df_c_out['d'].apply(fmt2),
        '95% CI': df_c_out.apply(lambda r: ci_str(r['CI_low'], r['CI_high'], 2), axis=1),
        'p_raw': df_c_out['p_raw'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"),
        'p_Holm': df_c_out['p_holm'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    })
    csv_contrasts.to_csv(PAIRWISE_DIR / f'FAA_Contrasts_{name}.csv', index=False)
    
    tex_contrasts = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{Pairwise contrasts of FAA (Holm-corrected) ({name_tex})}}\n"
        f"\\label{{tab:faa_contrasts_{name}}}\n"
        "\\begin{threeparttable}\n"
        "\\begin{tabular}{@{}ll S[table-format=1.2] S[table-format=1.2] l S[table-format=1.2e-2] S[table-format=1.2e-2]@{}}\n"
        "\\toprule\n"
        "\\multicolumn{2}{c}{\\textbf{Contrast}} & \\textbf{$\\Delta$} & \\textbf{d} & \\textbf{95\\% CI} & \\textbf{$p_\\text{raw}$} & \\textbf{$p_\\text{Holm}$} \\\\\n"
        "\\cmidrule(lr){1-2}\n"
        "\\textbf{Cond A} & \\textbf{Cond B} &  &  &  &  &  \\\\\n"
        "\\midrule\n"
    )
    for _, row in df_c_out.iterrows():
        tex_contrasts += f"{row['Cond A']} & {row['Cond B']} & {fmt2(row['Estimate'])} & {fmt2(row['d'])} & {ci_str(row['CI_low'], row['CI_high'], 2)} & {fmt_p_tex(row['p_raw'])} & {fmt_p_tex(row['p_holm'])} \\\\\n"
        
    tex_contrasts += (
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\footnotesize\n"
        "\\item \\textit{Note.} Each row compares two pairings of statement and face (Cond A vs Cond B). Labels indicate congruence of the \\emph{face} with the \\emph{statement}: Cong. = Congruent, Incong. = Incongruent; in-/out- refers to the partisan alignment of the face. Positive values indicate higher FAA for Cond A than Cond B. Estimates are provided as mean differences ($\\Delta$) and standardized Cohen's d.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}"
    )
    with open(PAIRWISE_DIR / f'FAA_Contrasts_{name}.tex', 'w') as f: f.write(tex_contrasts)

print(f"{metric_name} EMMs and Pairwise Contrasts successfully calculated and exported to individual CSV and TeX files.")


# %% Cell 4: Visualization of Main Effects, Interactions, and EMMs 
import matplotlib.pyplot as plt
import seaborn as sns

print("\n=============================================================================================")
print("GENERATING VISUALIZATIONS")
print("=============================================================================================")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Plot A: Coefficients (Fixed Effects) ---
df_plot_effects = df_all_results[~df_all_results['Predictor'].isin(['Intercept', 'Trial_Target'])].copy()

df_plot_effects['Predictor'] = df_plot_effects['Predictor'].str.replace('C(Face)[T.PVV]', 'Face (Out-)')
df_plot_effects['Predictor'] = df_plot_effects['Predictor'].str.replace('C(Statement)[T.right]', 'Statement (Out-)')
df_plot_effects['Predictor'] = df_plot_effects['Predictor'].str.replace('C(Face)[T.PVV]:C(Statement)[T.right]', 'Interaction: Face x Statement')

# Base barplot to get labels setup
sns.barplot(data=df_plot_effects, x='Predictor', y='Estimate', hue='Model_Type', ax=axes[0], alpha=0.0)

axes[0].cla() 
width = 0.25
x = np.arange(len(df_plot_effects['Predictor'].unique()))
raw_eff = df_plot_effects[df_plot_effects['Model_Type'] == 'Raw_SF']
party_eff = df_plot_effects[df_plot_effects['Model_Type'] == 'Party_Corrected_SF']
pol_eff = df_plot_effects[df_plot_effects['Model_Type'] == 'Politician_Corrected_SF']

axes[0].bar(x - width, raw_eff['Estimate'], width, yerr=raw_eff['SE'], capsize=5, label='Raw SF', color='#3498db', alpha=0.8)
axes[0].bar(x, party_eff['Estimate'], width, yerr=party_eff['SE'], capsize=5, label='Party Corrected SF', color='#2ecc71', alpha=0.8)
axes[0].bar(x + width, pol_eff['Estimate'], width, yerr=pol_eff['SE'], capsize=5, label='Politician Corrected SF', color='#e74c3c', alpha=0.8)

axes[0].set_xticks(x)
axes[0].set_xticklabels(raw_eff['Predictor'])
axes[0].axhline(0, color='black', linewidth=1)
axes[0].set_ylabel(f'{metric_name} Coefficient ({coef_sym})')
axes[0].set_title(f'A: {metric_name} Fixed Effects & Interactions')
axes[0].legend()

# --- Plot B: EMMs (Interaction Plot) ---
df_emms_all = pd.concat([emms_raw, emms_party, emms_pol], ignore_index=True)
df_emms_all['Condition'] = df_emms_all['Face'] + " / " + df_emms_all['Statement']

# Map the current "Face / Statement" labels to your desired names
label_map = {
    'GL / left': 'Congruent in-party',
    'GL / right': 'Incongruent in-party',
    'PVV / left': 'Incongruent out-party',
    'PVV / right': 'Congruent out-party'
}
df_emms_all['Condition_Label'] = df_emms_all['Condition'].map(label_map)

x_emms = np.arange(len(df_emms_all['Condition'].unique()))
raw_emms = df_emms_all[df_emms_all['Model'] == 'Raw_SF']
party_emms = df_emms_all[df_emms_all['Model'] == 'Party_Corrected_SF']
pol_emms = df_emms_all[df_emms_all['Model'] == 'Politician_Corrected_SF']

axes[1].errorbar(x_emms - 0.1, raw_emms['EMM_Val'], yerr=raw_emms['SE'], fmt='o', markersize=8, capsize=5, label='Raw SF', color='#3498db')
axes[1].errorbar(x_emms, party_emms['EMM_Val'], yerr=party_emms['SE'], fmt='^', markersize=8, capsize=5, label='Party Corrected SF', color='#2ecc71')
axes[1].errorbar(x_emms + 0.1, pol_emms['EMM_Val'], yerr=pol_emms['SE'], fmt='s', markersize=8, capsize=5, label='Politician Corrected SF', color='#e74c3c')

# Connect points for visual grouping by Face
for i in [0, 2]: # Connecting Left/Right for GL, then PVV
    axes[1].plot([x_emms[i]-0.1, x_emms[i+1]-0.1], [raw_emms.iloc[i]['EMM_Val'], raw_emms.iloc[i+1]['EMM_Val']], color='#3498db', linestyle='--', alpha=0.5)
    axes[1].plot([x_emms[i], x_emms[i+1]], [party_emms.iloc[i]['EMM_Val'], party_emms.iloc[i+1]['EMM_Val']], color='#2ecc71', linestyle='--', alpha=0.5)
    axes[1].plot([x_emms[i]+0.1, x_emms[i+1]+0.1], [pol_emms.iloc[i]['EMM_Val'], pol_emms.iloc[i+1]['EMM_Val']], color='#e74c3c', linestyle='--', alpha=0.5)

axes[1].set_xticks(x_emms)
axes[1].set_xticklabels(raw_emms['Condition_Label'])
axes[1].axhline(0, color='black', linewidth=1, linestyle=':')
axes[1].set_ylabel(f'{metric_name} EMM')
axes[1].set_title(f'B: {metric_name} Estimated Marginal Means (EMMs)')
axes[1].legend()

plt.tight_layout()
fig.savefig(REGRESSION_DIR / 'Model_Comparison_Visualizations.png', dpi=300)
plt.show()

print(f"Visualizations saved to: {REGRESSION_DIR / 'Model_Comparison_Visualizations.png'}")

# %% Cell 5: SF vs JF Contrasts & Visualizations (Clean vs. Modified Sample)
# These analyses are supplementary. They user raw FAA
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

print("\n=============================================================================================")
print("COMPUTING SF vs JF CONTRASTS & GENERATING VISUALIZATIONS")
print("=============================================================================================")

# -----------------------------------------------------------------------------
# 1. Prepare Clean Data (from already processed df_trials and df_baselines)
# -----------------------------------------------------------------------------
# Get Subject-level averages for the 4 SF conditions
df_sf_mean = df_trials.groupby(['Subject', 'Face', 'Statement'])['FAA_Raw'].mean().reset_index()
df_sf_mean['Condition'] = "SF_" + df_sf_mean['Face'] + "_" + df_sf_mean['Statement']
df_sf_pivot = df_sf_mean.pivot(index='Subject', columns='Condition', values='FAA_Raw')

# Get Subject-level averages for the 2 JF conditions
df_jf_pivot = df_baselines.set_index('Subject')[['GL_Mean', 'PVV_Mean']]
df_jf_pivot.columns = ['JF_GL', 'JF_PVV']

# Combine into a single Clean DataFrame (1 row per subject, 6 columns for conditions)
df_clean = df_sf_pivot.join(df_jf_pivot).dropna()

sf_conds = ['SF_GL_left', 'SF_GL_right', 'SF_PVV_left', 'SF_PVV_right']
jf_conds = ['JF_GL', 'JF_PVV']
all_conds = jf_conds + sf_conds

# -----------------------------------------------------------------------------
# 2. Paired T-Test Contrasts (SF vs JF)
# -----------------------------------------------------------------------------
print("\n--- Paired Contrasts: Raw FAA (SF) vs Raw FAA (JF) ---")

contrast_results = []
# Calculate t-tests for every combination of SF and JF (4 x 2 = 8 contrasts)
for sf in sf_conds:
    for jf in jf_conds:
        diff = df_clean[sf] - df_clean[jf]
        t_stat, p_raw = stats.ttest_rel(df_clean[sf], df_clean[jf])
        
        mean_diff = diff.mean()
        se_diff = diff.std() / np.sqrt(len(diff))
        d_val = mean_diff / diff.std()  # Cohen's d for paired samples
        
        contrast_results.append({
            'SF_Cond': sf.replace('SF_', ''),
            'JF_Cond': jf.replace('JF_', ''),
            'Estimate': mean_diff,
            'SE': se_diff,
            't_stat': t_stat,
            'd': d_val,
            'p_raw': p_raw
        })

df_contrasts_sf_jf = pd.DataFrame(contrast_results)
_, p_holm, _, _ = multipletests(df_contrasts_sf_jf['p_raw'], method='holm')
df_contrasts_sf_jf['p_holm'] = p_holm

# Print results in a text table similar to the TeX outputs
print(f"{'SF Cond':<12} | {'JF Cond':<8} | {'Delta':<8} | {'Cohen d':<8} | {'t-stat':<8} | {'p_raw':<8} | {'p_Holm':<8}")
print("-" * 75)
for _, row in df_contrasts_sf_jf.iterrows():
    sig = "***" if row['p_holm'] < 0.001 else "**" if row['p_holm'] < 0.01 else "*" if row['p_holm'] < 0.05 else "." if row['p_holm'] < 0.10 else ""
    print(f"{row['SF_Cond']:<12} | {row['JF_Cond']:<8} | {row['Estimate']:>8.3f} | {row['d']:>8.2f} | {row['t_stat']:>8.2f} | {row['p_raw']:>8.3f} | {row['p_holm']:>8.3f} {sig}")


# -----------------------------------------------------------------------------
# 3. Extract Modified Sample (Excluding ONLY sub-P03; keeping Flagged)
# -----------------------------------------------------------------------------
print("\nExtracting data for modified sample (re-including flagged participants, omitting sub-P03)...")

modified_subjects_data = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    # Explicitly exclude ONLY sub-P03 (and implicitly skip if files are missing)
    if subject == "sub-P03": 
        continue

    eeg_dir = subj_dir / 'eeg'
    sf_file = eeg_dir / f"{subject}_task-StatementsFaces_faa_{FAA_METHOD}-epo.fif" 
    jf_file = eeg_dir / f"{subject}_task-JustFaces_faa_{FAA_METHOD}-epo.fif"

    if sf_file.exists() and jf_file.exists():
        subj_means = {'Subject': subject}
        
        # --- JF Baseline Extraction ---
        epochs_jf = mne.read_epochs(jf_file, preload=True, verbose=False)
        time_mask_jf = (epochs_jf.times >= 0.0) & (epochs_jf.times <= 1.2)
        faa_data_jf = epochs_jf.get_data(picks='FAA')[:, 0, :]
        mean_faa_jf_trials = np.mean(faa_data_jf[:, time_mask_jf], axis=-1)
        
        party_vals = {'GL': [], 'PVV': []}
        for idx, ev_id in enumerate(epochs_jf.events[:, 2]):
            if ev_id in JF_EVENT_MAP:
                party, _ = JF_EVENT_MAP[ev_id]
                party_vals[party].append(mean_faa_jf_trials[idx])
                
        overall_mean = np.mean(mean_faa_jf_trials)
        subj_means['JF_GL'] = np.mean(party_vals['GL']) if party_vals['GL'] else overall_mean
        subj_means['JF_PVV'] = np.mean(party_vals['PVV']) if party_vals['PVV'] else overall_mean

        # --- SF Extraction ---
        epochs_sf = mne.read_epochs(sf_file, preload=True, verbose=False)
        time_mask_sf = (epochs_sf.times >= 0.0) & (epochs_sf.times <= 1.2)
        faa_data_sf = epochs_sf.get_data(picks='FAA')[:, 0, :]
        mean_faa_sf_trials = np.mean(faa_data_sf[:, time_mask_sf], axis=-1)
        
        sf_cond_vals = {cond: [] for cond in sf_conds}
        for i, ev_id in enumerate(epochs_sf.events[:, 2]):
            event_name = ID_TO_EVENT.get(ev_id)
            if not event_name: continue
            parts = event_name.split('/')
            cond_key = f"SF_{parts[0]}_{parts[2]}" # e.g., SF_GL_left
            if cond_key in sf_cond_vals:
                sf_cond_vals[cond_key].append(mean_faa_sf_trials[i])
                
        for cond in sf_conds:
            subj_means[cond] = np.mean(sf_cond_vals[cond]) if sf_cond_vals[cond] else np.nan
            
        modified_subjects_data.append(subj_means)

df_modified = pd.DataFrame(modified_subjects_data).dropna()

# -----------------------------------------------------------------------------
# 4. Generate the Visualizations
# -----------------------------------------------------------------------------
# Melt dataframes for Seaborn
df_clean_melt = df_clean.reset_index().melt(id_vars='Subject', value_vars=all_conds, var_name='Condition', value_name='FAA')
df_mod_melt = df_modified.melt(id_vars='Subject', value_vars=all_conds, var_name='Condition', value_name='FAA')

# Standardize labels for plots
label_map = {
    'JF_GL': 'JF: GL', 'JF_PVV': 'JF: PVV',
    'SF_GL_left': 'SF: GL (Left)', 'SF_GL_right': 'SF: GL (Right)',
    'SF_PVV_left': 'SF: PVV (Left)', 'SF_PVV_right': 'SF: PVV (Right)'
}
df_clean_melt['Condition_Label'] = df_clean_melt['Condition'].map(label_map)
df_mod_melt['Condition_Label'] = df_mod_melt['Condition'].map(label_map)
plot_order = [label_map[c] for c in all_conds]

# Colors: Grays/Blues for JF, Greens/Reds for SF to match previous GL/PVV aesthetics 
palette = {
    'JF: GL': '#95a5a6', 'JF: PVV': '#7f8c8d',           # Neutrals for JF
    'SF: GL (Left)': '#2ecc71', 'SF: GL (Right)': '#27ae60', # Greens for GL
    'SF: PVV (Left)': '#e74c3c', 'SF: PVV (Right)': '#c0392b' # Reds for PVV
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# Plot 1: Clean Data
sns.barplot(
    data=df_clean_melt, x='Condition_Label', y='FAA', order=plot_order, 
    palette=palette, errorbar='se', capsize=0.1, ax=axes[0], alpha=0.9
)
axes[0].set_title(f'Cleaned Sample (N={len(df_clean)})\nMeans ± SE', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Raw FAA (Unstandardized)', fontsize=12)
axes[0].set_xlabel('Task Condition', fontsize=12)
axes[0].tick_params(axis='x', rotation=30)
axes[0].axhline(0, color='black', linewidth=1)

# Plot 2: Modified Data
sns.barplot(
    data=df_mod_melt, x='Condition_Label', y='FAA', order=plot_order, 
    palette=palette, errorbar='se', capsize=0.1, ax=axes[1], alpha=0.9
)
axes[1].set_title(f'Modified Sample (Incl. Flagged, Excl. P03) (N={len(df_modified)})\nMeans ± SE', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')
axes[1].set_xlabel('Task Condition', fontsize=12)
axes[1].tick_params(axis='x', rotation=30)
axes[1].axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.show()

# %% Cell 6: JustFaces (JF) Main Effect Analysis (In-party vs Out-party)
import scipy.stats as stats
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix

print("\n=============================================================================================")
print(f"EXTRACTING AND ANALYZING JUSTFACES (JF) DATA - {metric_name.upper()}")
print("=============================================================================================")

# ------------------------------------------------------------------------------
# 1. Extract Single-Trial JF Data
# ------------------------------------------------------------------------------
print("Extracting single-trial JF data...")
jf_single_trials = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    jf_file = subj_dir / 'eeg' / f"{subject}_task-JustFaces_faa_{FAA_METHOD}-epo.fif"

    if jf_file.exists():
        epochs_jf = mne.read_epochs(jf_file, preload=True, verbose=False)
        time_mask_jf = (epochs_jf.times >= 0.0) & (epochs_jf.times <= 1.2)
        faa_data_jf = epochs_jf.get_data(picks='FAA')[:, 0, :]
        mean_faa_jf_trials = np.mean(faa_data_jf[:, time_mask_jf], axis=-1)
        
        for idx, ev_id in enumerate(epochs_jf.events[:, 2]):
            if ev_id in JF_EVENT_MAP:
                party, pol = JF_EVENT_MAP[ev_id]
                jf_single_trials.append({
                    'Subject': subject,
                    'Trial': idx + 1,
                    'Face': party,
                    'Politician': pol,
                    'FAA_Raw': mean_faa_jf_trials[idx]
                })

df_jf_trials = pd.DataFrame(jf_single_trials)
print(f"Extraction complete: {len(df_jf_trials)} total JF single trials.")

# ------------------------------------------------------------------------------
# 2. Prepare Variables & Standardization
# ------------------------------------------------------------------------------

df_proc_jf = df_jf_trials.copy()

# Unstandardized targets (for b, SE, p)
df_proc_jf['FAA_Raw_Target'] = df_proc_jf['FAA_Raw']
df_proc_jf['Trial_Target'] = df_proc_jf['Trial']

# Standardized targets (for beta)
df_proc_jf['FAA_Raw_Target_Z'] = (df_proc_jf['FAA_Raw'] - df_proc_jf['FAA_Raw'].mean()) / df_proc_jf['FAA_Raw'].std()
df_proc_jf['Trial_Target_Z'] = (df_proc_jf['Trial'] - df_proc_jf['Trial'].mean()) / df_proc_jf['Trial'].std()

# ------------------------------------------------------------------------------
# 3. Find Winning Model Structure
# ------------------------------------------------------------------------------
models_to_test_jf = {
    "JF Model 1 (GLM Base)": {"type": "ols", "formula": "FAA_Raw_Target ~ C(Face)"},
    "JF Model 2 (GLM + Trial)": {"type": "ols", "formula": "FAA_Raw_Target ~ C(Face) + Trial_Target"},
    "JF Model 3 (LMM Base)": {"type": "mixedlm", "formula": "FAA_Raw_Target ~ C(Face)", "groups": "Subject"},
    "JF Model 4 (LMM + Trial)": {"type": "mixedlm", "formula": "FAA_Raw_Target ~ C(Face) + Trial_Target", "groups": "Subject"}
}

fit_results_jf = []
for name, params in models_to_test_jf.items():
    try:
        if params["type"] == "ols":
            model = smf.ols(params["formula"], data=df_proc_jf).fit()
        else:
            model = smf.mixedlm(params["formula"], data=df_proc_jf, groups=df_proc_jf[params["groups"]]).fit(reml=False)
        fit_results_jf.append({'Model': name, 'AIC': model.aic, 'BIC': model.bic, 'LogLik': model.llf})
    except Exception as e:
        fit_results_jf.append({'Model': name, 'AIC': np.inf, 'BIC': np.inf, 'LogLik': np.nan})

df_fits_jf = pd.DataFrame(fit_results_jf).sort_values(by='AIC').reset_index(drop=True)
best_model_name_jf = df_fits_jf.iloc[0]['Model']
BEST_MODEL_CONFIG_JF = models_to_test_jf[best_model_name_jf]
print(f"Winning Structure for JF: **{best_model_name_jf}**\n")

# Save Fit Indices
df_fits_jf.to_csv(REGRESSION_DIR / 'JF_Model_Fit_Indices.csv', index=False)

# ------------------------------------------------------------------------------
# 4. Fit Final Model & Export Fixed Effects
# ------------------------------------------------------------------------------
formula_jf = BEST_MODEL_CONFIG_JF['formula']
formula_jf_z = formula_jf.replace('FAA_Raw_Target', 'FAA_Raw_Target_Z').replace('Trial_Target', 'Trial_Target_Z')

if BEST_MODEL_CONFIG_JF['type'] == 'mixedlm':
    model_jf = smf.mixedlm(formula_jf, data=df_proc_jf, groups=df_proc_jf[BEST_MODEL_CONFIG_JF['groups']]).fit(reml=True)
    model_jf_z = smf.mixedlm(formula_jf_z, data=df_proc_jf, groups=df_proc_jf[BEST_MODEL_CONFIG_JF['groups']]).fit(reml=True)
else:
    model_jf = smf.ols(formula_jf, data=df_proc_jf).fit()
    model_jf_z = smf.ols(formula_jf_z, data=df_proc_jf).fit()
    
res_jf = pd.DataFrame({
    'Predictor': model_jf.params.index,
    'Estimate': model_jf.params.values,
    'Beta': model_jf_z.params.values, 
    'SE': model_jf.bse.values,
    't_z_value': model_jf.tvalues.values,
    'p_value': model_jf.pvalues.values,
    'CI_Lower': model_jf.conf_int()[0].values,
    'CI_Upper': model_jf.conf_int()[1].values
})
res_jf = res_jf[~res_jf['Predictor'].str.contains('Var')]

print(f"--- {metric_name} Fixed Effects (JF Task) ---")
print(res_jf[['Predictor', 'Estimate', 'SE', 'p_value']].to_string(index=False, float_format="%.4f"))

# Export CSV & LaTeX for Fixed Effects
csv_out_jf = res_jf.copy()
csv_out_jf['Predictor'] = csv_out_jf['Predictor'].apply(pretty_term)

# Directly populate both
csv_out_jf['b'] = csv_out_jf['Estimate'].apply(fmt3)
csv_out_jf['beta'] = csv_out_jf['Beta'].apply(fmt2)

csv_out_jf['SE'] = csv_out_jf['SE'].apply(fmt3)
csv_out_jf['95% CI'] = csv_out_jf.apply(lambda r: ci_str(r['CI_Lower'], r['CI_Upper'], 2), axis=1)
csv_out_jf['p'] = csv_out_jf['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")

csv_export_jf = csv_out_jf[['Predictor', 'b', 'SE', 't_z_value', 'p', 'beta', '95% CI']].rename(columns={'t_z_value': 't'})
csv_export_jf.to_csv(REGRESSION_DIR / 'Fixed_Effects_JF_Raw.csv', index=False)

tex_rows_jf = ""
for _, row in res_jf.iterrows():
    pred_clean = pretty_term(row['Predictor'])
    b_val = fmt3(row['Estimate'])
    beta_val = fmt2(row['Beta']) # Use Beta here!
    se_val = fmt3(row['SE'])
    t_val = fmt2(row['t_z_value'])
    p_tex = fmt_p_tex(row['p_value'])
    ci_val = ci_str(row['CI_Lower'], row['CI_Upper'], 2)
    tex_rows_jf += f"{pred_clean} & {b_val} & {se_val} & {t_val} & {p_tex} & {beta_val} & {ci_val} \\\\\n"

tex_fixed_jf = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Fixed effects from the final model predicting raw FAA (JustFaces Task)}\n"
    "\\label{tab:faa_fixed_effects_jf}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.3] S[table-format=2.2] l S[table-format=1.2] l@{}}\n"
    "\\toprule\n"
    "\\textbf{Predictor} & \\textbf{$b$} & \\textbf{SE} & \\textbf{$t$} & \\textbf{$p$} & \\textbf{$\\beta$} & \\textbf{95\\% CI} \\\\\n"
    "\\midrule\n"
    f"{tex_rows_jf}"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\begin{tablenotes}\n\\footnotesize\n"
    "\\item \\textit{Note.} Unstandardized ($b$) and standardized ($\\beta$) coefficients shown with 95\\% CIs.\n"
    "\\end{tablenotes}\n"
    "\\end{threeparttable}\n"
    "\\end{table}"
)
with open(REGRESSION_DIR / 'Fixed_Effects_JF_Raw.tex', 'w') as f:
    f.write(tex_fixed_jf)

# ------------------------------------------------------------------------------
# 5. Estimated Marginal Means (EMMs) & Contrasts
# ------------------------------------------------------------------------------
print("\nComputing JF EMMs & Contrasts...")

pred_dict_jf = {'Face': ['GL', 'PVV']}
if 'Trial_Target' in formula_jf:
    pred_dict_jf['Trial_Target'] = [df_proc_jf['Trial_Target'].mean()] * 2
df_pred_jf = pd.DataFrame(pred_dict_jf)

# Manual EMM calculation via patsy (matching Cell 3 approach)
formula_rhs_jf = formula_jf.split('~')[1]
X_jf = dmatrix(formula_rhs_jf, df_pred_jf, return_type='dataframe')
fe_cols_jf = [col for col in model_jf.params.index if col in X_jf.columns]
X_jf_matrix, params_jf, cov_matrix_jf = X_jf[fe_cols_jf].values, model_jf.params[fe_cols_jf].values, model_jf.cov_params().loc[fe_cols_jf, fe_cols_jf].values

resid_var_jf = model_jf.scale if hasattr(model_jf, 'scale') else model_jf.mse_resid
resid_sd_jf = np.sqrt(resid_var_jf)

emms_jf_vals = X_jf_matrix.dot(params_jf)
se_emms_jf = np.sqrt(np.diag(X_jf_matrix.dot(cov_matrix_jf).dot(X_jf_matrix.T)))

df_emms_jf = pd.DataFrame({
    'Face': df_pred_jf['Face'],
    'EMM_Val': emms_jf_vals,
    'SE': se_emms_jf
})

# Contrast (GL vs PVV)
c_vec_jf = X_jf_matrix[0] - X_jf_matrix[1]
diff_jf = c_vec_jf.dot(params_jf)
se_diff_jf = np.sqrt(c_vec_jf.dot(cov_matrix_jf).dot(c_vec_jf.T))
z_stat_jf = diff_jf / se_diff_jf
p_raw_jf = 2 * stats.norm.sf(np.abs(z_stat_jf))

df_contrasts_jf = pd.DataFrame([{
    'Cond_A': 'GL', 'Cond_B': 'PVV', 
    'Estimate': diff_jf, 'SE': se_diff_jf, 
    'z_stat': z_stat_jf, 'd': diff_jf / resid_sd_jf,
    'CI_low': diff_jf - (1.96 * se_diff_jf), 
    'CI_high': diff_jf + (1.96 * se_diff_jf), 
    'p_raw': p_raw_jf
}])

# --- Map GL/PVV to In-party/Out-party before export ---
jf_label_map = {'GL': 'In-party', 'PVV': 'Out-party'}

# Export EMMs
df_emms_jf_out = df_emms_jf.copy()
df_emms_jf_out['Face'] = df_emms_jf_out['Face'].map(jf_label_map)
df_emms_jf_out.to_csv(PAIRWISE_DIR / 'FAA_EMMs_JF_Raw.csv', index=False)

tex_emms_jf = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    f"\\caption{{{metric_name} Estimated Marginal Means (EMMs) for JustFaces Task}}\n"
    "\\label{tab:faa_emms_jf}\n"
    "\\begin{tabular}{@{}l S[table-format=1.4] S[table-format=1.4]@{}}\n"
    "\\toprule\n"
    "\\textbf{Face} & \\textbf{EMM} & \\textbf{SE} \\\\\n"
    "\\midrule\n"
)
for _, row in df_emms_jf_out.iterrows():
    tex_emms_jf += f"{row['Face']} & {row['EMM_Val']:.4f} & {row['SE']:.4f} \\\\\n"
tex_emms_jf += "\\bottomrule\n\\end{tabular}\n\\end{table}"
with open(PAIRWISE_DIR / 'FAA_EMMs_JF_Raw.tex', 'w') as f: f.write(tex_emms_jf)

# Export Contrasts
df_contrasts_jf_out = df_contrasts_jf.copy()
df_contrasts_jf_out['Cond_A'] = df_contrasts_jf_out['Cond_A'].map(jf_label_map)
df_contrasts_jf_out['Cond_B'] = df_contrasts_jf_out['Cond_B'].map(jf_label_map)
df_contrasts_jf_out.to_csv(PAIRWISE_DIR / 'FAA_Contrasts_JF_Raw.csv', index=False)

tex_contrasts_jf = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Pairwise contrasts of FAA (JustFaces Task)}\n"
    "\\label{tab:faa_contrasts_jf}\n"
    "\\begin{tabular}{@{}ll S[table-format=1.3] S[table-format=1.2] l S[table-format=1.3]@{}}\n"
    "\\toprule\n"
    "\\textbf{Cond A} & \\textbf{Cond B} & \\textbf{$\\Delta$} & \\textbf{d} & \\textbf{95\\% CI} & \\textbf{$p$} \\\\\n"
    "\\midrule\n"
)
for _, row in df_contrasts_jf_out.iterrows():
    tex_contrasts_jf += f"{row['Cond_A']} & {row['Cond_B']} & {row['Estimate']:.3f} & {row['d']:.2f} & [{row['CI_low']:.2f}, {row['CI_high']:.2f}] & {row['p_raw']:.3f} \\\\\n"
tex_contrasts_jf += "\\bottomrule\n\\end{tabular}\n\\end{table}"
with open(PAIRWISE_DIR / 'FAA_Contrasts_JF_Raw.tex', 'w') as f: f.write(tex_contrasts_jf)

print("JF EMMs and Pairwise Contrasts successfully calculated and exported to individual CSV and TeX files.")

# %% Cell 7: Main Effect of Task (SF vs JF)
import scipy.stats as stats
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix

print("\n=============================================================================================")
print(f"EVALUATING MAIN EFFECT OF TASK (SF vs JF) - {metric_name.upper()}")
print("=============================================================================================")

# ------------------------------------------------------------------------------
# 1. Combine SF and JF Single-Trial Data
# ------------------------------------------------------------------------------
# Extract only the necessary columns and add a 'Task' identifier
df_sf_sub = df_trials[['Subject', 'Trial', 'FAA_Raw']].copy()
df_sf_sub['Task'] = 'SF'

df_jf_sub = df_jf_trials[['Subject', 'Trial', 'FAA_Raw']].copy()
df_jf_sub['Task'] = 'JF'

# Combine into one master dataframe
df_task = pd.concat([df_sf_sub, df_jf_sub], ignore_index=True)

print(f"Combined dataset: {len(df_task)} total trials ({len(df_sf_sub)} SF, {len(df_jf_sub)} JF).")

# ------------------------------------------------------------------------------
# 2. Prepare Variables & Standardization
# ------------------------------------------------------------------------------
# Unstandardized targets (for b, SE, p)
df_task['FAA_Raw_Target'] = df_task['FAA_Raw']
df_task['Trial_Target'] = df_task['Trial']

# Standardized targets (for beta)
df_task['FAA_Raw_Target_Z'] = (df_task['FAA_Raw'] - df_task['FAA_Raw'].mean()) / df_task['FAA_Raw'].std()
df_task['Trial_Target_Z'] = (df_task['Trial'] - df_task['Trial'].mean()) / df_task['Trial'].std()

# ------------------------------------------------------------------------------
# 3. Find Winning Model Structure
# ------------------------------------------------------------------------------
models_to_test_task = {
    "Task Model 1 (GLM Base)": {"type": "ols", "formula": "FAA_Raw_Target ~ C(Task)"},
    "Task Model 2 (GLM + Trial)": {"type": "ols", "formula": "FAA_Raw_Target ~ C(Task) + Trial_Target"},
    "Task Model 3 (LMM Base)": {"type": "mixedlm", "formula": "FAA_Raw_Target ~ C(Task)", "groups": "Subject"},
    "Task Model 4 (LMM + Trial)": {"type": "mixedlm", "formula": "FAA_Raw_Target ~ C(Task) + Trial_Target", "groups": "Subject"}
}

fit_results_task = []
for name, params in models_to_test_task.items():
    try:
        if params["type"] == "ols":
            model = smf.ols(params["formula"], data=df_task).fit()
        else:
            model = smf.mixedlm(params["formula"], data=df_task, groups=df_task[params["groups"]]).fit(reml=False)
        fit_results_task.append({'Model': name, 'AIC': model.aic, 'BIC': model.bic, 'LogLik': model.llf})
    except Exception as e:
        fit_results_task.append({'Model': name, 'AIC': np.inf, 'BIC': np.inf, 'LogLik': np.nan})

df_fits_task = pd.DataFrame(fit_results_task).sort_values(by='AIC').reset_index(drop=True)
best_model_name_task = df_fits_task.iloc[0]['Model']
BEST_MODEL_CONFIG_TASK = models_to_test_task[best_model_name_task]
print(f"Winning Structure for Task Analysis: **{best_model_name_task}**\n")

# Save Fit Indices
df_fits_task.to_csv(REGRESSION_DIR / 'Task_Model_Fit_Indices.csv', index=False)

# ------------------------------------------------------------------------------
# 4. Fit Final Model & Export Fixed Effects
# ------------------------------------------------------------------------------
formula_task = BEST_MODEL_CONFIG_TASK['formula']
formula_task_z = formula_task.replace('FAA_Raw_Target', 'FAA_Raw_Target_Z').replace('Trial_Target', 'Trial_Target_Z')

if BEST_MODEL_CONFIG_TASK['type'] == 'mixedlm':
    model_task = smf.mixedlm(formula_task, data=df_task, groups=df_task[BEST_MODEL_CONFIG_TASK['groups']]).fit(reml=True)
    model_task_z = smf.mixedlm(formula_task_z, data=df_task, groups=df_task[BEST_MODEL_CONFIG_TASK['groups']]).fit(reml=True)
else:
    model_task = smf.ols(formula_task, data=df_task).fit()
    model_task_z = smf.ols(formula_task_z, data=df_task).fit()
    
res_task = pd.DataFrame({
    'Predictor': model_task.params.index,
    'Estimate': model_task.params.values,
    'Beta': model_task_z.params.values, 
    'SE': model_task.bse.values,
    't_z_value': model_task.tvalues.values,
    'p_value': model_task.pvalues.values,
    'CI_Lower': model_task.conf_int()[0].values,
    'CI_Upper': model_task.conf_int()[1].values
})
res_task = res_task[~res_task['Predictor'].str.contains('Var')]

print(f"--- {metric_name} Fixed Effects (Task Main Effect) ---")
print(res_task[['Predictor', 'Estimate', 'SE', 'p_value']].to_string(index=False, float_format="%.4f"))

# Export CSV & LaTeX for Fixed Effects
csv_out_task = res_task.copy()
csv_out_task['Predictor'] = csv_out_task['Predictor'].str.replace('C(Task)\\[T.SF\\]', 'Task (SF vs JF)', regex=True)
csv_out_task['Predictor'] = csv_out_task['Predictor'].str.replace('Intercept', '(Intercept)', regex=True)

# Directly populate both
csv_out_task['b'] = csv_out_task['Estimate'].apply(fmt3)
csv_out_task['beta'] = csv_out_task['Beta'].apply(fmt2)

csv_out_task['SE'] = csv_out_task['SE'].apply(fmt3)
csv_out_task['95% CI'] = csv_out_task.apply(lambda r: ci_str(r['CI_Lower'], r['CI_Upper'], 2), axis=1)
csv_out_task['p'] = csv_out_task['p_value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")

csv_export_task = csv_out_task[['Predictor', 'b', 'SE', 't_z_value', 'p', 'beta', '95% CI']].rename(columns={'t_z_value': 't'})
csv_export_task.to_csv(REGRESSION_DIR / 'Fixed_Effects_Task_Raw.csv', index=False)

tex_rows_task = ""
for _, row in res_task.iterrows():
    pred_clean = str(row['Predictor']).replace('C(Task)[T.SF]', 'Task (SF vs JF)').replace('Intercept', '(Intercept)')
    b_val = fmt3(row['Estimate'])
    beta_val = fmt2(row['Beta']) 
    se_val = fmt3(row['SE'])
    t_val = fmt2(row['t_z_value'])
    p_tex = fmt_p_tex(row['p_value'])
    ci_val = ci_str(row['CI_Lower'], row['CI_Upper'], 2)
    tex_rows_task += f"{pred_clean} & {b_val} & {se_val} & {t_val} & {p_tex} & {beta_val} & {ci_val} \\\\\n"

tex_fixed_task = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Fixed effects from the final model predicting raw FAA (SF vs JF Task)}\n"
    "\\label{tab:faa_fixed_effects_task}\n"
    "\\begin{threeparttable}\n"
    "\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.3] S[table-format=2.2] l S[table-format=1.2] l@{}}\n"
    "\\toprule\n"
    "\\textbf{Predictor} & \\textbf{$b$} & \\textbf{SE} & \\textbf{$t$} & \\textbf{$p$} & \\textbf{$\\beta$} & \\textbf{95\\% CI} \\\\\n"
    "\\midrule\n"
    f"{tex_rows_task}"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\begin{tablenotes}\n\\footnotesize\n"
    "\\item \\textit{Note.} Unstandardized ($b$) and standardized ($\\beta$) coefficients shown with 95\\% CIs. Reference task is JF.\n"
    "\\end{tablenotes}\n"
    "\\end{threeparttable}\n"
    "\\end{table}"
)
with open(REGRESSION_DIR / 'Fixed_Effects_Task_Raw.tex', 'w') as f:
    f.write(tex_fixed_task)

# ------------------------------------------------------------------------------
# 5. Estimated Marginal Means (EMMs) & Contrasts
# ------------------------------------------------------------------------------
print("\nComputing Task EMMs & Contrasts...")

pred_dict_task = {'Task': ['JF', 'SF']}
if 'Trial_Target' in formula_task:
    pred_dict_task['Trial_Target'] = [df_task['Trial_Target'].mean()] * 2
df_pred_task = pd.DataFrame(pred_dict_task)

# Manual EMM calculation via patsy 
formula_rhs_task = formula_task.split('~')[1]
X_task = dmatrix(formula_rhs_task, df_pred_task, return_type='dataframe')
fe_cols_task = [col for col in model_task.params.index if col in X_task.columns]
X_task_matrix, params_task, cov_matrix_task = X_task[fe_cols_task].values, model_task.params[fe_cols_task].values, model_task.cov_params().loc[fe_cols_task, fe_cols_task].values

resid_var_task = model_task.scale if hasattr(model_task, 'scale') else model_task.mse_resid
resid_sd_task = np.sqrt(resid_var_task)

emms_task_vals = X_task_matrix.dot(params_task)
se_emms_task = np.sqrt(np.diag(X_task_matrix.dot(cov_matrix_task).dot(X_task_matrix.T)))

df_emms_task = pd.DataFrame({
    'Task': df_pred_task['Task'],
    'EMM_Val': emms_task_vals,
    'SE': se_emms_task
})

# Contrast (SF vs JF)
# Using index 1 (SF) - index 0 (JF) to see if SF enhances/reduces relative to baseline JF
c_vec_task = X_task_matrix[1] - X_task_matrix[0]
diff_task = c_vec_task.dot(params_task)
se_diff_task = np.sqrt(c_vec_task.dot(cov_matrix_task).dot(c_vec_task.T))
z_stat_task = diff_task / se_diff_task
p_raw_task = 2 * stats.norm.sf(np.abs(z_stat_task))

df_contrasts_task = pd.DataFrame([{
    'Cond_A': 'SF', 'Cond_B': 'JF', 
    'Estimate': diff_task, 'SE': se_diff_task, 
    'z_stat': z_stat_task, 'd': diff_task / resid_sd_task,
    'CI_low': diff_task - (1.96 * se_diff_task), 
    'CI_high': diff_task + (1.96 * se_diff_task), 
    'p_raw': p_raw_task
}])

# Export EMMs
df_emms_task.to_csv(PAIRWISE_DIR / 'FAA_EMMs_Task_Raw.csv', index=False)
tex_emms_task = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    f"\\caption{{{metric_name} Estimated Marginal Means (EMMs) for Task Main Effect}}\n"
    "\\label{tab:faa_emms_task}\n"
    "\\begin{tabular}{@{}l S[table-format=1.4] S[table-format=1.4]@{}}\n"
    "\\toprule\n"
    "\\textbf{Task} & \\textbf{EMM} & \\textbf{SE} \\\\\n"
    "\\midrule\n"
)
for _, row in df_emms_task.iterrows():
    tex_emms_task += f"{row['Task']} & {row['EMM_Val']:.4f} & {row['SE']:.4f} \\\\\n"
tex_emms_task += "\\bottomrule\n\\end{tabular}\n\\end{table}"
with open(PAIRWISE_DIR / 'FAA_EMMs_Task_Raw.tex', 'w') as f: f.write(tex_emms_task)

# Export Contrasts
df_contrasts_task.to_csv(PAIRWISE_DIR / 'FAA_Contrasts_Task_Raw.csv', index=False)
tex_contrasts_task = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    "\\caption{Pairwise contrasts of FAA (Task Main Effect)}\n"
    "\\label{tab:faa_contrasts_task}\n"
    "\\begin{tabular}{@{}ll S[table-format=1.3] S[table-format=1.2] l S[table-format=1.3]@{}}\n"
    "\\toprule\n"
    "\\textbf{Cond A} & \\textbf{Cond B} & \\textbf{$\\Delta$} & \\textbf{d} & \\textbf{95\\% CI} & \\textbf{$p$} \\\\\n"
    "\\midrule\n"
)
for _, row in df_contrasts_task.iterrows():
    tex_contrasts_task += f"{row['Cond_A']} & {row['Cond_B']} & {row['Estimate']:.3f} & {row['d']:.2f} & [{row['CI_low']:.2f}, {row['CI_high']:.2f}] & {row['p_raw']:.3f} \\\\\n"
tex_contrasts_task += "\\bottomrule\n\\end{tabular}\n\\end{table}"
with open(PAIRWISE_DIR / 'FAA_Contrasts_Task_Raw.tex', 'w') as f: f.write(tex_contrasts_task)

print("Task EMMs and Pairwise Contrasts successfully calculated and exported to individual CSV and TeX files.")

#%%