# %% #######################################################################################################################
# The goal of this script is to run Multiverse analysis and find the combination of specifications that increases validation effect size
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

# %% Master Multiverse Analysis: Manipulation Check (Angry vs Happy Non-Pols)
import os
import time
import mne
import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
import scipy.stats as stats
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress convergence warnings to keep the console output clean during the massive loop
warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# Configuration & Paths
# ==============================================================================
PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives'
MULTIVERSE_DIR = DERIVATIVES_DIR / 'multiverse'

# New Subdirectory for Manipulation Check
MANIP_DIR = MULTIVERSE_DIR / 'manipulation_check'
MANIP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# DASHBOARD CONFIGURATION
# ---------------------------------------------------------
# Choose which Top 3 rankings to display in the HTML dashboard.
# Options: "Effect_Size" (Largest Std Estimate), "AIC" (Lowest AIC), "BIC" (Lowest BIC)
DISPLAY_METRICS = ["Effect_Size", "AIC", "BIC"]

# ---------------------------------------------------------
# STATISTICAL ROBUSTNESS CONTROLS
# ---------------------------------------------------------
RUN_LOO_BOOTSTRAP = False     
RUN_PERMUTATION = False       
N_PERMUTATIONS = 1000        
# ---------------------------------------------------------

# Dimensions of the multiverse
BATCHES = {
    "Standard": {
        "folder": "Batch_01_Emotion_VirtualCz_8-12Hz",
        "file_suffix": "faa_diff-epo.fif",
        "exclusions": ["Strict", "Lenient", "Relaxed"],
        "qc_file": "Comprehensive_Exclusions_List.csv"
    },
    "Relaxed": {
        "folder": "Batch_02_Emotion_VirtualCz_8-12Hz_relaxed",
        "file_suffix": "faa_diff_iqr-epo.fif",
        "exclusions": ["Strict", "Lenient_B2", "Relaxed"], 
        "qc_file": "Participant_Level_QC_Flags.csv"
    }
}

MODELS = {
    "M1_GLM_Base": {"type": "ols", "add_trial": False},
    "M2_GLM_Trial": {"type": "ols", "add_trial": True},
    "M3_LMM_Base": {"type": "mixedlm", "add_trial": False},
    "M4_LMM_Trial": {"type": "mixedlm", "add_trial": True}
}

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

# ==============================================================================
# Helper Functions
# ==============================================================================
def parse_event_string(ev_str):
    group = "Non-Politician" if "nonpol" in ev_str else "In-Party" if "inparty" in ev_str else "Out-Party"
    emotion = "Angry" if "angry" in ev_str else "Happy" if "happy" in ev_str else "Neutral"
    return group, emotion

def get_excluded_subjects(batch_info, criterion):
    if criterion == "Relaxed": return set()
        
    qc_path = DERIVATIVES_DIR / 'quality_check' / batch_info['folder'] / batch_info['qc_file']
    exclusions = set()
    
    if not qc_path.exists():
        print(f"  [!] Missing QC file {qc_path.name}. Defaulting to no exclusions.")
        return exclusions
        
    df = pd.read_csv(qc_path)
    if batch_info['folder'].startswith("Batch_01"):
        col_name = f'Exclude_{criterion}'
        if col_name in df.columns:
            exclusions.update(df[df[col_name] == True]['Subject'].unique())
    else:
        for _, row in df.iterrows():
            sub = row['Subject']
            bad_data = (row.get('Tracker_Bad_Data', False) == True)
            bad_f3f4 = (pd.notna(row.get('Bad_F3_F4', "None")) and row.get('Bad_F3_F4', "None") != "None")
            if criterion == "Strict" and (bad_data or bad_f3f4): exclusions.add(sub)
            elif criterion == "Lenient_B2" and bad_data: exclusions.add(sub)
    return exclusions

def extract_single_trials(batch_info, excluded_subs):
    proc_dir = DERIVATIVES_DIR / 'processed' / batch_info['folder']
    subject_folders = sorted([d for d in proc_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    all_trials = []
    
    for subj_dir in subject_folders:
        subject = subj_dir.name
        if subject in excluded_subs: continue
        epo_file = subj_dir / 'eeg' / f"{subject}_task-emotion_{batch_info['file_suffix']}" 
        if not epo_file.exists(): continue
            
        epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
        t_static = (epochs.times >= 0.800) & (epochs.times <= 1.900)
        t_dynamic = (epochs.times >= 1.950) & (epochs.times <= 4.000)
        faa_data = epochs.get_data(picks='FAA')[:, 0, :]
        
        for i, event_id in enumerate(epochs.events[:, 2]):
            event_name = ID_TO_EVENT.get(event_id)
            if not event_name: continue
            group, emotion = parse_event_string(event_name)
            
            # FAST FILTER: Only append Non-Pols and Angry/Happy
            if group != "Non-Politician" or emotion not in ["Angry", "Happy"]:
                continue
                
            trial_data = faa_data[i, :]
            val_static = np.mean(trial_data[t_static])
            val_dynamic = np.mean(trial_data[t_dynamic])
            
            all_trials.append({
                'Subject': subject, 'Trial': i + 1, 'Group': group, 
                'Emotion': emotion, 'FAA_Diff': val_dynamic - val_static
            })
            
    df = pd.DataFrame(all_trials)
    if not df.empty:
        # Standardize strictly within this subset
        df['z_FAA'] = (df['FAA_Diff'] - df['FAA_Diff'].mean()) / df['FAA_Diff'].std()
        df['z_Trial'] = (df['Trial'] - df['Trial'].mean()) / df['Trial'].std()
    return df

def run_manipulation_models(df, formula_base, predictor_col, batch_name, exc_name):
    results = []
    n_trials = len(df)
    n_subs = df['Subject'].nunique()
    
    for mod_name, mod_info in MODELS.items():
        formula_std = f"z_FAA ~ {formula_base} + z_Trial" if mod_info['add_trial'] else f"z_FAA ~ {formula_base}"
        formula_unstd = f"FAA_Diff ~ {formula_base} + z_Trial" if mod_info['add_trial'] else f"FAA_Diff ~ {formula_base}"
        
        res_dict = {
            'Batch': batch_name, 'Exclusion': exc_name, 'model': mod_name, 
            'Formula': formula_std, 'N_Subjects': n_subs, 'N_Trials': n_trials,
            'AIC': np.nan, 'BIC': np.nan, 'Converged': False,
            'Target_Predictor': predictor_col, 
            'Std_Estimate': np.nan, 'Std_SE': np.nan, 'p_value': np.nan,
            'Unstd_Estimate': np.nan, 'Unstd_SE': np.nan
        }
        
        try:
            # Fit Standardized Model
            if mod_info['type'] == 'ols': 
                mod_std = smf.ols(formula_std, data=df).fit()
                mod_unstd = smf.ols(formula_unstd, data=df).fit()
            else: 
                mod_std = smf.mixedlm(formula_std, data=df, groups=df['Subject']).fit(reml=False)
                mod_unstd = smf.mixedlm(formula_unstd, data=df, groups=df['Subject']).fit(reml=False)
                
            res_dict['AIC'] = mod_std.aic
            res_dict['BIC'] = mod_std.bic
            res_dict['Converged'] = True
            
            # Extract Parameters
            for param in mod_std.params.index:
                if predictor_col in param: 
                    res_dict['Std_Estimate'] = mod_std.params[param]
                    res_dict['Std_SE'] = mod_std.bse[param]
                    res_dict['p_value'] = mod_std.pvalues[param]
                    
                    # Get unstandardized equivalents
                    res_dict['Unstd_Estimate'] = mod_unstd.params[param]
                    res_dict['Unstd_SE'] = mod_unstd.bse[param]
                    break

        except Exception as e:
            res_dict['Error'] = str(e)
            
        results.append(res_dict)
    return results

# ==============================================================================
# Main Execution Loop
# ==============================================================================
print("===============================================================================")
print("INITIALIZING MANIPULATION CHECK MULTIVERSE (Angry vs Happy | Non-Pols)")
print("===============================================================================")

master_results = []

for batch_name, batch_info in BATCHES.items():
    print(f"\nProcessing {batch_name}...")
    for exc_criterion in batch_info['exclusions']:
        print(f"  -> Exclusion Criterion: {exc_criterion}")
        excluded_subs = get_excluded_subjects(batch_info, exc_criterion)
        df_trials = extract_single_trials(batch_info, excluded_subs)
        
        if df_trials.empty: continue
            
        # Target Contrast: Angry vs Happy (Baseline = Happy)
        res = run_manipulation_models(df_trials, "C(Emotion, Treatment('Happy'))", "T.Angry", batch_name, exc_criterion)
        master_results.extend(res)

# Compile to DataFrame
df_master = pd.DataFrame(master_results)
df_master['Significant'] = df_master['p_value'] < 0.05
# Calculate absolute effect size for ranking
df_master['Abs_Std_Estimate'] = df_master['Std_Estimate'].abs()

master_csv_path = MANIP_DIR / "Manipulation_Master_Results.csv"
df_master.to_csv(master_csv_path, index=False)
print(f"\n✅ Manipulation computation complete. Master CSV saved.")

# ==============================================================================
# Output Generation: Specification Curve & HTML Dashboard
# ==============================================================================
print("\nGENERATING SPECIFICATION CURVES AND HTML DASHBOARD...")

def image_to_base64(img_path):
    if not img_path.exists(): return ""
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

# Sort by effect size for plotting
df_plot = df_master[df_master['Converged'] == True].copy()
df_plot.sort_values('Std_Estimate', inplace=True)
df_plot.reset_index(drop=True, inplace=True)
df_plot['Spec_ID'] = df_plot.index

FIXED_SPEC_ORDER = {
    'Batch': ['Standard', 'Relaxed'],
    'Exclusion': ['Strict', 'Lenient', 'Lenient_B2', 'Relaxed'],
    'model': ['M1_GLM_Base', 'M2_GLM_Trial', 'M3_LMM_Base', 'M4_LMM_Trial']
}

# --------------------------------------------------------------------------
# Plotting Specification Curve
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.08)

# Top Panel: Effect Sizes
ax_eff = fig.add_subplot(gs[0])
sig_mask = df_plot['Significant'] == True
non_sig_mask = df_plot['Significant'] == False

size_sig = df_plot[sig_mask]['N_Subjects'] * 3.5 
size_nonsig = df_plot[non_sig_mask]['N_Subjects'] * 3.5

# Plot Non-Significant
ax_eff.vlines(df_plot[non_sig_mask]['Spec_ID'], df_plot[non_sig_mask]['Std_Estimate'] - df_plot[non_sig_mask]['Std_SE']*1.96, df_plot[non_sig_mask]['Std_Estimate'] + df_plot[non_sig_mask]['Std_SE']*1.96, colors='lightgray', alpha=0.7, linewidth=1.5)
ax_eff.scatter(df_plot[non_sig_mask]['Spec_ID'], df_plot[non_sig_mask]['Std_Estimate'], c='silver', s=size_nonsig, alpha=0.8, zorder=3)

# Plot Significant
ax_eff.vlines(df_plot[sig_mask]['Spec_ID'], df_plot[sig_mask]['Std_Estimate'] - df_plot[sig_mask]['Std_SE']*1.96, df_plot[sig_mask]['Std_Estimate'] + df_plot[sig_mask]['Std_SE']*1.96, colors='black', alpha=0.9, linewidth=1.5)
ax_eff.scatter(df_plot[sig_mask]['Spec_ID'], df_plot[sig_mask]['Std_Estimate'], c='black', s=size_sig, alpha=1.0, zorder=4)

ax_eff.axhline(0, color='black', linestyle='--', linewidth=1.5)
ax_eff.set_ylabel("Standardized Mean Diff (Angry - Happy)", fontsize=12)
ax_eff.set_title("Manipulation Check: Specification Curve", fontsize=14, fontweight='bold')
ax_eff.set_xticks([]) 

# Add Legend
n_min, n_max = int(df_plot['N_Subjects'].min()), int(df_plot['N_Subjects'].max())
if n_min == n_max:
    legend_elements = [plt.scatter([], [], s=n_max * 3.5, color='gray', alpha=0.8, label=f'N = {n_max}')]
else:
    n_mid = int((n_min + n_max) / 2)
    legend_elements = [
        plt.scatter([], [], s=n_min * 3.5, color='gray', alpha=0.8, label=f'N = {n_min}'),
        plt.scatter([], [], s=n_mid * 3.5, color='gray', alpha=0.8, label=f'N = {n_mid}'),
        plt.scatter([], [], s=n_max * 3.5, color='gray', alpha=0.8, label=f'N = {n_max}')
    ]
ax_eff.legend(handles=legend_elements, title="Sample Size\n(Subjects)", loc='upper left', frameon=True, fontsize=10)

# Bottom Panel: Specification Matrix
ax_mat = fig.add_subplot(gs[1])
y_labels, y_ticks = [], []
current_y, color_idx = 0, 0
total_rows = sum([len(v) for v in FIXED_SPEC_ORDER.values()])
row_palette = sns.color_palette("husl", total_rows)

for cat in ['Batch', 'Exclusion', 'model']:
    for val in FIXED_SPEC_ORDER[cat]:
        if val in df_plot[cat].unique():
            y_labels.append(f"{cat}: {val}")
            y_ticks.append(current_y)
            row_color = row_palette[color_idx]
            spec_rows = df_plot[df_plot[cat] == val]
            
            b1_data = spec_rows[spec_rows['Batch'] == 'Standard']
            if not b1_data.empty: ax_mat.scatter(b1_data['Spec_ID'], [current_y]*len(b1_data), color=row_color, marker='o', s=60, alpha=0.85)

            b2_data = spec_rows[spec_rows['Batch'] == 'Relaxed']
            if not b2_data.empty: ax_mat.scatter(b2_data['Spec_ID'], [current_y]*len(b2_data), color=row_color, marker='^', s=60, alpha=0.85)
            
            current_y -= 1
        color_idx += 1
    current_y -= 1 

ax_mat.set_yticks(y_ticks)
ax_mat.set_yticklabels(y_labels, fontsize=11, ha='right', va='center')
ax_mat.set_xlim(ax_eff.get_xlim())
ax_mat.set_xlabel("Specification Number (Sorted by Effect Size)", fontsize=12)
ax_mat.grid(axis='y', linestyle=':', alpha=0.6)

plt.subplots_adjust(left=0.25, right=0.96, top=0.92, bottom=0.08)
fig_path = MANIP_DIR / "Manipulation_SpecCurve.png"
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------------------
# HTML Dashboard Construction
# --------------------------------------------------------------------------
top_3_cols = ['Batch', 'Exclusion', 'model', 'N_Subjects', 'Std_Estimate', 'Unstd_Estimate', 'p_value']
top_3_fit_cols = ['Batch', 'Exclusion', 'model', 'N_Subjects', 'AIC', 'BIC', 'p_value']

metric_sections_html = ""

if "Effect_Size" in DISPLAY_METRICS:
    top_3_eff = df_plot.sort_values('Abs_Std_Estimate', ascending=False).head(3).copy()
    metric_sections_html += f"""
    <div class="section">
        <h2>🏆 Top 3 Pipelines (Max Effect Size)</h2>
        <p class="description">These are the top three pipeline combinations that yielded the largest absolute standardized effect size (Cohen's d equivalent) for the Angry - Happy contrast.</p>
        <div class="top-pipeline">
            {top_3_eff[top_3_cols].to_html(index=False, classes='data-table', float_format='%.4f')}
        </div>
    </div>
    """

if "AIC" in DISPLAY_METRICS:
    top_3_aic = df_plot.sort_values('AIC', ascending=True).head(3).copy()
    metric_sections_html += f"""
    <div class="section">
        <h2>📈 Top 3 Pipelines (Lowest AIC)</h2>
        <p class="description">These pipelines provide the best relative model fit according to the Akaike Information Criterion (penalizing for model complexity).</p>
        <div class="top-pipeline" style="border-left-color: #3498db; background: #ebf5fb;">
            {top_3_aic[top_3_fit_cols].to_html(index=False, classes='data-table', float_format='%.4f')}
        </div>
    </div>
    """

if "BIC" in DISPLAY_METRICS:
    top_3_bic = df_plot.sort_values('BIC', ascending=True).head(3).copy()
    metric_sections_html += f"""
    <div class="section">
        <h2>📉 Top 3 Pipelines (Lowest BIC)</h2>
        <p class="description">These pipelines provide the best relative model fit according to the Bayesian Information Criterion (harsher penalty for complexity than AIC).</p>
        <div class="top-pipeline" style="border-left-color: #9b59b6; background: #f5eef8;">
            {top_3_bic[top_3_fit_cols].to_html(index=False, classes='data-table', float_format='%.4f')}
        </div>
    </div>
    """


html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Emotion Project - Manipulation Check</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 20px; }}
        .header {{ background-color: #27ae60; color: #fff; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ margin: 0; font-size: 26px; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #27ae60; padding-bottom: 8px; margin-top: 50px; font-size: 22px; }}
        .container {{ max-width: 1300px; margin: auto; }}
        .section {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 40px; }}
        
        .top-pipeline {{ background: #e8f8f5; border: 1px solid #a3e4d7; padding: 25px; border-radius: 6px; margin-bottom: 30px; border-left: 5px solid #1abc9c; }}
        
        table.data-table {{ width: 100%; border-collapse: collapse; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); background: #fff; }}
        table.data-table th, table.data-table td {{ padding: 10px 15px; border: 1px solid #e0e0e0; text-align: left; }}
        table.data-table th {{ background-color: #f1f2f6; color: #2c3e50; font-weight: bold; }}
        table.data-table tr:hover {{ background-color: #f8f9fa; }}
        
        .img-container {{ text-align: center; margin-top: 30px; padding: 20px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; }}
        img {{ max-width: 100%; height: auto; }}
        .description {{ font-size: 15px; color: #555; margin-bottom: 20px; line-height: 1.5; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Manipulation Check: Angry vs. Happy Faces (Non-Politicians)</h1>
            <p>Finding the pipeline that maximizes effect size</p>
        </div>
        
        {metric_sections_html}

        <div class="section">
            <h2>📊 Specification Curve</h2>
            <p class="description">Visual distribution of the Standardized Mean Difference across all possible analytical decisions. Significant results are depicted in black.</p>
            <div class='img-container'>
                <img src='{image_to_base64(fig_path)}' alt='Manipulation Specification Curve' />
            </div>
        </div>

        <div class="section">
            <h2>📋 Master Results Table</h2>
            <p class="description">Complete output showing both Standardized and Unstandardized estimates across the entire multiverse, alongside fitness criteria.</p>
            <div style="max-height: 500px; overflow-y: scroll;">
                {df_master[['Batch', 'Exclusion', 'model', 'N_Subjects', 'Std_Estimate', 'Unstd_Estimate', 'AIC', 'BIC', 'p_value', 'Significant']].sort_values('Std_Estimate', ascending=False).to_html(index=False, classes='data-table', float_format='%.4f')}
            </div>
        </div>
    </div>
</body>
</html>
"""

dashboard_path = MANIP_DIR / "Manipulation_Dashboard.html"
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n✅ All individual tables, plots, and HTML Dashboard saved to:\n   {MANIP_DIR}")
print("===============================================================================")
print("MANIPULATION CHECK COMPLETED SUCCESSFULLY.")
print("===============================================================================")
# %%