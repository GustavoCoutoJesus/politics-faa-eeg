# %% #######################################################################################################################
# The goal of this script is to produce a Jitter Plot of Average FAA per Condition with EMMs and CIs/SEs
# Extracts single-trial FAA, aggregates to participant means, loads EMMs/Contrasts
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

import os
import mne
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats
import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BATCH_NAME = "Batch_01_Emotion_VirtualCz_8-12Hz"
FAA_METHOD = "diff"

USE_Z_SCORES = False  # Toggle: Set to False to plot unstandardized raw values

# Choose which p-value to use for the significance brackets. 
# Options: "p_raw" or "p_holm"
P_VALUE_METRIC = "p_raw"

# 1. Target Model Configuration
# For Happy Pol vs Non-Pol, use: "Mod4_HappyOnly"
# (Other options: "Mod1_AllFaces", "Mod2_AngryOnly", "Overall_AllEmotions", etc.)
MODEL_NAME = "Mod4_HappyOnly" 

# 2. Plot Text & Aesthetic Settings
PLOT_TITLE = "FAA Dynamic Difference: Happy Faces"
Y_AXIS_LABEL = "Leftward FAA"

# 3. Data Structure Definitions (Matches the structure of the chosen Model)
CONDITION_COL = "Pol_Category"  # The column separating the groups (e.g., "Pol_Category", "Group", "Emotion")
CONDITIONS_TO_PLOT = ["Non-Politician", "Politician"] # Exact names expected in the EMMs CSV
COLOR_MAP = {"Non-Politician": "#f39c12", "Politician": "#8e44ad", "In-Party": "#3498db", "Out-Party": "#e74c3c", "Neutral": "#bdc3c7", "Angry": "#c0392b", "Happy": "#2ecc71"}
X_TICK_LABELS = ["Non-Politician", "Politician"]

# 4. Data Filter (If you are targeting a subset, define it here so the background scatter matches)
# E.g., for Mod4_HappyOnly we only want to plot data where Emotion == "Happy"
DATA_FILTER_COL = "Emotion" # Set to None if no filter needed
DATA_FILTER_VAL = "Happy"   # Set to None if no filter needed

# Auto-append data type to labels for clarity
data_type_label = "Z-scored" if USE_Z_SCORES else "Raw"
#PLOT_TITLE += f" ({data_type_label})"
#Y_AXIS_LABEL += f" ({data_type_label})"
# -------------------------------

PROJECT_FOLDER = Path("DEFINE YOUR FOLDER NAME HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids_emotion' / 'derivatives'

# Directories
EEG_PROCESSED_DIR = DERIVATIVES_DIR / 'processed' / BATCH_NAME
PAIRWISE_DIR = DERIVATIVES_DIR / 'pairwise' / BATCH_NAME / FAA_METHOD
FIGURES_DIR = DERIVATIVES_DIR / 'figures' / BATCH_NAME / FAA_METHOD
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Exclusions 
EXCLUDE_SUBJECTS = {""}  

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

# ==============================================================================
# Helper Functions
# ==============================================================================
def set_matplotlib_defaults(fontsize: int = 9, fontfamily: str = 'Arial', usetex: bool = False):
    params = {
        'font.size': fontsize, 'font.family': 'sans-serif', 'font.sans-serif': [fontfamily],
        'text.usetex': usetex, 'axes.labelsize': fontsize, 'axes.titlesize': fontsize + 1,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'lines.linewidth': 1, 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    }
    plt.rcParams.update(params)

set_matplotlib_defaults(fontsize=11, fontfamily='Arial', usetex=False)

def _load_emm_and_contrasts(stats_dir, model_name, cond_col):
    """Reads FAA EMMs and Contrasts from model-specific CSVs, searching subfolders if needed."""
    emm_map, p_map = {}, {}
    stats_dir = Path(stats_dir)
    
    # 1. Look directly in stats_dir first, then search recursively in subfolders
    emm_file_name = f"EMMs_{model_name}.csv"
    con_file_name = f"Contrasts_{model_name}.csv"

    emm_matches = list(stats_dir.rglob(emm_file_name))
    con_matches = list(stats_dir.rglob(con_file_name))

    emm_path = emm_matches[0] if emm_matches else stats_dir / emm_file_name
    con_path = con_matches[0] if con_matches else stats_dir / con_file_name
    
    # 2. Parse EMMs
    if emm_path.exists():
        df_emm = pd.read_csv(emm_path)
        for _, r in df_emm.iterrows():
            cond = str(r[cond_col])
            
            emm_value = float(r["EMM_Val"])
            se_value = float(r["SE"])
            
            emm_map[cond] = {
                "EMM": emm_value, 
                "SE": se_value, 
                "CI_low": emm_value - (1.96 * se_value),
                "CI_high": emm_value + (1.96 * se_value)
            }
    else:
        print(f"Warning: EMM file not found -> {emm_path}")
            
    # 3. Parse Contrasts
    if con_path.exists():
        df_con = pd.read_csv(con_path)
        # Handle case-insensitive column naming ('p_raw' vs 'p_Holm' vs 'p_holm')
        p_col_candidates = [P_VALUE_METRIC, P_VALUE_METRIC.lower(), "p_holm", "p_Holm", "p_raw"]
        p_col = next((c for c in p_col_candidates if c in df_con.columns), None)

        if p_col:
            for _, r in df_con.iterrows():
                p_map[(str(r["Cond A"]), str(r["Cond B"]))] = float(r[p_col])
                p_map[(str(r["Cond B"]), str(r["Cond A"]))] = float(r[p_col])
        else:
            print(f"Warning: Neither '{P_VALUE_METRIC}' nor standard p-value columns found in {con_path}")
    else:
        print(f"Warning: Contrast file not found -> {con_path}")
            
    return emm_map, p_map

# ==============================================================================
# 1. Data Extraction (Standalone from preprocessed files)
# ==============================================================================
print(f"Extracting Participant-Level FAA Data ({data_type_label})...")
subject_folders = sorted([d for d in EEG_PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
all_trials = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    epo_file = subj_dir / 'eeg' / f"{subject}_task-emotion_faa_{FAA_METHOD}-epo.fif"
    if epo_file.exists():
        epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
        
        t_static = (epochs.times >= 0.800) & (epochs.times <= 1.900)
        t_dynamic = (epochs.times >= 1.950) & (epochs.times <= 4.000)
        faa_data = epochs.get_data(picks='FAA')[:, 0, :]
        
        for i, event_id in enumerate(epochs.events[:, 2]):
            event_name = ID_TO_EVENT.get(event_id)
            if not event_name: continue
            
            group, emotion = parse_event_string(event_name)
            pol_category = "Non-Politician" if group == "Non-Politician" else "Politician"
            
            trial_data = faa_data[i, :]
            val_static = np.mean(trial_data[t_static])
            val_dynamic = np.mean(trial_data[t_dynamic])
            diff_val = val_dynamic - val_static
            
            all_trials.append({
                'Participant': subject,
                'Group': group,
                'Pol_Category': pol_category,
                'Emotion': emotion,
                'Value': diff_val
            })
    else:
        print(f" -> Missing {FAA_METHOD} file for {subject}. Skipping...")

df_trials = pd.DataFrame(all_trials)

# Filter Data if targeted subset is configured
#if DATA_FILTER_COL and DATA_FILTER_VAL:
#    df_trials = df_trials[df_trials[DATA_FILTER_COL] == DATA_FILTER_VAL].copy()

# Standardize values across the selected subset to match EMMs
if USE_Z_SCORES:
    df_trials['Value'] = (df_trials['Value'] - df_trials['Value'].mean()) / df_trials['Value'].std()

# Aggregate to Subject-Level means for the plot
plot_df = df_trials.groupby(['Participant', CONDITION_COL])['Value'].mean().reset_index()

# ==============================================================================
# 2. Main Plotting Function
# ==============================================================================
def plot_faa_jitter(plot_df, stats_dir, model_name, cond_col, conditions, color_map, 
                    plot_title, y_label, faa_method="norm", y_lims=None, plot_jitter=False, 
                    error_type="ci", sig_bracket_lw=0.8, sig_star_fontsize=9, 
                    sig_row_gap=0.25, fig_width=4.5, x_tick_labels=None):
    """
    Highly configurable plotting function supporting dynamic condition tracking.
    """
    emm_map, p_map = _load_emm_and_contrasts(stats_dir, model_name, cond_col)
    
    fig = plt.figure(figsize=(fig_width, 4))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[0.20, 1.00], hspace=0.05)
    
    ax_sig = fig.add_subplot(gs[0])
    ax     = fig.add_subplot(gs[1], sharex=ax_sig)

    ax_sig.set_ylim(0, 1)
    ax_sig.axis("off")
    x_positions = np.arange(len(conditions))
    left_pad = 0.1

    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

    cond_to_vals = {c: plot_df.loc[plot_df[cond_col] == c, "Value"] for c in conditions}

    for i, cond in enumerate(conditions):
        cond_vals = cond_to_vals.get(cond, pd.Series([], dtype=float))
        if cond_vals.empty: continue

        # Plot Jitter
        if plot_jitter:
            jitter = (np.random.rand(len(cond_vals)) - 0.5) * 0.25
            ax.scatter(np.full(len(cond_vals), x_positions[i]) + jitter, 
                       cond_vals.values, s=18, alpha=0.3, 
                       color=color_map.get(cond, "grey"), edgecolor="none", zorder=1.2)

        # EMM / Error Bars
        if cond in emm_map:
            mean_val = emm_map[cond]["EMM"]
            if error_type.lower() == "se":
                yerr = np.array([[emm_map[cond]["SE"]], [emm_map[cond]["SE"]]])
            else:
                yerr = np.array([[mean_val - emm_map[cond]["CI_low"]], [emm_map[cond]["CI_high"] - mean_val]])
        else:
            mean_val = cond_vals.mean()
            n = len(cond_vals)
            if error_type.lower() == "se":
                err = (cond_vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            else:
                err = stats.t.ppf(0.975, n - 1) * (cond_vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            yerr = np.array([[err], [err]])

        ax.errorbar(x_positions[i], mean_val, yerr=yerr, color="black", capsize=0, fmt='o', markersize=5, zorder=2)

    ax.set_xlim(-0.5 - left_pad, len(conditions) - 0.5)
    ax_sig.set_xlim(-0.5 - left_pad, len(conditions) - 0.5)

    if y_lims:
        ax.set_ylim(y_lims)
    else:
        all_vals = np.concatenate([v for v in cond_to_vals.values() if not v.empty])
        y_min_auto, y_max_auto = all_vals.min(), all_vals.max()
        padding = (y_max_auto - y_min_auto) * 0.15
        ax.set_ylim(y_min_auto - padding, y_max_auto + padding)
    
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel(y_label)

    if x_tick_labels is None:
        x_tick_labels = conditions
        
    ax.xaxis.set_major_locator(FixedLocator(x_positions))
    ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))

    p_label = "p-raw" if P_VALUE_METRIC == "p_raw" else "Holm"
    ax_sig.set_title(f"{plot_title}", pad=8) #\n(Model: {model_name} | {error_type.upper()} | {p_label}
    sns.despine(ax=ax, offset=2, trim=True)

    p_label = "p-raw" if P_VALUE_METRIC == "p_raw" else "Holm"
    ax_sig.set_title(f"{plot_title}", pad=8) #\n(Model: {model_name} | {error_type.upper()} | {p_label}
    sns.despine(ax=ax, offset=2, trim=True)

    # -------------------------------------------------------------------------
    # Baseline & Motivational Direction Annotations
    # -------------------------------------------------------------------------
    # 1. Dashed baseline at y = 0
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)

    # 2. Baseline Labels (Both centered on base_x)
    # Adjust base_x to move the whole centered text block left/right
    base_x = -0.32          # Center anchor for both words
    base_y_offset = 0.005  # Equal vertical distance from y=0

    # Upper text: Just Faces 
    ax.text(
        base_x, base_y_offset, "Static Faces",
        color="black", alpha=0.55, fontsize=7.5,
        ha="center", va="bottom"
    )
    # Lower text: baseline 
    ax.text(
        base_x, -base_y_offset, "baseline",
        color="black", alpha=0.55, fontsize=7.5,
        ha="center", va="top"
    )

    # 3. Approach & Avoidance arrows shifted further right and away from y=0
    arrow_x = -0.40        # Shifted into the despined margin
    text_x_offset = 0.035  # Gap between arrow stem and text
    c_approach = "#1B7837"
    c_avoidance = "#D55E00"

    # Vertical bounds (starting away from the 0 line to avoid text overlap)
    y_start_pos = 0.035
    y_end_pos   = 0.155
    y_start_neg = -0.035
    y_end_neg   = -0.155

    # Upward Arrow: Approach
    ax.annotate(
        "", 
        xy=(arrow_x, y_end_pos), xytext=(arrow_x, y_start_pos),
        arrowprops=dict(arrowstyle="->", color=c_approach, lw=1.6, shrinkA=0, shrinkB=0)
    )
    ax.text(
        arrow_x - text_x_offset, (y_start_pos + y_end_pos) / 2, "Approach", 
        color=c_approach, fontsize=8.5, rotation=90, 
        ha="right", va="center", fontweight="semibold"
    )

    # Downward Arrow: Avoidance
    ax.annotate(
        "", 
        xy=(arrow_x, y_end_neg), xytext=(arrow_x, y_start_neg),
        arrowprops=dict(arrowstyle="->", color=c_avoidance, lw=1.6, shrinkA=0, shrinkB=0)
    )
    ax.text(
        arrow_x - text_x_offset, (y_start_neg + y_end_neg) / 2, "Avoidance", 
        color=c_avoidance, fontsize=8.5, rotation=90, 
        ha="right", va="center", fontweight="semibold"
    )

    # Significance Brackets
    sig_pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            a, b = conditions[i], conditions[j]
            p = p_map.get((a, b), np.nan)
            if not np.isnan(p) and p < 0.1:
                sig_pairs.append((i, j, p))

    # Significance Brackets
    sig_pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            a, b = conditions[i], conditions[j]
            p = p_map.get((a, b), np.nan)
            if not np.isnan(p) and p < 0.1:
                sig_pairs.append((i, j, p))

    sig_pairs.sort(key=lambda t: (t[1] - t[0])) 

    if sig_pairs:
        y_base = 0.2 
        for idx, (xi, xj, p) in enumerate(sig_pairs):
            x0, x1 = x_positions[xi], x_positions[xj]
            y = y_base + (idx * sig_row_gap) 
            ax_sig.plot([x0, x0, x1, x1], [y - 0.05, y, y, y - 0.05], lw=sig_bracket_lw, color="black", clip_on=False)
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "."))
            ax_sig.text((x0 + x1) / 2, y, stars, ha="center", va="bottom", fontsize=sig_star_fontsize)

    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, top=0.90)
    
    prefix = "FAA_JitterPlot" if plot_jitter else "FAA_EMMPlot"
    out_name = f"{prefix}_{faa_method}_{model_name}_{error_type.upper()}_{data_type_label}.png" 
    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.show()

# ==============================================================================
# 3. Execute Plotting
# ==============================================================================
print("\nPlotting Data...")

plot_faa_jitter(
    plot_df=plot_df, 
    stats_dir=PAIRWISE_DIR, 
    model_name=MODEL_NAME,           
    cond_col=CONDITION_COL,
    conditions=CONDITIONS_TO_PLOT,
    color_map=COLOR_MAP,
    plot_title=PLOT_TITLE,           
    y_label=Y_AXIS_LABEL,            
    y_lims=(-0.18,0.18),                    
    faa_method=FAA_METHOD, 
    plot_jitter=False,
    error_type="se",            
    sig_bracket_lw=1.0,         
    sig_star_fontsize=16,       
    sig_row_gap=0.5,            
    fig_width=4.0,                                               
    x_tick_labels=X_TICK_LABELS
)

# %% Cell 4: 3-Level Group Plot (Happy Faces: In-Party vs Out-Party vs Non-Pol)

print("\nConfiguring and Plotting 3-Level Group Data (Happy Faces)...")

# 1. New Target Model Configuration
MODEL_NAME_3LVL = "Happy_Only_Contrasts" 
CONDITION_COL_3LVL = "Group"
CONDITIONS_TO_PLOT_3LVL = ["Non-Politician", "Out-Party", "In-Party"] # Logical display order
X_TICK_LABELS_3LVL = ["Non-Pol", "Out-Party", "In-Party"]

PLOT_TITLE_3LVL = f"Happy Expression"

# 2. Re-aggregate Data
plot_df_3lvl = df_trials.groupby(['Participant', CONDITION_COL_3LVL])['Value'].mean().reset_index()

# 3. Execute Plotting
plot_faa_jitter(
    plot_df=plot_df_3lvl, 
    stats_dir=PAIRWISE_DIR, 
    model_name=MODEL_NAME_3LVL,           
    cond_col=CONDITION_COL_3LVL,
    conditions=CONDITIONS_TO_PLOT_3LVL,
    color_map=COLOR_MAP,           
    plot_title=PLOT_TITLE_3LVL,           
    y_label=Y_AXIS_LABEL,            
    y_lims=(-0.18, 0.18),              
    faa_method=FAA_METHOD, 
    plot_jitter=False,
    error_type="se",            
    sig_bracket_lw=1.0,         
    sig_star_fontsize=16,       
    sig_row_gap=0.5,            
    fig_width=4.0,                   # Made slightly wider (5.0 instead of 4.0) to fit 3 bars cleanly                            
    x_tick_labels=X_TICK_LABELS_3LVL
)
#%% Cell 4: 3-Level Group Plot (Neutral Faces: In-Party vs Out-Party vs Non-Pol)

print("\nConfiguring and Plotting 3-Level Group Data (Neutral Faces)...")

# 1. New Target Model Configuration
MODEL_NAME_3LVL = "Neutral_Only_Contrasts" 
CONDITION_COL_3LVL = "Group"
CONDITIONS_TO_PLOT_3LVL = ["Non-Politician", "Out-Party", "In-Party"] # Logical display order
X_TICK_LABELS_3LVL = ["Non-Pol", "Out-Party", "In-Party"]

PLOT_TITLE_3LVL = f"Neutral Expression"

# 2. Re-aggregate Data
plot_df_3lvl = df_trials.groupby(['Participant', CONDITION_COL_3LVL])['Value'].mean().reset_index()

# 3. Execute Plotting
plot_faa_jitter(
    plot_df=plot_df_3lvl, 
    stats_dir=PAIRWISE_DIR, 
    model_name=MODEL_NAME_3LVL,           
    cond_col=CONDITION_COL_3LVL,
    conditions=CONDITIONS_TO_PLOT_3LVL,
    color_map=COLOR_MAP,             
    plot_title=PLOT_TITLE_3LVL,           
    y_label=Y_AXIS_LABEL,            
    y_lims=(-0.18, 0.18),                
    faa_method=FAA_METHOD, 
    plot_jitter=False,
    error_type="se",            
    sig_bracket_lw=1.0,         
    sig_star_fontsize=16,       
    sig_row_gap=0.5,            
    fig_width=4.0,                                             
    x_tick_labels=X_TICK_LABELS_3LVL
)
#%% Cell 4: 3-Level Group Plot (Angry Faces: In-Party vs Out-Party vs Non-Pol)

print("\nConfiguring and Plotting 3-Level Group Data (Angry Faces)...")

# 1. New Target Model Configuration
MODEL_NAME_3LVL = "Angry_Only_Contrasts" 
CONDITION_COL_3LVL = "Group"
CONDITIONS_TO_PLOT_3LVL = ["Non-Politician", "Out-Party", "In-Party"] # Logical display order
X_TICK_LABELS_3LVL = ["Non-Pol", "Out-Party", "In-Party"]

PLOT_TITLE_3LVL = f"Angry Expression"

# 2. Re-aggregate Data
plot_df_3lvl = df_trials.groupby(['Participant', CONDITION_COL_3LVL])['Value'].mean().reset_index()

# 3. Execute Plotting
plot_faa_jitter(
    plot_df=plot_df_3lvl, 
    stats_dir=PAIRWISE_DIR, 
    model_name=MODEL_NAME_3LVL,           
    cond_col=CONDITION_COL_3LVL,
    conditions=CONDITIONS_TO_PLOT_3LVL,
    color_map=COLOR_MAP,            
    plot_title=PLOT_TITLE_3LVL,           
    y_label=Y_AXIS_LABEL,            
    y_lims=(-0.18, 0.18),               
    faa_method=FAA_METHOD, 
    plot_jitter=False,
    error_type="se",            
    sig_bracket_lw=1.0,         
    sig_star_fontsize=16,       
    sig_row_gap=0.5,            
    fig_width=4.0,                                             
    x_tick_labels=X_TICK_LABELS_3LVL
)

#%% Cell 5: Emotion Main Effect Plot (Neutral vs. Angry vs. Happy)

print("\nConfiguring and Plotting Emotion Main Effect Data...")

# 1. New Target Model Configuration
MODEL_NAME_EMOTION = "Overall_Emotion_Contrasts" 
CONDITION_COL_EMOTION = "Emotion"
CONDITIONS_TO_PLOT_EMOTION = ["Neutral", "Angry", "Happy"] # Logical display order
X_TICK_LABELS_EMOTION = ["Neutral", "Angry", "Happy"]

PLOT_TITLE_EMOTION = f"Non-politicians All Expressions"

# 2. Re-aggregate Data
plot_df_emotion = df_trials.groupby(['Participant', CONDITION_COL_EMOTION])['Value'].mean().reset_index()

# 3. Execute Plotting
plot_faa_jitter(
    plot_df=plot_df_emotion, 
    stats_dir=PAIRWISE_DIR, 
    model_name=MODEL_NAME_EMOTION,           
    cond_col=CONDITION_COL_EMOTION,
    conditions=CONDITIONS_TO_PLOT_EMOTION,
    color_map=COLOR_MAP,           
    plot_title=PLOT_TITLE_EMOTION,           
    y_label=Y_AXIS_LABEL,         
    y_lims=(-0.18, 0.18),            
    faa_method=FAA_METHOD, 
    plot_jitter=False,
    error_type="se",            
    sig_bracket_lw=1.0,         
    sig_star_fontsize=16,       
    sig_row_gap=0.5,            
    fig_width=4.0,                                         
    x_tick_labels=X_TICK_LABELS_EMOTION
)
# %%
