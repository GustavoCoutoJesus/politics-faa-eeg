# %% #######################################################################################################################
# The goal of this script is to produce a Jitter Plot of Average FAA per Condition with EMMs and CIs/SEs
# Extracts single-trial FAA, aggregates to participant means, loads EMMs/Contrasts
# Gustavo Couto de Jesus | @gustavocjesus.bsky.social | www.linkedin.com/in/gustavocj | https://github.com/GustavoCoutoJesus
############################################################################################################################

import os
import time
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
BATCH_NAME = "Batch_06_VirtualCz_8-12Hz"
FAA_METHOD = "diff"

# --- NEW USER CONFIGURATIONS ---
USE_Z_SCORES = False  # Toggle: Set to False to plot unstandardized raw values

# Choose which p-value to use for the significance brackets. 
# Options: "p_raw" or "p_Holm"
P_VALUE_METRIC = "p_raw"

# Choose which model's EMMs and Contrasts to load. 
# Options: "Raw_SF", "Party_Corrected_SF", "Politician_Corrected_SF"
MODEL_NAME = "Politician_Corrected_SF" 

# Customize plot text
PLOT_TITLE = "FAA Response to Statements & Faces"
Y_AXIS_LABEL = "Leftward FAA"

# Auto-append data type to labels for clarity
data_type_label = "Z-scored" if USE_Z_SCORES else "Raw"

PROJECT_FOLDER = Path("DEFINE YOUR PROJECT FOLDER HERE")
DERIVATIVES_DIR = PROJECT_FOLDER / 'bids' / 'derivatives'

# Where the Preprocessed Data lives
EEG_PROCESSED_DIR = DERIVATIVES_DIR / 'processed' / BATCH_NAME

# Where the Statistical CSVs live
PAIRWISE_DIR = DERIVATIVES_DIR / 'pairwise' / BATCH_NAME / FAA_METHOD

# Where the figure will be saved
FIGURES_DIR = DERIVATIVES_DIR / 'figures' / BATCH_NAME / FAA_METHOD
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_SUBJECTS = {"sub-P03", "sub-P11", "sub-P12"}  

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

def _load_emm_and_contrasts(stats_dir, model_name):
    """Reads FAA EMMs and Contrasts from model-specific CSVs."""
    emm_map, p_map = {}, {}
    stats_dir = Path(stats_dir)
    
    emm_path = stats_dir / f"FAA_EMMs_{model_name}.csv"
    con_path = stats_dir / f"FAA_Contrasts_{model_name}.csv"
    
    # Mapping dictionary to match the clean strings used in your new tables
    param_map = {
        "GL-left": ("In-party", "Congruent in-party face"),
        "GL-right": ("In-party", "Incongruent in-party face"),
        "PVV-left": ("Out-party", "Incongruent out-party face"),
        "PVV-right": ("Out-party", "Congruent out-party face")
    }
    
    if emm_path.exists():
        df_emm = pd.read_csv(emm_path)
        for p_key, (stmt_val, face_val) in param_map.items():
            r_match = df_emm[(df_emm['Statement'] == stmt_val) & (df_emm['Face pairing'] == face_val)]
            if not r_match.empty:
                r = r_match.iloc[0]
                emm_value = float(r["M"])
                se_value = float(r["SE"])
                emm_map[p_key] = {
                    "EMM": emm_value, 
                    "SE": se_value, 
                    "CI_low": emm_value - (1.96 * se_value),
                    "CI_high": emm_value + (1.96 * se_value)
                }
    else:
        print(f"Warning: EMM file not found -> {emm_path}")
            
    if con_path.exists():
        df_con = pd.read_csv(con_path)
        
        # Build an abbreviation mapping to read the contrast labels
        abbr_map = {
            "GL-left": "Cong. in- face",
            "GL-right": "Incong. in- face",
            "PVV-left": "Incong. out- face",
            "PVV-right": "Cong. out- face"
        }
        
        # Determine whether to extract 'p_raw' or 'p_Holm' based on user choice
        p_col = P_VALUE_METRIC if P_VALUE_METRIC in df_con.columns else "p_Holm"
        
        for _, r in df_con.iterrows():
            # Map the clean condition strings back to plotting labels
            cond_a_key = next((k for k, v in abbr_map.items() if v == r["Cond A"]), r["Cond A"])
            cond_b_key = next((k for k, v in abbr_map.items() if v == r["Cond B"]), r["Cond B"])
            
            p_map[(cond_a_key, cond_b_key)] = float(r[p_col])
            p_map[(cond_b_key, cond_a_key)] = float(r[p_col])
    else:
        print(f"Warning: Contrast file not found -> {con_path}")
            
    return emm_map, p_map

# ==============================================================================
# 1. Data Extraction
# ==============================================================================
print(f"Extracting Participant-Level FAA Data ({data_type_label})...")
subject_folders = sorted([d for d in EEG_PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
all_trials = []

for subj_dir in subject_folders:
    subject = subj_dir.name
    if subject in EXCLUDE_SUBJECTS: continue

    sf_file = subj_dir / 'eeg' / f"{subject}_task-StatementsFaces_faa_{FAA_METHOD}-epo.fif"
    if sf_file.exists():
        epochs = mne.read_epochs(sf_file, preload=True, verbose=False)
        time_mask = (epochs.times >= 0.0) & (epochs.times <= 1.2)
        faa_data = epochs.get_data(picks='FAA')[:, 0, :]
        mean_faa_per_trial = np.mean(faa_data[:, time_mask], axis=-1)
        
        for i, event_id in enumerate(epochs.events[:, 2]):
            event_name = ID_TO_EVENT.get(event_id)
            if not event_name: continue
            parts = event_name.split('/')
            all_trials.append({
                'Participant': subject,
                'Condition': f"{parts[0]}-{parts[2]}", # e.g. GL-left
                'Value': mean_faa_per_trial[i]
            })
    else:
        print(f" -> Missing {FAA_METHOD} file for {subject}. Skipping...")

df_trials = pd.DataFrame(all_trials)

# Ensure the background scatter points are standardizd if USE_Z_SCORES is True
if USE_Z_SCORES:
    df_trials['Value'] = (df_trials['Value'] - df_trials['Value'].mean()) / df_trials['Value'].std()

# Aggregate to Subject-Level means for the plot
plot_df = df_trials.groupby(['Participant', 'Condition'])['Value'].mean().reset_index()

# ==============================================================================
# 2. Main Plotting Function
# ==============================================================================
def plot_faa_jitter(plot_df, stats_dir, model_name, plot_title, y_label, faa_method="norm", 
                    y_lims=None, plot_jitter=False, error_type="ci", sig_bracket_lw=0.8, 
                    sig_star_fontsize=9, sig_row_gap=0.25, fig_width=4.5, 
                    x_tick_labels=None, group_label_y_pos=-0.18):
    """
    model_name: string, suffix for which EMMs/Contrasts to load (e.g., "Raw_SF").
    plot_title: string, sets the title of the figure.
    y_label: string, sets the Y-axis label.
    y_lims: tuple (low, high) or None for automatic.
    plot_jitter: bool, whether to plot underlying participant scatter dots (defaults to False).
    error_type: "ci" for 95% Confidence Intervals, "se" for Standard Error.
    sig_bracket_lw: line width for the significance brackets.
    sig_star_fontsize: font size for the significance asterisks.
    sig_row_gap: spacing between multiple layers of significance brackets.
    fig_width: float, adjusting the horizontal dimension of the resulting plot.
    x_tick_labels: list of strings to replace the bottom tick labels.
    group_label_y_pos: float, adjusts vertical position of 'In-party face' & 'Out-party face'.
    """
    conditions = ["PVV-right", "PVV-left", "GL-right", "GL-left"]
    color_map = {"PVV-right": "#E41A1C", "PVV-left": "#F781BF", "GL-right": "#1FA1A1", "GL-left": "#377EB8"}
    
    # Pass the config model_name down to the loader
    emm_map, p_map = _load_emm_and_contrasts(stats_dir, model_name)
    
    # Adjust width dynamically based on user input
    fig = plt.figure(figsize=(fig_width, 4.5))
    
    # height_ratios [0.20, 1.0] makes the top (sig) area much thinner
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[0.35, 1.00], hspace=0.05)
    
    ax_sig = fig.add_subplot(gs[0])
    ax     = fig.add_subplot(gs[1], sharex=ax_sig)

    ax_sig.set_ylim(0, 1)
    ax_sig.axis("off")
    x_positions = np.arange(len(conditions))
    left_pad = 0.1

    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

    cond_to_vals = {c: plot_df.loc[plot_df["Condition"] == c, "Value"] for c in conditions}

    for i, cond in enumerate(conditions):
        cond_vals = cond_to_vals.get(cond, pd.Series([], dtype=float))
        if cond_vals.empty: continue

        # Plot Jitter only if enabled 
        if plot_jitter:
            jitter = (np.random.rand(len(cond_vals)) - 0.5) * 0.25
            ax.scatter(np.full(len(cond_vals), x_positions[i]) + jitter, 
                       cond_vals.values, s=18, alpha=0.3, 
                       color=color_map.get(cond, "grey"), edgecolor="none", zorder=1.2)

        # EMM / Error Bars (Checking for CI vs SE preference)
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

        ax.errorbar(x_positions[i], mean_val, yerr=yerr, color="black", capsize=0, fmt='o', markersize=4, zorder=2)

    ax.set_xlim(-0.5 - left_pad, len(conditions) - 0.5)
    ax_sig.set_xlim(-0.5 - left_pad, len(conditions) - 0.5)

    # Custom Y-Axis Limits
    if y_lims:
        ax.set_ylim(y_lims)
    else:
        # Dynamic padding that accommodates the data spread
        all_vals = np.concatenate([v for v in cond_to_vals.values() if not v.empty])
        y_min_auto, y_max_auto = all_vals.min(), all_vals.max()
        padding = (y_max_auto - y_min_auto) * 0.15
        ax.set_ylim(y_min_auto - padding, y_max_auto + padding)
    
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    # Use the configurable Y_AXIS_LABEL
    ax.set_ylabel(y_label)

    # X-Axis Labels Setup
    if x_tick_labels is None:
        x_tick_labels = ["Congruent", "Incongruent", "Incongruent", "Congruent"]
        
    ax.xaxis.set_major_locator(FixedLocator(x_positions))
    ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))

    # X-Axis Group Labels with Adjustable Positioning
    trans = ax.get_xaxis_transform()
    ax.text(0.5, group_label_y_pos, "Out-party face", ha="center", va="top", transform=trans, fontweight='bold')
    ax.text(2.5, group_label_y_pos, "In-party face", ha="center", va="top", transform=trans, fontweight='bold')

    # Update Title Automatically based on Config Plot Title and Error Type
    #p_label = "p-raw" if P_VALUE_METRIC == "p_raw" else "Holm"
    ax_sig.set_title(f"{plot_title}", pad=2) #\n(Model: {model_name} | {error_type.upper()} | {p_label})
    sns.despine(ax=ax, offset=2, trim=True)

    # -------------------------------------------------------------------------
    # Baseline & Motivational Direction Annotations
    # -------------------------------------------------------------------------
    # 1. Dashed baseline at y = 0
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)

    # 2. Baseline Labels (Both centered on base_x)
    # --> Adjust base_x to move the whole centered text block left/right
    base_x = -0.3         # Center anchor for both words
    base_y_offset = 0.005  # Equal vertical distance from y=0

    # Upper text: Just Faces (Centered)
    ax.text(
        base_x, base_y_offset, "Just Faces",
        color="black", alpha=0.55, fontsize=8.5,
        ha="center", va="bottom"
    )
    # Lower text: baseline (Centered directly underneath)
    ax.text(
        base_x, -base_y_offset, "baseline",
        color="black", alpha=0.55, fontsize=8.5,
        ha="center", va="top"
    )

    # 3. Approach & Avoidance arrows shifted further right and away from y=0
    arrow_x = -0.40        # Shifted further right into the open despined margin
    text_x_offset = 0.035  # Gap between arrow stem and text
    c_approach = "#1B7837"
    c_avoidance = "#D55E00"

    # Vertical bounds (starting further away from the 0 line to avoid text overlap)
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
    # -------------------------------------------------------------------------

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
        # Pushed up brackets
        y_base = 0.2 
        for idx, (xi, xj, p) in enumerate(sig_pairs):
            x0, x1 = x_positions[xi], x_positions[xj]
            y = y_base + (idx * sig_row_gap) # Uses custom row gap
            ax_sig.plot([x0, x0, x1, x1], [y - 0.05, y, y, y - 0.05], lw=sig_bracket_lw, color="black", clip_on=False)
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "."))
            ax_sig.text((x0 + x1) / 2, y, stars, ha="center", va="bottom", fontsize=sig_star_fontsize)

    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, top=0.90)
    
    # Save the plot (Appends model name to output file)
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

# Call with your specified parameters and new configurations
plot_faa_jitter(
    plot_df=plot_df, 
    stats_dir=PAIRWISE_DIR, 
    model_name=MODEL_NAME,           # From Config
    plot_title=PLOT_TITLE,           # From Config
    y_label=Y_AXIS_LABEL,            # From Config
    y_lims=(-0.18,0.18),                
    faa_method=FAA_METHOD, 
    plot_jitter=False,
    error_type="se",            
    sig_bracket_lw=1.0,         
    sig_star_fontsize=12,       
    sig_row_gap=0.22,            
    fig_width=5.0,                                               
    x_tick_labels=["Congruent", "Incongruent", "Incongruent", "Congruent"], 
    group_label_y_pos=-0.1                                      
)

# %%