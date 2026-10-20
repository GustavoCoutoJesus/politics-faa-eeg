#!/usr/bin/env python3
"""Interactive raw-EEG viewer and Progress Tracker.

Loops through EEG files in a directory, opens an MNE-readable file in
a scrolling raw plot, and saves bad-channel marks and annotations to
separate CSV files to preserve the original raw data.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import mne

# Define the target events that should be colored RED
TARGET_EVENTS = {
    61551, 61552, 61553, 61561, 61562, 61563,
    61571, 61572, 61573, 61581, 61582, 61583,
    61651, 61652, 61653, 61661, 61662, 61663,
    61671, 61672, 61673, 61681, 61682, 61683
}

def read_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True)
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True)
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=True)
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=True)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True)
    return mne.io.read_raw(path, preload=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path,
                        help="Path to the directory containing raw EEG files")
    parser.add_argument("--tracker", type=Path, default=Path("raw_inspection.csv"),
                        help="Path to the progress tracker CSV")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Window length in seconds (default: 10)")
    parser.add_argument("--n-channels", type=int, default=32,
                        help="Channels visible at once (default: 32)")
    parser.add_argument("--scaling", type=float, default=50e-6,
                        help="EEG scaling in volts (default: 50e-6)")
    parser.add_argument("--highpass", type=float, default=1,
                        help="Display-only high-pass in Hz (default: 1.0)") 
    parser.add_argument("--lowpass", type=float, default=40.0,
                        help="Display-only low-pass in Hz (default: 40)")
    
    args = parser.parse_args()

    if not args.data_dir.exists():
        parser.error(f"Directory not found: {args.data_dir}")

    # 1. Initialize or Load Tracker
    tracker_path = args.tracker
    tracker_columns = ['Filename', 'Subject', 'Num_Bad_Channels', 'Bad_Channels', 'Interval_Length_mins', 'Bad_Data']
    
    if tracker_path.exists():
        tracker_df = pd.read_csv(tracker_path)
        print(f"Loaded existing progress tracker from {tracker_path}")
        for col in tracker_columns:
            if col not in tracker_df.columns:
                tracker_df[col] = None
    else:
        tracker_df = pd.DataFrame(columns=tracker_columns)
        print(f"Created new progress tracker: {tracker_path.name}")

    # 2. Find all EEG files
    all_files = sorted(list(args.data_dir.rglob("*.bdf")))
    if not all_files:
        print(f"No .bdf files found in {args.data_dir}")
        sys.exit(0)

    # Automatically start at the first uninspected file
    start_idx = 0
    if not tracker_df.empty:
        for i, fp in enumerate(all_files):
            if fp.name not in tracker_df['Filename'].values:
                start_idx = i
                break
        else:
            start_idx = len(all_files) - 1  # If all are done, default to the last one

    current_idx = start_idx

    # 3. Main Subject Loop (Index-based for forward/back/jump navigation)
    while 0 <= current_idx < len(all_files):
        file_path = all_files[current_idx]
        filename_str = file_path.name
        
        print("\n" + "="*60)
        print(f"Inspecting File {current_idx + 1}/{len(all_files)}: {filename_str}")
        print("="*60)

        # Load Data
        try:
            raw = read_raw(file_path)
            events = mne.find_events(raw, min_duration=0.01, output='onset', verbose=False)
        except Exception as e:
            print(f"Error loading {filename_str}: {e}")
            current_idx += 1
            continue

        # Extract Stats Before Cropping
        raw_length_mins = raw.times[-1] / 60.0
        total_events = len(events)
        target_event_rows = [row for row in events if row[2] in TARGET_EVENTS]
        target_count = len(target_event_rows)

        # Calculate Cropping Boundaries
        if target_event_rows:
            first_sample = target_event_rows[0][0]
            last_sample = target_event_rows[-1][0]
            
            first_time = first_sample / raw.info['sfreq']
            last_time = last_sample / raw.info['sfreq']
            
            tmin = max(0, first_time - 180)
            tmax = min(raw.times[-1], last_time + 180)
            
            interval_length_mins = (tmax - tmin) / 60.0
        else:
            tmin = 0
            tmax = raw.times[-1]
            interval_length_mins = raw_length_mins

        # --- PRE-LOAD EXISTING EDITS (If navigating back to a finished file) ---
        is_bad_data = [False]
        existing_row = tracker_df[tracker_df['Filename'] == filename_str]
        
        if not existing_row.empty:
            print("--> Notice: This file was previously inspected. Loading existing marks...")
            
            # Pre-load bad channels
            saved_bads = existing_row.iloc[0]['Bad_Channels']
            if pd.notna(saved_bads) and isinstance(saved_bads, str) and saved_bads.strip():
                raw.info['bads'] = [b.strip() for b in saved_bads.split(',')]
            
            # Pre-load bad data flag
            saved_bad_data = existing_row.iloc[0]['Bad_Data']
            if pd.notna(saved_bad_data) and saved_bad_data == "bad_data":
                is_bad_data[0] = True

        # --- SETUP DASHBOARD WINDOW ---
        dash_fig = plt.figure(figsize=(10, 4))
        dash_fig.canvas.manager.set_window_title(f"Info Page: {filename_str}")
        
        # Add Stats Text
        stats_text = (
            f"FILE: {filename_str}\n"
            f"----------------------------------------\n"
            f"Total Events Found:   {total_events}\n"
            f"Target Events Found:  {target_count}\n\n"
            f"Raw File Length:      {raw_length_mins:.2f} mins\n"
            f"Cropped View Length:  {interval_length_mins:.2f} mins\n"
            f"(+3 min buffer applied)"
        )
        dash_fig.text(0.05, 0.55, stats_text, va='bottom', fontsize=10, family='monospace')

        # Add "Mark as Bad" Button
        ax_btn = plt.axes([0.7, 0.65, 0.25, 0.15])
        btn_bad = Button(ax_btn, 'Mark Data as BAD', color='lightcoral', hovercolor='red')

        # Initialize button state visually based on pre-loaded data
        if is_bad_data[0]:
            btn_bad.label.set_text("MARKED AS BAD (Click to Undo)")
            btn_bad.color = 'darkred'
            btn_bad.label.set_color('white')

        def toggle_bad_data(event):
            is_bad_data[0] = not is_bad_data[0]
            if is_bad_data[0]:
                btn_bad.label.set_text("MARKED AS BAD (Click to Undo)")
                btn_bad.color = 'darkred'
                btn_bad.label.set_color('white')
            else:
                btn_bad.label.set_text("Mark Data as BAD")
                btn_bad.color = 'lightcoral'
                btn_bad.label.set_color('black')
            dash_fig.canvas.draw()
            
        btn_bad.on_clicked(toggle_bad_data)

        # Add Event Timeline Map
        ax_map = plt.axes([0.05, 0.15, 0.9, 0.3])
        if total_events > 0:
            event_times = events[:, 0] / raw.info['sfreq']
            event_colors_map = ['red' if ev in TARGET_EVENTS else 'cyan' for ev in events[:, 2]]
            
            ax_map.vlines(event_times, ymin=0, ymax=1, colors=event_colors_map, linewidth=0.5)
            ax_map.axvspan(tmin, tmax, color='yellow', alpha=0.2, label='Cropped Window')
            ax_map.legend(loc='upper right', fontsize=8)

        ax_map.set_xlim(0, raw.times[-1])
        ax_map.set_yticks([])
        ax_map.set_xlabel("Time (seconds)")
        ax_map.set_title("Full Raw Timeline Map (Red = Targets, Cyan = Others)", fontsize=10)

        plt.show(block=False)

        # --- EXECUTE CROPPING AND MNE PLOT ---
        print(f"Cropping data: {tmin:.1f}s to {tmax:.1f}s (+ 3min buffers).")
        raw.crop(tmin=tmin, tmax=tmax)
        
        cropped_events = mne.find_events(raw, min_duration=0.01, output='onset', verbose=False)
        unique_events = set(cropped_events[:, 2]) if len(cropped_events) > 0 else set()
        event_colors = {ev: 'red' if ev in TARGET_EVENTS else 'cyan' for ev in unique_events}

        print("--> Window opened. Mark bad channels by clicking channel names.")
        print("--> Close the graphical window when you are finished to save.")
        
        plot_events = cropped_events if len(cropped_events) > 0 else None

        raw.plot(
            duration=args.duration,
            n_channels=args.n_channels,
            scalings=dict(eeg=args.scaling),
            highpass=args.highpass,
            lowpass=args.lowpass,
            events=plot_events,
            event_color=event_colors,
            filtorder=4,
            remove_dc=True,
            show_scrollbars=True,
            show_scalebars=True,
            block=True,
            title=f"MNE Timeseries: {filename_str}"
        )

        plt.close(dash_fig)

        # Extract manual edits
        bad_channels = raw.info['bads']
        annotations = raw.annotations

        # Save Annotations
        if len(annotations) > 0:
            annot_path = file_path.parent / f"{file_path.stem}_annotations.csv"
            annotations.save(annot_path, overwrite=True)
            print(f"Saved {len(annotations)} annotations to {annot_path.name}")

        # Update Progress Tracker (Overwrite row if it exists, otherwise append)
        bad_data_status = "bad_data" if is_bad_data[0] else ""
        
        new_row_data = {
            'Filename': filename_str,
            'Subject': file_path.parent.parent.name,
            'Num_Bad_Channels': len(bad_channels),
            'Bad_Channels': ", ".join(bad_channels),
            'Interval_Length_mins': round(interval_length_mins, 2),
            'Bad_Data': bad_data_status
        }
        
        idx_in_tracker = tracker_df.index[tracker_df['Filename'] == filename_str].tolist()
        
        if idx_in_tracker:
            for col, val in new_row_data.items():
                tracker_df.at[idx_in_tracker[0], col] = val
        else:
            tracker_df = pd.concat([tracker_df, pd.DataFrame([new_row_data])], ignore_index=True)
            
        tracker_df.to_csv(tracker_path, index=False)
        print(f"Saved progress! Marked {len(bad_channels)} bad channels. Bad Data Flag: {is_bad_data[0]}")

        # --- NAVIGATION PROMPT LOOP ---
        while True:
            print("\nSelect next action:")
            print("  [F] Forward (Next Subject)")
            print("  [B] Back (Previous Subject)")
            print("  [O] Open this subject again")
            print("  [S] Select a specific subject from list")
            print("  [N] eNd / Quit Tracker")
            
            user_choice = input("Choice: ").strip().upper()
            
            if user_choice == 'F':
                current_idx += 1
                break
                
            elif user_choice == 'B':
                if current_idx > 0:
                    current_idx -= 1
                else:
                    print("--> Already at the first file.")
                break
                
            elif user_choice == 'O':
                # Leave current_idx the same to reload
                break
                
            elif user_choice == 'S':
                print("\n--- Available Subjects ---")
                for i, fpath in enumerate(all_files):
                    status = "[Done]" if fpath.name in tracker_df['Filename'].values else "[    ]"
                    print(f"{i:02d}: {status} {fpath.name}")
                print("--------------------------")
                
                try:
                    sel = input("Enter the number of the subject to load (or press Enter to cancel): ").strip()
                    if not sel:
                        print("Selection cancelled.")
                        continue
                    
                    sel_idx = int(sel)
                    if 0 <= sel_idx < len(all_files):
                        current_idx = sel_idx
                        break
                    else:
                        print(f"Invalid choice. Please pick a number between 0 and {len(all_files)-1}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
                
            elif user_choice == 'N':
                print("Exiting inspector. Your progress is completely saved.")
                sys.exit(0)
                
            else:
                print("Invalid input. Please choose F, B, O, S, or N.")

    print("\nAll files in the directory have been processed.")

if __name__ == "__main__":
    main()