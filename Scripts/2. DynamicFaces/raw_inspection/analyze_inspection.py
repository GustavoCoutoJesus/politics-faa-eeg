#!/usr/bin/env python3
"""
Summarizes and visualizes the results of the raw EEG inspection.
Reads from 'raw_inspection.csv' and generates a summary dashboard.
"""
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze raw EEG inspection results.")
    parser.add_argument("--csv", type=Path, default=Path("raw_inspection.csv"),
                        help="Path to the raw_inspection.csv tracker file.")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: Could not find '{args.csv}'. Make sure you are in the correct folder.")
        return

    # 1. Load Data
    df = pd.read_csv(args.csv)
    
    # 2. Data Cleaning & Parsing
    # Standardize 'Bad_Data' column (handle NaNs for participants that were NOT marked bad)
    if 'Bad_Data' in df.columns:
        df['Bad_Data_Flag'] = df['Bad_Data'].apply(lambda x: 'Bad Data' if str(x).strip() == 'bad_data' else 'Usable Data')
    else:
        df['Bad_Data_Flag'] = 'Usable Data' # Fallback if column doesn't exist

    # Extract all bad channels into a single list to count frequencies
    # We drop NaNs, convert to string, split by comma, and strip whitespace
    all_bad_channels = []
    if 'Bad_Channels' in df.columns:
        valid_channel_rows = df['Bad_Channels'].dropna().astype(str)
        for row in valid_channel_rows:
            channels = [ch.strip() for ch in row.split(',') if ch.strip()]
            all_bad_channels.extend(channels)
            
    channel_counts = Counter(all_bad_channels)

    # 3. Terminal Summary
    print("\n" + "="*50)
    print("🧠 EEG INSPECTION SUMMARY")
    print("="*50)
    print(f"Total Files Inspected:  {len(df)}")
    
    bad_data_count = (df['Bad_Data_Flag'] == 'Bad Data').sum()
    print(f"Files Marked as 'Bad':  {bad_data_count} ({bad_data_count/len(df)*100:.1f}%)")
    print(f"Usable Files Remaining: {len(df) - bad_data_count}")
    
    print("\n--- Bad Channels Stats ---")
    print(f"Average Bad Channels per subject: {df['Num_Bad_Channels'].mean():.2f}")
    print(f"Max Bad Channels in one subject:  {df['Num_Bad_Channels'].max()}")
    
    if channel_counts:
        top_5 = channel_counts.most_common(5)
        print("\nTop 5 Most Frequently Bad Channels:")
        for ch, count in top_5:
            print(f"  - {ch}: {count} participants")
    print("="*50 + "\n")

    # 4. Visualization Dashboard
    plt.style.use('ggplot') # Gives the plots a nice aesthetic
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title("EEG Data Quality Dashboard")

    # Top-Left: Distribution of Bad Channel Counts
    ax1 = axes[0, 0]
    ax1.hist(df['Num_Bad_Channels'].dropna(), bins=range(0, df['Num_Bad_Channels'].max() + 2), 
             color='skyblue', edgecolor='black', align='left')
    ax1.set_title('Distribution of Bad Channels per Participant')
    ax1.set_xlabel('Number of Bad Channels')
    ax1.set_ylabel('Number of Participants')
    ax1.set_xticks(range(0, df['Num_Bad_Channels'].max() + 1))

    # Top-Right: Top 15 Most Frequently Bad Channels
    ax2 = axes[0, 1]
    if channel_counts:
        # Get top 15
        top_n = 15
        most_common = channel_counts.most_common(top_n)
        channels, counts = zip(*most_common)
        
        # Plot horizontal bar chart
        ax2.barh(channels, counts, color='coral', edgecolor='black')
        ax2.invert_yaxis() # Put the highest frequency at the top
        ax2.set_title(f'Top {min(top_n, len(channels))} Most Frequently Bad Channels')
        ax2.set_xlabel('Number of Participants')
    else:
        ax2.text(0.5, 0.5, "No bad channels marked!", ha='center', va='center')
        ax2.set_title('Top Bad Channels')

    # Bottom-Left: Usable vs Bad Data Pie Chart
    ax3 = axes[1, 0]
    status_counts = df['Bad_Data_Flag'].value_counts()
    colors = ['#8fd9b6' if idx == 'Usable Data' else '#ff9999' for idx in status_counts.index]
    ax3.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    ax3.set_title('Overall Data Quality (Usable vs. Bad Data)')

    # Bottom-Right: Interval Length Distribution
    ax4 = axes[1, 1]
    if 'Interval_Length_mins' in df.columns:
        ax4.hist(df['Interval_Length_mins'].dropna(), bins=15, color='mediumpurple', edgecolor='black')
        ax4.set_title('Task Interval Length Distribution')
        ax4.set_xlabel('Length (minutes)')
        ax4.set_ylabel('Number of Participants')
    else:
        ax4.text(0.5, 0.5, "Interval Length data not found", ha='center', va='center')

    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    main()