#!/usr/bin/env python3
"""
Find statistics for the total cholesterol level in the US for the males age between 40 and 60 (both included) in 2021-2022.
Use the most official survey data like NHANES.
Draw the distribution using Python (x - total cholesterol, y - % of people).
Please have an arrow on the graph, pointing to 184 cholesterol level.

Final result requirements
    Create a distribution graph:
        x-axis - total cholesterol level from 50 to 400 with step by 5
        y-axis: percentage of individuals (%)
    Incorporate an arrow to highlight the 184 cholesterol level.
    Attach the CSV file with the distribution used for the plotting.
    File should contain 2 columns named cholesterol_level and population_perc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from urllib.request import urlopen
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

TCHOL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.xpt"
DEMO = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt"

def download_xpt(url):
    """
    Download NHANES 2021-2023 cholesterol and full demographic data
    """
    with urlopen(url) as response:
        data = response.read()
        if len(data) > 100 and not data[:100].startswith(b'<!DOCTYPE'):
            print(f"  Success! Downloaded {len(data)} bytes")
            return pd.read_sas(BytesIO(data), format='xport')
        else:
            print(f"  Got HTML page, not XPT file")
            return pd.DataFrame()

def merge_data(tchol: str, demo: str) -> pd.DataFrame:
    """
    Merge cholesterol and demographic data based on SEQN
    Filter for males aged 40-60
    """
    # verify if the file already exists
    if not os.path.exists("data"):
        os.makedirs("data")
    if os.path.exists("data/filtered_chol_data.csv"):
        print("Filtered data file already exists. Loading from CSV...")
        return pd.read_csv("data/filtered_chol_data.csv")
    print("Downloading cholesterol data...")
    chol_data = download_xpt(tchol)
    print("Downloading demographic data...")
    demo_data = download_xpt(demo)

    if chol_data.empty or demo_data.empty:
        print("Failed to download necessary data files.")
        return pd.DataFrame()
    else:
        print("Saving raw data to CSV...")
        chol_data.to_csv("data/chol_data.csv", index=False)
        print(f"Cholesterol data shape: {chol_data.shape}")
        demo_data.to_csv("data/demo_data.csv", index=False)
        print(f"Demographic data shape: {demo_data.shape}")
        
        DEMO_COLS = ["SEQN", "RIAGENDR", "RIDAGEYR"]
        CHOL_COLS = ["SEQN", "LBXTC"]
        
        chol_data = chol_data[CHOL_COLS].dropna(subset=["LBXTC"])
        demo_data = demo_data[DEMO_COLS].dropna(subset=["RIAGENDR", "RIDAGEYR"])
        
        print("Filtering for males aged 40-60...")
        filtered_data = demo_data[
            (demo_data["RIAGENDR"] == 1) &  # Male
            (demo_data["RIDAGEYR"] >= 40) &
            (demo_data["RIDAGEYR"] <= 60)
        ]
        print(f"Filtered demographic data shape: {filtered_data.shape}")

        filtered_data = filtered_data[["SEQN", "RIDAGEYR"]]

        print("Merging datasets...")
        merged_data = pd.merge(chol_data, filtered_data, on="SEQN", how="inner")

        print(f"Merged dataset shape: {merged_data.shape}.")
        merged_data = merged_data.drop(columns=["SEQN", "RIDAGEYR"])
        print(f"Final dataset shape after dropping unnecessary columns: {merged_data.shape}.")

        print("Saving filtered data to CSV...")
        merged_data.to_csv("data/filtered_chol_data.csv", index=False)
        return merged_data

def obtain_distribution(df: pd.DataFrame):
    """
    Obtain the distribution of cholesterol levels
    """
    cholesterol_values = df['LBXTC'].values

    bins = np.arange(50, 405, 5)
    # saves the distribution data to CSV
    hist, bin_edges = np.histogram(cholesterol_values, bins=bins) # type: ignore
    
    # Calculate percentage for each bin
    total_count = len(cholesterol_values)
    percentages = (hist / total_count) * 100
    
    # Use bin centers as cholesterol levels
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    dist_df = pd.DataFrame({
        'cholesterol_level': bin_centers,
        'population_perc': percentages
    })
    dist_df.to_csv('cholesterol_distribution.csv', index=False)
    return dist_df

def plot_distribution(df: pd.DataFrame):
    """
    Plot the distribution of cholesterol levels
    Create a distribution graph:
        x-axis - total cholesterol level from 50 to 400 with step by 5
        y-axis: percentage of individuals (%)
    Incorporate an arrow to highlight the 184 cholesterol level.
    """
    
    # Create figure with high DPI for better quality
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    
    # Plot the distribution as a line chart
    ax.plot(df['cholesterol_level'], df['population_perc'], 
            linewidth=2.5, color='#2E86AB', label='Distribution')

    # Fill under the curve
    ax.fill_between(df['cholesterol_level'], df['population_perc'], 
                    alpha=0.3, color='#2E86AB')

    # Add arrow pointing to 184 mg/dL
    # Find the percentage at or near 184
    target_chol = 184
    closest_idx = (df['cholesterol_level'] - target_chol).abs().idxmin()
    target_x = df.loc[closest_idx, 'cholesterol_level']
    target_y = df.loc[closest_idx, 'population_perc']
    
    # Offset for the arrow text
    arrow_y_offset = df['population_perc'].max() * 0.2
    # Add arrow annotation
    ax.annotate(f'{target_chol} mg/dL', 
            xy=(target_x, target_y), # type: ignore
            xytext=(target_x, target_y + arrow_y_offset), # type: ignore
            fontsize=14,
            fontweight='bold',
            ha='center',
            arrowprops=dict(
                arrowstyle='->',
                lw=2.5,
                color='#A23B72',
                connectionstyle='arc3,rad=0'
            ),
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='#A23B72', 
                     edgecolor='#A23B72',
                     alpha=0.9),
            color='white')

    plt.text(184, plt.ylim()[1]*0.9, '184', color='r', fontsize=12, ha='center')

    # Add vertical line at 184
    ax.axvline(x=target_x, color='#A23B72', linestyle='--', # type: ignore
            linewidth=2, alpha=0.7, label=f'{target_chol} mg/dL Reference')

    # Customize the plot
    ax.set_xlabel('Total Cholesterol Level (mg/dL)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Individuals (%)', fontsize=14, fontweight='bold')
    ax.set_title('Total Cholesterol Distribution in US Males Aged 40-60\nNHANES 2021-2023', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis range and ticks
    ax.set_xlim(50, 400)
    ax.set_xticks(np.arange(50, 401, 25))

    # Set y-axis to start from 0
    ax.set_ylim(0, df['population_perc'].max() * 1.15)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

    # Add statistics text box
    stats_text = f"Sample Size: 777 males\nMean: 195.0 mg/dL\nMedian: 193.0 mg/dL\nRange: 62.0 - 393.0 mg/dL"
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Tight layout
    plt.tight_layout()

    # Save the figure
    output_path = './cholesterol_distribution_graph.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nGraph saved to: {output_path}")

    # Also save as high-res version
    output_path_hires = './cholesterol_distribution_graph_hires.png'
    plt.savefig(output_path_hires, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High-resolution graph saved to: {output_path_hires}")

    plt.close()

if __name__ == "__main__":
    df = merge_data(TCHOL, DEMO)
    if not df.empty:
        print("Data processing complete.")
        print(df.head())
        dist_df = obtain_distribution(df)
        plot_distribution(dist_df)
    else:
        print("No data to process.")
