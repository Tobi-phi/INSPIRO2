import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from get_extra_stuff import get_visual, output_path

### Function to plot light curves for a star by splitting the time interval - unused function

def plot_light_curve_split_time_interval(date_arr, mag_arr, mag_err_arr, filters, star):
    visual = get_visual()
    plt.figure(figsize=(10,5))
    
    n_filters = len(filters)

    gap_threshold = 1.0  # days
    
    # First, we need to find the sections based on the first filter (or you can choose any)
    # Flatten the JD array to numeric values (assume it's just a numpy array of floats)
    dates = date_arr[0]  # pick first filter as reference
    # Find indices where gap > gap_threshold
    gaps = np.where(np.diff(dates) > gap_threshold)[0]
    
    # Define sections
    section_start = 0
    sections = []
    for gap_idx in gaps:
        sections.append((section_start, gap_idx+1))  # end is inclusive
        section_start = gap_idx + 1
    sections.append((section_start, len(dates)))  # last section
    
    # Now plot each section with “continuous” x-axis per section
    x_offset = 0
    tick_positions = []
    tick_labels = []
    
    for start, end in sections:
        # width of section
        section_width = end - start
        for filt_idx in range(n_filters):
            section_dates = date_arr[filt_idx][start:end]
            section_mag = mag_arr[filt_idx][start:end]
            section_err = mag_err_arr[filt_idx][start:end]
            
            plt.errorbar(
                x_offset + np.arange(section_width),
                section_mag,
                yerr=section_err,
                fmt='o',
                markersize=4,
                capsize=2,
                label=filters[filt_idx] if start==0 else None  # label only once
            )
        # set ticks for x-axis (optional: show first JD of section)
        tick_positions.append(x_offset + section_width/2)
        tick_labels.append(f"{dates[start]:.1f}")
        x_offset += section_width + 1  # add 1 empty unit between sections
    
    plt.gca().invert_yaxis()  # magnitude scale
    plt.xlabel('Julian Date (sections, gaps > 1 day skipped)')
    plt.ylabel('Apparent magnitude')
    plt.title(f'Light Curves for {star.star} (split by gaps)')
    plt.xticks(tick_positions, tick_labels)
    plt.grid(True)
    plt.legend()
    
    output_path_specific_star = output_path / f'{star.star}'
    output_path_specific_star.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_specific_star / f'split_light_curve_star{star.id}.png')
    plt.close()