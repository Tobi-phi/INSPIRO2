import matplotlib.pyplot as plt
from pathlib import Path
import json

from get_extra_stuff import get_visual, output_path, jd_start_default

### Function to plot light curves for a star over the full time interval

def plot_light_curve_full_time_interval(date_arr, mag_arr, mag_err_arr, filters, star, jd_start=jd_start_default):
    visual = get_visual()
    plt.figure(figsize=(8,5))

    colors = [visual['filter_colors'][f] for f in filters]

    for date, mag, mag_err, filter, color in zip(date_arr, mag_arr, mag_err_arr, filters, colors):
        plt.errorbar(date['jd_relative'].value, mag, yerr=mag_err, fmt='o', markersize=4, capsize=2, label=filter, color=color)

    plt.gca().invert_yaxis()  # magnitude scale: smaller = brighter
    plt.xlabel(f'Julian Date - {jd_start}', fontsize=visual['axis_size'])
    plt.ylabel('Apparent magnitude', fontsize=visual['axis_size'])
    plt.title(f'Light Curves for {star.star} for different filters', fontsize=visual['title_size'])

    plt.grid(True)
    plt.legend(fontsize=visual['label_size'])

    output_path_specific_star = output_path / f'{star.star}'
    output_path_specific_star.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_specific_star / f'{star.star}_full_light_curve.png')
    plt.close()