import matplotlib.pyplot as plt
from pathlib import Path
import json
from brokenaxes import brokenaxes

from get_extra_stuff import get_visual, output_path, jd_start_default

### Function to plot light curves for a star over partial time intervals

splits = {
    1: [(4.525, 4.86), (6.65, 6.875)],
    2: [(), ()],
    3: [(), ()],
    4: [(), ()], #yes
    5: [(), ()],
    6: [(), ()],
    7: [(), ()],
    8: [(), ()],
    9: [(), ()], #yes
    10: [(12.5, 13.1), (40.6, 41), (41.5, 41.8), (42.5, 43), (43.5, 43.75)], #yes
    11: [(), ()],
    12: [(13.575, 13.625), (14.655, 14.9)], #yes
    13: [(16.75, 16.95)], #yes
    14: [(40.5, 40.65), (41.5, 41.7), (43.5, 43.65)] #yes
}


def plot_light_curve_partial_time_interval(date_arr, mag_arr, mag_err_arr, filters, star, jd_start=jd_start_default):
    visual = get_visual()
    colors = [visual['filter_colors'][f] for f in filters]

    intervals = [s for s in splits[star.id] if s]
    if not intervals:
        print(f"No split intervals defined for star {star.star} (ID {star.id}). Plotting full curve.")
        intervals = [(date_arr[0]['jd_relative'].value.min(), date_arr[0]['jd_relative'].value.max())]

    bax = brokenaxes(
        xlims=intervals,
        hspace=0.05,
        despine=False,
        fig=plt.figure(figsize=(9, 5))
    )

    for date, mag, mag_err, filter, color in zip(date_arr, mag_arr, mag_err_arr, filters, colors):
        bax.errorbar(date['jd_relative'].value, mag, yerr=mag_err, fmt='o', markersize=4, capsize=2, label=filter, color=color)

    bax.invert_yaxis()  # magnitude scale: smaller = brighter
    bax.set_xlabel(f'Julian Date - {jd_start}', fontsize=visual['axis_size'], labelpad=20)
    bax.set_ylabel('Apparent magnitude', fontsize=visual['axis_size'])
    bax.set_title(f'Light Curves for {star.star} for different filters', fontsize=visual['title_size'], pad=15)

    bax.grid(True)
    bax.legend(fontsize=visual['label_size'], loc='lower right')


    output_path_specific_star = output_path / f'{star.star}'
    output_path_specific_star.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_specific_star / f'{star.star}_partial_light_curve.png')
    plt.close()