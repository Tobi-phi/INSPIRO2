import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from get_data_surveys import get_data_assasn, get_data_gaia

# Define paths
data_dir = Path(__file__).parent.parent / 'data'
asassn_dir = data_dir / 'asassn_data'
gaia_dir = data_dir / 'gaia_data'
output_dir = Path(__file__).parent.parent / 'output'

sys.path.append(str(Path(__file__).resolve().parent.parent)) 
from classes import star4, star9, star10, star12, star13, star14
from get_star_info import get_star_info


def plot_light_curve_asassn(star):
    jd, mag, mag_err, camera = get_data_assasn(star)
    unique_cameras = np.unique(camera)
    colors = plt.cm.tab10.colors 

    plt.figure(figsize=(8, 6))
    jd_start = 2355000
    for i, cam in enumerate(unique_cameras):
        mask = camera == cam
        plt.errorbar(
            jd[mask] - jd_start, mag[mask], yerr=mag_err[mask],
            fmt='o', markersize=4,
            ecolor='gray', capsize=4,
            color=colors[i % len(colors)],
            label=f'Camera {cam}'
        )

    from lomb_scargle_periodogram import compute_lomb_scargle
    _, _, _, _, _, _, _, min_times, _, second_min_times = compute_lomb_scargle(jd, mag, mag_err, star)
    for i, t in enumerate(min_times):
        plt.axvline(x=t - jd_start, color='red', linestyle='--', alpha=0.7,label='Minima' if i == 0 else None)
    for i, t2 in enumerate(second_min_times):
        plt.axvline(x=t2 - jd_start, color='blue', linestyle='--', alpha=0.7,label='Second Minima' if i == 0 else None)

    plt.gca().invert_yaxis()
    plt.xlabel(f'JD - {jd_start}')
    plt.ylabel('Apparent Magnitude')
    plt.title(f'Light Curve for Star {star.id} (ASAS-SN)')
    plt.grid()
    plt.legend()
    plt.savefig(output_dir / f'star{star.id}' / f'light_curve_asassn_star_{star.id}.png')
    plt.close()

def plot_light_curve_gaia(star):
    jd_g, mag_g, jd_bp, mag_bp, jd_rp, mag_rp = get_data_gaia(star)
    plt.figure(figsize=(8, 6))
    plt.scatter(jd_g, mag_g, s=10, color='blue', label='G Band')
    plt.scatter(jd_bp, mag_bp, s=10, color='green', label='BP Band')
    plt.scatter(jd_rp, mag_rp, s=10, color='red', label='RP Band')

    plt.gca().invert_yaxis()
    plt.xlabel('JD')
    plt.ylabel('Apparent Magnitude')
    plt.title(f'Light Curve for Star {star.id} (Gaia)')
    plt.grid()
    plt.legend()
    plt.savefig(output_dir / f'star{star.id}' / f'light_curve_gaia_star_{star.id}.png')
    plt.close()

def plot_light_curve_for_all_data_sources(star):
    jd_start = 0
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    fig, ax1 = plt.subplots(figsize=(8, 6))

    if star.asassn_data_path is not None and star.id != 13:
        jd_asassn, mag_asassn, mag_err_asassn, camera_asassn = get_data_assasn(star)

        unique_cameras = np.unique(camera_asassn)
        colors = plt.cm.tab10.colors 
        for i, cam in enumerate(unique_cameras):
            mask = camera_asassn == cam
            ax1.errorbar(
                jd_asassn[mask] - jd_start, mag_asassn[mask], yerr=mag_err_asassn[mask],
                fmt='o', markersize=4,
                ecolor='gray', capsize=4,
                color=colors[i % len(colors)],
                label=f'ASAS-SN V band;\nCamera {cam}'
            )

    if star.gaia_data_path is not None:
        jd_g_gaia, mag_g_gaia, jd_bp_gaia, mag_bp_gaia, jd_rp_gaia, mag_rp_gaia = get_data_gaia(star)
        
        ax1.scatter(jd_g_gaia - jd_start, mag_g_gaia, s=20, color='green', label='Gaia G Band')
        ax1.scatter(jd_bp_gaia - jd_start, mag_bp_gaia, s=20, color='blue', label='Gaia BP Band')
        ax1.scatter(jd_rp_gaia - jd_start, mag_rp_gaia, s=20, color='red', label='Gaia RP Band')

    # if star.my_data_path is not None:
    #     for path in star.my_data_path:
    #         filter = 'G' if 'G_report' in path.name else 'R'

    #         df = pd.read_csv(data_dir / path, sep="\t", skiprows=10)
    #         df.columns = [col.strip("#") for col in df.columns]
    #         mag0 = df["MAG"].to_numpy()
    #         bad_indices = np.where(mag0 == '?')[0]

    #         jd0 = df["DATE"].to_numpy()
    #         mag_err0 = df["MERR"].to_numpy()

    #         mag_my_data = np.array(np.delete(mag0, bad_indices), dtype=float)
    #         jd_my_data = np.array(np.delete(jd0, bad_indices), dtype=float)
    #         mag_err_my_data = np.array(np.delete(mag_err0, bad_indices), dtype=float)

    #         ax2.errorbar(
    #             jd_my_data - jd_start, mag_my_data, yerr=mag_err_my_data,
    #             fmt='o', markersize=4,
    #             ecolor='gray', capsize=4,
    #             color='limegreen' if filter == 'G' else 'orange',
    #             label=f'My Data: {filter} Band'
    #         )
    # ax2.set_xlim(jd_my_data.min()-0.1, jd_my_data.max()+0.1)
    # ax2.grid()
    # ax2.set_xlabel('JD')
    
    if star.id == 4:
        ax1.set_xlim(jd_asassn.min()-20, jd_asassn.max()+20)
    else:
        ax1.set_xlim(jd_g_gaia.min()-20, jd_g_gaia.max()+20)
    ax1.grid()
    ax1.set_xlabel('JD', fontsize=14)
    ax1.set_ylabel('Apparent Magnitude', fontsize=14)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.1)
    fig.suptitle(f'Light Curve for Star {star.id} (All Data Sources)', y=0.95, fontsize=16)
    fig.legend(loc='upper right',bbox_to_anchor=(0.9, 0.88), fontsize=12)
    plt.savefig(output_dir / f'star{star.id}' / f'light_curve_all_sources_star_{star.id}.png')
    plt.close()



    


def plot_all_light_curves_gaia(ids):
    for star in ids:
        plot_light_curve_gaia(star)

def plot_all_light_curves_asassn(ids):
    for star in ids:
        plot_light_curve_asassn(star)

def plot_all_light_curves_all_data_sources(ids):
    for star in ids:
        plot_light_curve_for_all_data_sources(star)

# plot_all_light_curves_asassn([star4])
# plot_all_light_curves_all_data_sources([star4])

# plot_all_light_curves_asassn([star13])
# plot_all_light_curves_all_data_sources([star13])