import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.units import u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import sys

from get_data_surveys import get_data_assasn, get_data_gaia

sys.path.append(str(Path(__file__).resolve().parent.parent)) 
from classes import star4, star9, star10, star12, star13, star14

### This file performs Lomb-Scargle periodogram analysis and plots the phase folded light curves

# Define paths
data_dir = Path(__file__).parent.parent / 'data'
asassn_dir = data_dir / 'asassn_data'
gaia_dir = data_dir / 'gaia_data'
output_dir = Path(__file__).parent.parent / 'output'


def compute_lomb_scargle(t, mag, mag_err, star):
    def error_estimate(freq, power, best_frequency, best_idx):
        i = best_idx

        around_peak = 30
        start = max(0, i - around_peak)
        end   = min(len(freq), i + around_peak)
        freq_window = freq[start:end]
        power_window = power[start:end]
        
        half_power = power[i] / 2
        mask = power_window >= half_power
        f_peak = freq[i]
        f_low = np.min(freq_window[mask])
        f_high = np.max(freq_window[mask])
        freq_err = (f_high - f_low) / 2
        period_err = freq_err / (best_frequency ** 2)
        return freq_err, period_err
    
    def find_second_freq(freq, power, best_frequency):
        max_idx = np.argmax(power)
        main_f = freq[max_idx]

        mask = np.abs(freq - main_f) > 0.07 * main_f 
        filtered_indices = np.argsort(power[mask])[::-1]

        second_idx = np.where(mask)[0][filtered_indices[0]]
        second_frequency = freq[second_idx]

        return second_frequency, second_idx

    # Frequency calculation
    losc = LombScargle(t, mag, mag_err,
                       center_data=True,
                       fit_mean=True,
                       nterms=2)
    if star.id == 4:
        maximum_frequency = 0.02
    else:
        maximum_frequency = 1.0

    freq, power = losc.autopower(
        minimum_frequency=0.0001,
        maximum_frequency=maximum_frequency,
        samples_per_peak=50,
        nyquist_factor=5
    )
    best_idx = np.argmax(power)
    best_frequency = freq[best_idx]
    best_period = 1 / best_frequency
    best_freq_err, best_period_err = error_estimate(freq, power, best_frequency, best_idx)

    second_frequency, second_idx = find_second_freq(freq, power, best_frequency)
    second_period = 1 / second_frequency
    second_freq_err, second_period_err = error_estimate(freq, power, second_frequency, second_idx)
    # Minima calculation
    t_dense = np.linspace(np.min(t), np.max(t), 10000)
    model = losc.model(t_dense, best_frequency)
    min_indices = (np.diff(np.sign(np.diff(model))) < 0).nonzero()[0] + 1
    min_times = t_dense[min_indices]
    # print(f'Min indices: {min_indices}')
    # print(f"Min times: {min_times}")

    second_model = losc.model(t_dense, second_frequency)
    second_min_indices = (np.diff(np.sign(np.diff(second_model))) < 0).nonzero()[0] + 1
    second_min_times = t_dense[second_min_indices]
    # print(f'Second Min indices: {second_min_indices}')
    # print(f"Second Min times: {second_min_times}")

    return freq, power, [best_frequency, best_freq_err], [best_period, best_period_err], [second_frequency, second_freq_err], [second_period, second_period_err], min_indices, min_times, second_min_indices, second_min_times

def plot_lomb_scargle(star):
    jd_asassn, mag_asassn, mag_err_asassn, camera_asassn = get_data_assasn(star)
    jd_g_gaia, mag_g_gaia, jd_bp_gaia, mag_bp_gaia, jd_rp_gaia, mag_rp_gaia = get_data_gaia(star)
    freq_asassn, power_asassn, best_frequency_asassn, best_period_asassn, second_frequency_asassn, second_period_asassn, min_indices_asassn, min_times_asassn, second_min_indices_asassn, second_min_times_asassn = compute_lomb_scargle(jd_asassn, mag_asassn, mag_err_asassn, star)
    freq_g_gaia, power_g_gaia, best_frequency_g_gaia, best_period_g_gaia, second_frequency_g_gaia, second_period_g_gaia, min_indices_g_gaia, min_times_g_gaia, second_min_indices_g_gaia, second_min_times_g_gaia = compute_lomb_scargle(jd_g_gaia, mag_g_gaia, 0.01*np.ones_like(mag_g_gaia), star)
    freq_bp_gaia, power_bp_gaia, best_frequency_bp_gaia, best_period_bp_gaia, second_frequency_bp_gaia, second_period_bp_gaia, min_indices_bp_gaia, min_times_bp_gaia, second_min_indices_bp_gaia, second_min_times_bp_gaia = compute_lomb_scargle(jd_bp_gaia, mag_bp_gaia, 0.01*np.ones_like(mag_bp_gaia), star)
    freq_rp_gaia, power_rp_gaia, best_frequency_rp_gaia, best_period_rp_gaia, second_frequency_rp_gaia, second_period_rp_gaia, min_indices_rp_gaia, min_times_rp_gaia, second_min_indices_rp_gaia, second_min_times_rp_gaia = compute_lomb_scargle(jd_rp_gaia, mag_rp_gaia, 0.01*np.ones_like(mag_rp_gaia), star)
    plt.figure(figsize=(8, 6))
    if star.id == 4:
        plt.plot(freq_asassn, power_asassn, label='ASAS-SN', color='orange')
    plt.plot(freq_g_gaia, power_g_gaia, label='Gaia G', color='green')
    plt.plot(freq_bp_gaia, power_bp_gaia, label='Gaia BP', color='blue')
    plt.plot(freq_rp_gaia, power_rp_gaia, label='Gaia RP', color='red')
    plt.xlabel('Frequency [1/day]', fontsize=14)
    plt.ylabel('Lomb-Scargle Power', fontsize=14)
    plt.title('Lomb-Scargle Periodogram', fontsize=16)
    plt.grid()

    best_color = 'black'
    second_color = 'magenta'

    if star.id==4:
        plt.xlim(0, 0.02)
        plt.text(0.95, 0.95, f"Best frequency:\n$\\nu_1$ = ({best_frequency_asassn[0]:.4f}±{best_frequency_asassn[1]:.4f}) 1/day\n= ({best_frequency_asassn[0]*365.25:.2f}±{best_frequency_asassn[1]*365.25:.2f}) 1/year\nBest period: $P_1$ = ({best_period_asassn[0]:.1f}±{best_period_asassn[1]:.1f}) days",
             transform=plt.gca().transAxes,
             va='top', ha='right',
             color=best_color,
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=best_color, alpha=0.5))
        plt.text(0.95, 0.7, f"Second frequency:\n$\\nu_2$ = ({second_frequency_asassn[0]:.4f}±{second_frequency_asassn[1]:.4f}) 1/day\n= ({second_frequency_asassn[0]*365.25:.2f}±{second_frequency_asassn[1]*365.25:.2f}) 1/year\nSecond period: $P_2$ = ({second_period_asassn[0]:.1f}±{second_period_asassn[1]:.1f}) days",
             transform=plt.gca().transAxes,
             va='top', ha='right',
             color=second_color,
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=second_color, alpha=0.5))
    
        plt.axvline(x=best_frequency_asassn[0], color=best_color, linestyle='--', label='Best Frequency')
        plt.axvline(x=second_frequency_asassn[0], color=second_color, linestyle='--', label='Second Frequency')
    else:
        plt.xlim(0, 1)
        plt.text(0.95, 0.95, f"Best frequency:\n$\\nu_1$ = ({best_frequency_g_gaia[0]:.4f}±{best_frequency_g_gaia[1]:.4f}) 1/day\n= ({best_frequency_g_gaia[0]*365.25:.2f}±{best_frequency_g_gaia[1]*365.25:.2f}) 1/year\nBest period: $P_1$ = ({best_period_g_gaia[0]:.1f}±{best_period_g_gaia[1]:.1f}) days",
             transform=plt.gca().transAxes,
             va='top', ha='right',
             color=best_color,
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=best_color, alpha=0.5))
        plt.axvline(x=best_frequency_g_gaia[0], color=best_color, linestyle='--', label='Best Frequency')

    
    # plt.axvline(x=best_frequency_asassn[0]/2, color='purple', linestyle='--', label='Best Frequency / 2')

    plt.legend(loc='lower right', fontsize=12)
    plt.savefig(output_dir / f'star{star.id}' / f'lomb_scargle_asassn_star_{star.id}.png')
    plt.close()

    print(f"Lomb-Scargle results for Star {star.id} ({star.star}):")
    print('------')
    print("From ASAS-SN Data:")
    print(f"Best frequency: {best_frequency_asassn} 1/day = {np.array(best_frequency_asassn)*365.25} 1/year")
    print(f"Best period: {best_period_asassn} days")
    print(f"log10(Best period): {np.log10(best_period_asassn)}") 

    print(f"Second frequency: {second_frequency_asassn} 1/day = {np.array(second_frequency_asassn)*365.25} 1/year")
    print(f"Second period: {second_period_asassn} days")
    log10_second_period = np.log10(second_period_asassn[0])
    log10_second_period_plus = np.log10(second_period_asassn[0] + second_period_asassn[1]) - log10_second_period
    log10_second_period_minus = np.log10(second_period_asassn[0] - second_period_asassn[1]) - log10_second_period
    print(f"log10(Second period): {np.log10(second_period_asassn[0])} (+{log10_second_period_plus}, -{log10_second_period_minus})")
    print('------')
    print("From Gaia G Data:")
    print(f"Best frequency: {best_frequency_g_gaia} 1/day = {np.array(best_frequency_g_gaia)*365.25} 1/year")
    print(f"Best period: {best_period_g_gaia} days")
    print(f"log10(Best period): {np.log10(best_period_g_gaia)}")
    print('------')
    print("From Gaia BP Data:")
    print(f"Best frequency: {best_frequency_bp_gaia} 1/day = {np.array(best_frequency_bp_gaia)*365.25} 1/year")
    print(f"Best period: {best_period_bp_gaia} days")
    print(f"log10(Best period): {np.log10(best_period_bp_gaia)}")
    print('------')
    print("From Gaia RP Data:")
    print(f"Best frequency: {best_frequency_rp_gaia} 1/day = {np.array(best_frequency_rp_gaia)*365.25} 1/year")
    print(f"Best period: {best_period_rp_gaia} days")
    print(f"log10(Best period): {np.log10(best_period_rp_gaia)}")
    print('------')

    print("----------------------------------------------------------")


def plot_data_in_one_period(star):
    jd, mag, mag_err, camera = get_data_assasn(star)
    freq, power, best_frequency, best_period, second_frequency, second_period, min_indices, min_times, second_min_indices, second_min_times = compute_lomb_scargle(jd, mag, mag_err, star)

    print(jd.shape, mag.shape, mag_err.shape)
    print(min_indices[0])
    print(second_min_indices[0])

    phases = ((jd - jd[min_indices[0]]) / best_period[0]) % 1

    phases2 = ((jd - jd[second_min_indices[0]]) / second_period[0]) % 1

    avg_mag = np.average(mag)

    max_mag = np.max(mag)
    min_mag = np.min(mag)
    central_mag = (max_mag + min_mag) / 2
    delta_mag = mag - central_mag


    # Plot with apparent magnitude
    plt.figure(figsize=(8, 6))
    plt.errorbar(phases, mag, yerr=mag_err, fmt='o', markersize=4, alpha=0.5, label=f'Phase with One Period: ({best_period[0]:.4f} +/- {best_period[1]:.4f}) days')
    plt.errorbar(phases2, mag, yerr=mag_err, fmt='o', markersize=4, alpha=0.5, label=f'Phase with Two Periods: ({second_period[0]:.4f} +/- {second_period[1]:.4f}) days')

    plt.axhline(y=avg_mag, color='green', linestyle='--', label=f'Average Magnitude={avg_mag:.2f}')
    plt.axhline(y=central_mag, color='red', linestyle='--', label=f'Central Magnitude={central_mag:.2f}')

    plt.gca().invert_yaxis()
    plt.title(f'Light Curve for Star {star.id} folded in One Period')
    plt.xlabel('Phase')
    plt.ylabel('Apparent Magnitude')
    plt.grid()
    plt.legend()
    plt.savefig(output_dir / f'star{star.id}' / f'light_curve_phase_asassn_star_{star.id}.png')
    plt.close()

    #plot with delta magnitude
    plt.figure(figsize=(8, 6))
    plt.errorbar(phases, delta_mag, yerr=mag_err, fmt='o', markersize=4, alpha=0.5, label=f'Phase with One Period: ({best_period[0]:.4f} +/- {best_period[1]:.4f}) days')
    # plt.errorbar(phases2, delta_mag, yerr=mag_err, fmt='o', markersize=4, alpha=0.5, label=f'Phase with Two Periods: ({second_period[0]:.4f} +/- {second_period[1]:.4f}) days')

    plt.axhline(y=avg_mag - central_mag, color='green', linestyle='--', label=f'Average Magnitude={avg_mag:.2f}')
    plt.axhline(y=0, color='red', linestyle='--', label=f'Central Magnitude={central_mag:.2f}')

    plt.gca().invert_yaxis()
    plt.title(f'Light Curve for Star {star.id} folded in One Period')
    plt.xlabel('Phase')
    plt.ylabel(r'$\Delta$ Magnitude')
    plt.grid()
    plt.legend()
    plt.savefig(output_dir / f'star{star.id}' / f'light_curve_phase_asassn_star_{star.id}_Delta_mag.png')
    plt.close()

def plot_data_in_one_period_all_data(star):
    jd_asassn, mag_asassn, mag_err_asassn, camera_asassn = get_data_assasn(star)
    jd_g_gaia, mag_g_gaia, jd_bp_gaia, mag_bp_gaia, jd_rp_gaia, mag_rp_gaia = get_data_gaia(star)

    if star.id == 4:      
        freq, power, best_frequency, best_period, second_frequency, second_period, min_indices, min_times, second_min_indices, second_min_times = compute_lomb_scargle(jd_asassn, mag_asassn, mag_err_asassn, star)
    else:
        freq, power, best_frequency, best_period, second_frequency, second_period, min_indices, min_times, second_min_indices, second_min_times = compute_lomb_scargle(jd_g_gaia, mag_g_gaia, 0.01*np.ones_like(mag_g_gaia), star)

    print(jd_asassn.shape, mag_asassn.shape, mag_err_asassn.shape)
    print(min_indices[0])
    print(second_min_indices[0])

    phases_asassn = ((jd_asassn - jd_asassn[min_indices[0]]) / best_period[0]) % 1
    phases2_asassn = ((jd_asassn - jd_asassn[second_min_indices[0]]) / second_period[0]) % 1

    phases_g_gaia = ((jd_g_gaia - jd_asassn[min_indices[0]]) / best_period[0]) % 1
    phases_bp_gaia = ((jd_bp_gaia - jd_asassn[min_indices[0]]) / best_period[0]) % 1
    phases_rp_gaia = ((jd_rp_gaia - jd_asassn[min_indices[0]]) / best_period[0]) % 1

    phases2_g_gaia = ((jd_g_gaia - jd_asassn[second_min_indices[0]]) / second_period[0]) % 1
    phases2_bp_gaia = ((jd_bp_gaia - jd_asassn[second_min_indices[0]]) / second_period[0]) % 1
    phases2_rp_gaia = ((jd_rp_gaia - jd_asassn[second_min_indices[0]]) / second_period[0]) % 1

    if star.my_data_path is not None:
        mag_my_data_arr, jd_my_data_arr, mag_err_my_data_arr, filters_my_data_arr, phases_my_data_arr, phases2_my_data_arr = np.array([]), np.array([]), np.array([]), np.array([], dtype=str), np.array([]), np.array([])
        for path in star.my_data_path:
            filter = 'G' if 'G_report' in path.name else 'R'

            df = pd.read_csv(data_dir / path, sep="\t", skiprows=10)
            df.columns = [col.strip("#") for col in df.columns]
            mag0 = df["MAG"].to_numpy()
            bad_indices = np.where(mag0 == '?')[0]

            jd0 = df["DATE"].to_numpy()
            mag_err0 = df["MERR"].to_numpy()

            mag_clean = np.delete(mag0, bad_indices).astype(float)
            jd_clean = np.delete(jd0, bad_indices).astype(float)
            mag_err_clean = np.delete(mag_err0, bad_indices).astype(float)

            phases_my_data = ((jd_clean - jd_asassn[min_indices[0]]) / best_period[0]) % 1
            phases2_my_data = ((jd_clean - jd_asassn[second_min_indices[0]]) / second_period[0]) % 1

            mag_my_data_arr = np.append(mag_my_data_arr, mag_clean)
            jd_my_data_arr = np.append(jd_my_data_arr, jd_clean)
            mag_err_my_data_arr = np.append(mag_err_my_data_arr, mag_err_clean)
            filters_my_data_arr = np.append(filters_my_data_arr, filter)
            phases_my_data_arr = np.append(phases_my_data_arr, phases_my_data)
            phases2_my_data_arr = np.append(phases2_my_data_arr, phases2_my_data)

            print('-------------------------------------')
            print('Dataset', path.name)
            print('Shape of jd_my_data_arr:', jd_my_data_arr.shape, 'Length', len(jd_my_data_arr))
            print('shape of mag_my_data_arr:', mag_my_data_arr.shape, 'Length', len(mag_my_data_arr))
            print('Shape of mag_err_my_data_arr:', mag_err_my_data_arr.shape, 'Length', len(mag_err_my_data_arr))
            print('Shape of phases_my_data_arr:', phases_my_data_arr.shape, 'Length', len(phases_my_data_arr))
            print('Shape of phases2_my_data_arr:', phases2_my_data_arr.shape, 'Length', len(phases2_my_data_arr))
            print('Shape of filters:', filters_my_data_arr.shape, 'Length', len(filters_my_data_arr))
            print('-------------------------------------')

    avg_mag = np.average(mag_asassn)

    max_mag = np.max(mag_asassn)
    min_mag = np.min(mag_asassn)
    central_mag = (max_mag + min_mag) / 2
    delta_mag = mag_asassn - central_mag

    print(f'Average Magnitude ASAS-SN: {avg_mag}')
    print(f'Central Magnitude ASAS-SN: {central_mag}')
    print(f'Amplitude ASAS-SN: {max_mag - min_mag}')

    # Plot with apparent magnitude
    plt.figure(figsize=(8, 6))
    # plt.errorbar(phases_asassn, mag_asassn, yerr=mag_err_asassn, fmt='o', markersize=4, alpha=0.5, label=f'ASAS-SN Phase with One Period: ({best_period[0]:.4f} +/- {best_period[1]:.4f}) days')
    if star.id == 4:
        plt.errorbar(phases2_asassn, mag_asassn, yerr=mag_err_asassn, fmt='o', markersize=4, alpha=0.5, color='darkorange', label=f'ASAS-SN V Band')
    else:
        pass
        # plt.errorbar(phases_asassn, mag_asassn, yerr=mag_err_asassn, fmt='o', markersize=4, alpha=0.5, color='darkorange', label=f'ASAS-SN V Band')
    # plt.errorbar(phases_g_gaia, mag_g_gaia, fmt='o', markersize=4, alpha=0.5, label='Gaia G Band Phase with One Period')
    # plt.errorbar(phases_bp_gaia, mag_bp_gaia, fmt='o', markersize=4, alpha=0.5, label='Gaia BP Band Phase with One Period')    
    # plt.errorbar(phases_rp_gaia, mag_rp_gaia, fmt='o', markersize=4, alpha=0.5, label='Gaia RP Band Phase with One Period')

    gaia_g_mean = np.mean(mag_g_gaia)
    gaia_bp_mean = np.mean(mag_bp_gaia)
    gaia_rp_mean = np.mean(mag_rp_gaia)
    print(f'Gaia G mean: {gaia_g_mean}, ASAS-SN mean: {np.mean(mag_asassn)}')
    print(f'Gaia BP mean: {gaia_bp_mean}, ASAS-SN mean: {np.mean(mag_asassn)}')
    print(f'Gaia RP mean: {gaia_rp_mean}, ASAS-SN mean: {np.mean(mag_asassn)}')

    color_gaia_bp_rp = gaia_bp_mean - gaia_rp_mean
    print(f'Color Gaia BP - RP: {color_gaia_bp_rp}')

    gaia_g_adjustment =  np.mean(mag_asassn) - np.mean(mag_g_gaia)
    gaia_bp_adjustment = np.mean(mag_bp_gaia) - np.mean(mag_asassn)
    gaia_rp_adjustment = np.mean(mag_asassn) - np.mean(mag_rp_gaia)
    # gaia_g_adjustment, gaia_bp_adjustment, gaia_rp_adjustment = 0,0,0

    if star.id == 4:
        plt.errorbar(phases2_g_gaia, mag_g_gaia + gaia_g_adjustment, fmt='o', markersize=4, alpha=0.5, color='green', label=f'Gaia G Band + {gaia_g_adjustment:.2f}')
        plt.errorbar(phases2_bp_gaia, mag_bp_gaia - gaia_bp_adjustment, fmt='o', markersize=4, alpha=0.5, color='blue', label=f'Gaia BP Band - {gaia_bp_adjustment:.2f}')
        plt.errorbar(phases2_rp_gaia, mag_rp_gaia + gaia_rp_adjustment, fmt='o', markersize=4, alpha=0.5, color='red', label=f'Gaia RP Band + {gaia_rp_adjustment:.2f}')
    else:
        plt.errorbar(phases_g_gaia, mag_g_gaia + gaia_g_adjustment, fmt='o', markersize=4, alpha=0.5, color='green', label=f'Gaia G Band + {gaia_g_adjustment:.2f}')
        plt.errorbar(phases_bp_gaia, mag_bp_gaia - gaia_bp_adjustment, fmt='o', markersize=4, alpha=0.5, color='blue', label=f'Gaia BP Band - {gaia_bp_adjustment:.2f}')
        plt.errorbar(phases_rp_gaia, mag_rp_gaia + gaia_rp_adjustment, fmt='o', markersize=4, alpha=0.5, color='red', label=f'Gaia RP Band + {gaia_rp_adjustment:.2f}')

    if star.my_data_path is not None:
        if len(star.my_data_path) == 1:
            filter = filters_my_data_arr[0::][0]
            color = 'lime' if filter == 'G' else 'brown'
            plt.errorbar(phases2_my_data_arr, mag_my_data_arr, yerr=mag_err_my_data_arr, fmt='o', markersize=4, alpha=0.5, color=color, label=f'My GoChile Data: {filter} Band')
        else:
            for i in range(len(star.my_data_path)):
                filter = filters_my_data_arr[i::][0]
                color = 'lime' if filter == 'G' else 'brown'
                print('----------------------------------------------------------')
                print(len(star.my_data_path))
                print(phases2_my_data_arr.shape, mag_my_data_arr.shape, mag_err_my_data_arr.shape)
                print(len(phases2_my_data_arr[i::]), len(mag_my_data_arr[i::]), len(mag_err_my_data_arr[i::]))
                plt.errorbar(phases2_my_data_arr[i::], mag_my_data_arr[i::], yerr=mag_err_my_data_arr[i::], fmt='o', markersize=4, alpha=0.5, color=color, label=f'My GoChile Data: {filter} Band')
            


    plt.axhline(y=avg_mag, color='deeppink', linestyle='--', label=f'$m_{{avg,ASAS-SN}}$={avg_mag:.2f}')
    plt.axhline(y=central_mag, color='black', linestyle='--', label=f'$m_{{central,ASAS-SN}}$={central_mag:.2f}')

    plt.gca().invert_yaxis()
    if star.id == 4:
        plt.ylim(16.15, 14.5)
        which_period = 'Second'
        period = second_period
    if star.id == 9:
        plt.ylim(9.54, 9.49)
        which_period = 'Best'
        period = best_period
    if star.id == 13:
        plt.ylim(10.7, 10.5)
        which_period = 'Best'
        period = best_period
    plt.title(f'Phase-folded Light Curve for Star {star.id} using {which_period} Period:\n$P_2$ = ({period[0]:.1f} +/- {period[1]:.1f}) days', fontsize=16)
    plt.xlabel('Phase', fontsize=14)
    plt.ylabel('Apparent Magnitude $m$', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12, ncol=2)
    plt.savefig(output_dir / f'star{star.id}' / f'light_curve_phase_all_data_star_{star.id}.png')
    plt.close()




ids=[star4,star9,star12,star13,star14]
ids=[star4, star9]
ids=[star4]


def plot_all_lomb_scargle_asassn(ids):
    for star in ids:
        plot_lomb_scargle(star)

def plot_all_data_in_one_period(ids):
    for star in ids:
        plot_data_in_one_period(star)

def plot_all_data_in_one_period_all_data(ids):
    for star in ids:
        plot_data_in_one_period_all_data(star)

# plot_all_lomb_scargle_asassn([star4])
# plot_all_data_in_one_period([star4])
plot_all_data_in_one_period_all_data([star4])

# plot_all_lomb_scargle_asassn([star9, star12, star13, star14])

# plot_all_lomb_scargle_asassn([star9, star13])
# plot_all_data_in_one_period([star9, star13])
# plot_all_data_in_one_period_all_data([star9, star13])