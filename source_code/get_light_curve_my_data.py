import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from astropy.time import Time

from get_star_info import get_star_info, get_all_star_info
from get_extra_stuff import data_path, output_path, jd_start_default
import plots

### This plots the light curve for my own GOChile data for given stars

# Function to rename data files
def rename_data():
    for file in data_path.iterdir():
        if file.is_file():
            if len(file.name) > 40:
                parts = file.name.split('_')
                star = parts[0]
                filt = parts[2]
                # ind  = parts[-1].split('.')[0]
                # if ind == 'report':
                #     ind = 0
                new_name = f'{star}_{filt}_report.txt'
                new_path = data_path / new_name
                file.rename(new_path)
                print(f'Renamed file for {star} in filter {filt}.')

# Function to read the specific formation of a text file and return a DataFrame
def read_txt_file(path):
    df = pd.read_csv(data_path / path, sep="\t", skiprows=10)
    df.columns = [col.strip("#") for col in df.columns]
    return df

# Function to transform Julian Date multiple other dates
def transform_date(jd, jd_start = jd_start_default):
    jd = Time(jd, format='jd')
    jd_start = Time(jd_start, format='jd')

    dates = {}
    dates['jd'] = jd
    dates['jd_relative'] = jd - jd_start

    utc = jd.to_datetime()
    dates['datetime'] = utc
    dates['yyyy'] = [t.year for t in utc]
    dates['mm-dd'] = [t.strftime('%m-%d') for t in utc]

    return dates

# Function that tels us if an array is
def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# Function to get and plot the light curve for a specific star
def get_light_curve(star_id):
    star = get_star_info(star_id)

    dfs = [read_txt_file(p) for p in star.paths]
    filters = star.filters

    # Get needed data for each filter
    date_arr, mag_arr, mag_err_arr = [], [], []
    for df in dfs:
        mag0 = df["MAG"].to_numpy()
        bad_indices = np.where(mag0 == '?')[0]

        date0 = df["DATE"].to_numpy()
        mag_err0 = df["MERR"].to_numpy()

        mag = np.array(np.delete(mag0, bad_indices), dtype=float)
        date = np.array(np.delete(date0, bad_indices), dtype=float)
        mag_err = np.array(np.delete(mag_err0, bad_indices), dtype=float)

        date = transform_date(date)

        date_arr.append(date)
        mag_arr.append(mag)
        mag_err_arr.append(mag_err)


    # PLOT DIFFERENT PLOTS

    # Full time interval plot
    plots.plot_light_curve_full_time_interval(date_arr, mag_arr, mag_err_arr, filters, star, jd_start=jd_start_default)

    # Partial time interval plot
    plots.plot_light_curve_partial_time_interval(date_arr, mag_arr, mag_err_arr, filters, star, jd_start=jd_start_default)

    # Split time interval plot
    # plots.plot_light_curve_split_time_interval(date_arr, mag_arr, mag_err_arr, filters, star)


i_arr = [4, 9, 10, 12, 13, 14]
for i in i_arr:
    get_light_curve(i)