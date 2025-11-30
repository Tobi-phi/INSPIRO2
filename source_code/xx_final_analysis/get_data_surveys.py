import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

### File to extract data from ASASSN and Gaia surveys

def get_data_assasn(star):
    df = pd.read_csv(star.asassn_data_path, sep=',')
    hjd_col = 'hjd' if 'hjd' in df.columns else 'HJD'
    camera_col = 'camera' if 'camera' in df.columns else 'Camera'

    df[hjd_col] = pd.to_numeric(df[hjd_col], errors='coerce')
    df['mag'] = pd.to_numeric(df['mag'], errors='coerce')
    df['mag_err'] = pd.to_numeric(df['mag_err'], errors='coerce')

    df = df.dropna(subset=[hjd_col, 'mag', 'mag_err', camera_col])
    df = df[df['mag_err'] <= 90]

    hjd = df[hjd_col].values.astype(float)
    mag = df['mag'].values.astype(float)
    mag_err = df['mag_err'].values.astype(float)
    camera = df[camera_col].values.astype(str) 

    return hjd_to_jdutc(hjd, star, obs_location=None), mag, mag_err, camera

def get_data_gaia(star):
    df = pd.read_csv(star.gaia_data_path, sep=',')

    def process_band(time_col, mag_col):
        if time_col not in df.columns or mag_col not in df.columns:
            return np.array([]), np.array([])

        tcb = pd.to_numeric(df[time_col], errors='coerce').to_numpy()
        mag = pd.to_numeric(df[mag_col], errors='coerce').to_numpy()

        mask = np.isfinite(tcb) & np.isfinite(mag)
        tcb = tcb[mask]
        mag = mag[mask]

        if len(tcb) == 0:
            return np.array([]), np.array([])
        
        return gaiatcb_to_jdutc(tcb), mag

    jd_g, mag_g = process_band('g_transit_time', 'g_transit_mag')
    jd_bp, mag_bp = process_band('bp_obs_time', 'bp_mag')
    jd_rp, mag_rp = process_band('rp_obs_time', 'rp_mag')


    return jd_g, mag_g, jd_bp, mag_bp, jd_rp, mag_rp

def gaiatcb_to_jdutc(gaiatcb_minus_ref):
    bjd = gaiatcb_minus_ref + 2455197.5
    t = Time(bjd, format='jd', scale='tcb')
    return t.utc.jd

def hjd_to_jdutc(hjd_arr, star, obs_location=None):
    hjd = np.asarray(hjd_arr)
    coord = SkyCoord(ra=star.ra*u.degree, dec=star.dec*u.degree)
    if obs_location is None:
        obs_location = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m) #Sets this loaction as the actual observation location is not provided
    t_hjd = Time(hjd, format='jd', scale='utc', location=obs_location)
    lt = t_hjd.light_travel_time(coord, kind='heliocentric') # heliocentric light-travel time

    t_jd = t_hjd - lt

    return t_jd.utc.jd