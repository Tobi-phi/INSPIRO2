from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path

data_dir = Path(__file__).parent / 'data'

# Define a class to hold variable star information
class VariableStar:
    def __init__(self, star, id, name, gaia_name, variable, variable_type, ra, dec, filters, paths):
        self.star = star
        self.id = id
        self.name = name
        self.gaia_name = gaia_name
        self.variable = variable
        self.variable_type = variable_type
        self.ra = ra
        self.dec = dec
        self.filters = filters
        self.paths = paths

class VariableStar2:
    def __init__(self, star, id, name, gaia_name, variable, variable_type, ra_hours, dec_degrees, ra, dec, my_data_path, gaia_data_path, asassn_data_path, geo_dist=None, photo_geo_dist=None, dist=None, area_dist=None, color='black'):
        self.star = star
        self.id = id
        self.name = name
        self.gaia_name = gaia_name
        self.variable = variable
        self.variable_type = variable_type
        self.ra_hours = ra_hours
        self.ra = ra
        self.dec_degrees = dec_degrees
        self.dec = dec
        self.my_data_path = my_data_path
        self.gaia_data_path = gaia_data_path
        self.asassn_data_path = asassn_data_path
        self.geo_dist = geo_dist
        self.photo_geo_dist = photo_geo_dist
        self.dist = dist
        self.area_dist = area_dist
        self.color = color

star4 = VariableStar2(
    star='Star 4',
    id=4,
    name='Gaia DR3 4685832478414732800',
    gaia_name='Gaia DR3 4685832478414732800',
    variable=True,
    variable_type='CEP',
    ra_hours='00 47 01.37',
    dec_degrees='-73 26 46.2',
    ra=SkyCoord('00 47 01.37 -73 26 46.2', unit=(u.hourangle, u.deg)).ra.deg,
    dec=SkyCoord('00 47 01.37 -73 26 46.2', unit=(u.hourangle, u.deg)).dec.deg,
    my_data_path= [data_dir / 'star4_G_report.txt'],
    gaia_data_path=data_dir / 'gaia_data' / 'star4' / 'EPOCH_PHOTOMETRY-Gaia DR3 4685832478414732800.csv',
    asassn_data_path=data_dir / 'asassn_data' / 'star4' / 'AP36404381.csv',
    geo_dist=[20063.22460000, 14593.96390000, 28909.63090000],
    photo_geo_dist=[15286.76460000, 12803.49320000, 17402.03320000],
    dist=4258.1055,
    area_dist=[62440, 62440 - 470, 62440 + 470],
    color='magenta'
)

star9 = VariableStar2(
    star='Star 9',
    id=9,
    name='Gaia DR3 6098713338227827584',
    gaia_name='Gaia DR3 6098713338227827584',
    variable=True,
    variable_type='BCEP',
    ra_hours='14 45 23.44',
    dec_degrees='-44 21 20.5',
    ra=SkyCoord('14 45 23.44 -44 21 20.5', unit=(u.hourangle, u.deg)).ra.deg,
    dec=SkyCoord('14 45 23.44 -44 21 20.5', unit=(u.hourangle, u.deg)).dec.deg,
    my_data_path= [data_dir / 'star9_G_report.txt', data_dir / 'star9_R_report.txt'],
    gaia_data_path=data_dir / 'gaia_data' / 'star9' / 'EPOCH_PHOTOMETRY-Gaia DR3 6098713338227827584.csv',
    asassn_data_path=data_dir / 'asassn_data' / 'star9' / 'AP54786673.csv',
    geo_dist=[1447.29126000, 1391.75281000, 1523.96399000],
    photo_geo_dist=[1450.33362000, 1388.65784000, 1513.71118000]
)

star10 = VariableStar2(
    star='Star 10',
    id=10,
    name='LMC V2588',
    gaia_name='Gaia DR3 4660334872473302400',
    variable=True,
    variable_type='CEP',
    ra_hours='05 26 33.78',
    dec_degrees='-66 23 12.8',
    ra=SkyCoord('05 26 33.78 -66 23 12.8', unit=(u.hourangle, u.deg)).ra.deg,
    dec=SkyCoord('05 26 33.78 -66 23 12.8', unit=(u.hourangle, u.deg)).dec.deg,
    my_data_path= [data_dir / 'star10_G_report.txt', data_dir / 'star10_R_report.txt'],
    gaia_data_path= None,
    asassn_data_path=data_dir / 'asassn_data' / 'star10' / 'light_curve_9d4aa9a3-976d-49a7-9d92-46ecb799b5e7.csv',
)

star12 = VariableStar2(
    star='Star 12',
    id=12,
    name='Gaia DR3 6870452273367422080',
    gaia_name='Gaia DR3 6870452273367422080',
    variable=True,
    variable_type='BCEP',
    ra_hours='19 56 40.48',
    dec_degrees='-17 55 17.1',
    ra=SkyCoord('19 56 40.48 -17 55 17.1', unit=(u.hourangle, u.deg)).ra.deg,
    dec=SkyCoord('19 56 40.48 -17 55 17.1', unit=(u.hourangle, u.deg)).dec.deg,
    my_data_path= [data_dir / 'star12_G_report.txt', data_dir / 'star12_R_report.txt'],
    gaia_data_path=data_dir / 'gaia_data' / 'star12' / 'EPOCH_PHOTOMETRY-Gaia DR3 6870452273367422080.csv',
    asassn_data_path=data_dir / 'asassn_data' / 'star12' / 'APJ195640.48-175517.2.csv'
)

star13 = VariableStar2(
    star='Star 13',
    id=13,
    name='Gaia DR3 4660554298082052864',
    gaia_name='Gaia DR3 4660554298082052864',
    variable=True,
    variable_type='BCEP',
    ra_hours='05 21 41.04',
    dec_degrees='-65 52 07.2',
    ra=SkyCoord('05 21 41.04 -65 52 07.2', unit=(u.hourangle, u.deg)).ra.deg,
    dec=SkyCoord('05 21 41.04 -65 52 07.2', unit=(u.hourangle, u.deg)).dec.deg,
    my_data_path= [data_dir / 'star13_G_report.txt', data_dir / 'star13_R_report.txt'],
    gaia_data_path=data_dir / 'gaia_data' / 'star13' / 'EPOCH_PHOTOMETRY-Gaia DR3 4660554298082052864.csv',
    asassn_data_path=data_dir / 'asassn_data' / 'star13' / 'light_curve_03cf973e-1a6f-4ef7-b6c1-eadc147ea7d2_wpropermotion.csv'
)

star14 = VariableStar2(
    star='Star 14',
    id=14,
    name='Gaia DR3 5959309664916881024',
    gaia_name='Gaia DR3 5959309664916881024',
    variable=True,
    variable_type='CEP',
    ra_hours='17 33 25.28',
    dec_degrees='-40 56 28.3',
    ra=SkyCoord('17 33 25.28 -40 56 28.3', unit=(u.hourangle, u.deg)).ra.deg,
    dec=SkyCoord('17 33 25.28 -40 56 28.3', unit=(u.hourangle, u.deg)).dec.deg,
    my_data_path= [data_dir / 'star14_G_report.txt', data_dir / 'star14_R_report.txt'],
    gaia_data_path=data_dir / 'gaia_data' / 'star14' / 'EPOCH_PHOTOMETRY-Gaia DR3 5959309664916881024.csv',
    asassn_data_path=data_dir / 'asassn_data' / 'star14' / 'light_curve_56c96c57-17ab-440e-a41a-dcbfc746919a_wpropermotion.csv'
)