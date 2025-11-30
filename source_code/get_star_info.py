import pandas as pd
from pathlib import Path

from classes import VariableStar


# Function to get all information for all stars
def get_all_star_info():
    catalog = pd.read_csv(Path(__file__).parent  / 'Official_log-Stars_&_Ids.csv')
    stars = {}
    for _, row in catalog.iterrows():

        star_id = row["Star id"]
        filters = ['G', 'R', 'B'] if star_id == 7 else ['G', 'R']
        filters = ['G'] if star_id == 4 else filters
        paths = [f'star{star_id}_{filt}_report.txt' for filt in filters]

        s = VariableStar(
            star=row["Star"],
            id=star_id,
            name=row["Object name"],
            gaia_name=row["gaia id"],
            variable=row["Var?"],
            variable_type=row["Var type"],
            ra=row["COORDS J200 (RA DEC)"].split()[0],
            dec=row["COORDS J200 (RA DEC)"].split()[1],
            filters=filters,
            paths=paths
        )
        stars[s.star] = s
    return stars    

# Function to get information for a specific star by it's ID
def get_star_info(star_id):
    catalog = pd.read_csv(Path(__file__).parent  / 'Official_log-Stars_&_Ids.csv')
    row = catalog.loc[catalog["Star id"] == star_id].iloc[0]

    star_id = row["Star id"]
    filters = ['G', 'R', 'B'] if star_id == 7 else ['G', 'R']
    filters = ['G'] if star_id == 4 else filters
    paths = [f'star{star_id}_{filt}_report.txt' for filt in filters]

    s = VariableStar(
        star=row["Star"],
        id=star_id,
        name=row["Object name"],
        gaia_name=row["gaia id"],
        variable=row["Var?"],
        variable_type=row["Var type"],
        ra=row["COORDS J200 (RA DEC)"].split()[0],
        dec=row["COORDS J200 (RA DEC)"].split()[1],
        filters=filters,
        paths=paths
    )
    return s