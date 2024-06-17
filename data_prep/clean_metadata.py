# This script attempts to perform post-processing of the metadata.csv
# of the SoundingEarth dataset.
import os
import sys
import re
import importlib.util
import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.point import Point
from shapely.geometry import Point
from shapely.ops import nearest_points
from cleantext import clean     #pip install clean-text (cleantext is a different package)
from tqdm import tqdm
from config import cfg

# Set to false to include these samples in the final_metadata.csv
ignore_corrupt_audio = True
ignore_low_sample_rate = True
ignore_missing_address = True
ignore_missing_continent = True

# used fixed buffer of 250km to get nearest continent to use as continent 
# important for locations near continent borders
distance_threshold_km = 250
sr_kHz_lesser_ignore = 16
shapefile_path = "data_prep/World_Continents/World_Continents.shp"

# Initialize geolocator
geolocator = Nominatim(user_agent="Spectrum4Geo")


def reverse_geocoding(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en')
        address = location.address
        return address
    except:
        address = None
        return address


def clean_description(description):
    sent = re.sub(r'(<br\s*/>)',' ',description)
    output = clean(sent,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks= True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                   # replace all URLs with a special token
        no_emails=True,                 # replace all email addresses with a special token
        no_phone_numbers=True,          # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct= True,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
                )
    output = re.sub(r'\s+',' ',output)
    return output


def get_caption(lat, lon, title, description):
    if pd.notna(description):
        description = description
    else:
        description = title
    
    address = reverse_geocoding(lat=lat, lon=lon)
    if address != None:
        caption = clean_description(description + '. The location of the sound is: ' + address + '.')
    else:
        caption = clean_description(description + '.')
    return caption


def find_continent(point_geom, continent_map):
    for _, continent_row in continent_map.iterrows():
        if continent_row.geometry.contains(point_geom):
            return continent_row['CONTINENT']
        nearest_geom = nearest_points(point_geom, continent_row.geometry)[1]
        if geodesic((point_geom.y, point_geom.x), (nearest_geom.y, nearest_geom.x)).kilometers <= distance_threshold_km:
            return continent_row['CONTINENT']
    return None


def calculate_distance_km(point1, point2):
    return geodesic((point1.y, point1.x), (point2.y, point2.x)).kilometers 


def main():
    data_path = cfg.data_path
    meta_df = pd.read_csv(os.path.join(data_path, 'metadata.csv'))
    print(f"Total samples loaded: {len(meta_df)}")
    exclusion_df = meta_df[['key', 'short_key']].copy()

    # Filter corrupt audio files
    if ignore_corrupt_audio:
        corrupt_ids = pd.read_csv(os.path.join(data_path, "corrupt_ids_final.csv"))['key'].tolist()
        exclusion_df['corrupt_audio'] = exclusion_df['key'].isin(corrupt_ids).astype(int)
        print(f"Samples with corrupt audio: {exclusion_df['corrupt_audio'].sum()}")

    # Filter low sample rate
    if ignore_low_sample_rate:
        low_sr_filter = meta_df['mp3samplerate'] < sr_kHz_lesser_ignore*1e3
        exclusion_df[f'lower_than_{sr_kHz_lesser_ignore}kHz_sample_rate'] = low_sr_filter.astype(int)
        print(f"Samples with sample rate lower than {sr_kHz_lesser_ignore}kHz: {low_sr_filter.sum()}")

    # Process captions and get addresses
    meta_df['caption'] = [get_caption(row.latitude, row.longitude, row.title, row.description) for row in tqdm(meta_df.itertuples(), total=meta_df.shape[0])]
    meta_df['address'] = meta_df['caption'].apply(lambda x: "The location of the sound is" + x.split("location of the sound is")[1] if "location of the sound is" in x else None)

    if ignore_missing_address:
        missing_address_filter = meta_df['address'].isna()
        exclusion_df[f'missing_address'] = missing_address_filter.astype(int)
        print(f"Samples without address information: {missing_address_filter.sum()}")

    # Filter by continents
    if ignore_missing_continent:
        geometry = [Point(lon, lat) for lon, lat in zip(meta_df['longitude'], meta_df['latitude'])]
        points_gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
        continent_map = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        meta_df['continent'] = [find_continent(point, continent_map) for point in tqdm(points_gdf.geometry, desc="Processing continents")]
        exclusion_df['missing_continent'] = meta_df['continent'].isna().astype(int)
        print(f"Samples without continent information: {exclusion_df['missing_continent'].sum()}")

    # Filter final dataset
    filters_combined = (exclusion_df.iloc[:, 2:].sum(axis=1) > 0)
    meta_df = meta_df[~filters_combined]
    print(f"Total valid samples after filtering: {len(meta_df)}")

    # Save reasons for exclusion
    exclusion_df = exclusion_df[filters_combined]
    exclusion_df.to_csv(os.path.join(data_path, 'exclusion_reasons.csv'), index=False)

    # Save final metadata
    meta_df.to_csv(os.path.join(data_path, 'final_metadata.csv'), index=False)


if __name__ == "__main__":
    if importlib.util.find_spec("unidecode"):
        main()
    else:
        raise ModuleNotFoundError("""
                                  The required package 'unidecode' is not installed. 
                                  (Since the GPL-licensed package `unidecode` is not installed, using Python's `unicodedata` package which yields worse results.) 
                                  -> pip install unidecode
                                  """)