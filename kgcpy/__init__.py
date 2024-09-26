from PIL import Image
import pandas as pd
import numpy as np
from importlib import resources
import io
Image.MAX_IMAGE_PIXELS = None

# Helper function to load png file
def loadKMZImage(file):
    """
    Load the image file from the kgcpy package and return as a NumPy array.
    """
    # Load the image as binary data
    with resources.files('kgcpy').joinpath(file).open('rb') as fp:
        img_data = fp.read()
    
    # Open the image using Pillow
    img = Image.open(io.BytesIO(img_data))
    
    # Convert the image to a NumPy array for faster access
    return np.array(img)

# Helper function to load different CSV files
def loadCSV(file):
    with resources.files('kgcpy').joinpath(file).open('rb') as fp:
        kgc_csv = fp.read()

    if file == 'kg_zoneNum.csv':
        return pd.read_csv(io.BytesIO(kgc_csv))
    elif file == 'zipcodes.csv':
        return pd.read_csv(io.BytesIO(kgc_csv), index_col=0, dtype={'zip':'string'})
    elif file == 'df_quantile.csv':
        return pd.read_csv(io.BytesIO(kgc_csv), index_col=0, dtype={'kg_zone':'string'})

# Loading image and CSV files for lookup functions below
img = loadKMZImage('kmz_int_reshape.png')
kg_zoneNum_df = loadCSV('kg_zoneNum.csv')
zips_df = loadCSV('zipcodes.csv')
quantiles_df = loadCSV('df_quantile.csv')

# This function will return the climate zone for the co-ordinates provided.
def lookupCZ(lat,lon):
    """
    This function will return the climate zone for the co-ordinates provided.
    _summary_

    Args:
        lat (_type_): latitude
        lon (_type_): longitude

    Returns:
        _type_: _description_
    """

    # Get the KG zone values of the pixel at position (x, y)
    x = round((lon+180)*(img.size[0])/360 - 0.5)
    y = round(-(lat-90)*(img.size[1])/180 - 0.5)
    num = img.getpixel((x, y))

    # Use the loc method to find the index of the row that matches the input values
    res = kg_zoneNum_df['kg_zone'].loc[kg_zoneNum_df['zoneNum'] == num]

    return res.values[0]

# This function will return the data frame with the longitude and latitude of the zip codes
def translateZipCode(zipcode):
    """
    This function will return the data frame with the longitude and latitude of the zip codes

    _summary_

    Args:
        zipcode (_type_): zipcode

    Returns:
        _type_: _description_
    """
    zipcode = str(zipcode)
        
    try:
        rows = zips_df.loc[zips_df['zip'] == zipcode]
        if len(rows) == 0:
            return f"No matching rows found for zipcode {zipcode}"
        else:
            return rows['lat'].iloc[0], rows['lon'].iloc[0]
    except Exception as e:
        return f"Search failed: {e}"

# Get irradiance quantiles for each Koppen Geiger Climate Zone
def irradianceQuantile(kg_zone):
    """
    Get irradiance quantiles for each Koppen Geiger Climate Zone

    _summary_

    Args:
        kg_zone (_type_): Koppen Geiger zone

    Returns:
        _type_: _description_
    """
    # kg_zone = str(kg_zone)

    try:
        rows = quantiles_df.loc[quantiles_df['kg_zone'] == kg_zone]
        if len(rows) == 0:
            return f"Climate zone {kg_zone} doesn't exist"
        else:
            return rows['quantilep98'].iloc[0], rows['quantilep80'].iloc[0], rows['quantilep50'].iloc[0], rows['quantilep30'].iloc[0]
    except Exception as e:
        return f"Search failed: {e}"

# The inputed number to nearest ’fine’ (100s) resolution grid point.
def roundCoordinates(lat,lon):
    """
    The inputed number to nearest 'fine' (100s) resolution grid point.

    _summary_

    Args:
        lat (_type_): latitude
        lon (_type_): longitude

    Returns:
        _type_: _description_
    """

    # Get the RGB values of the pixel at position (x, y)
    x = round((lon+180)*(img.size[0])/360 - 0.5)
    y = round(-(lat-90)*(img.size[1])/180 - 0.5)

    lonRound = round(((x + 0.5) * 360 / img.size[0] - 180), 2)
    latRound = round ((- (y + 0.5) * 180 / img.size[1] + 90), 2)

    return latRound, lonRound

#get possible climate zones from nearby pixels, and compare to the center pixel; same as function CZUncertainty() in kgc R package 
def nearbyCZ(lat,lon,size=1):
    """
    get possible climate zones from nearby pixels, and compare to the center pixel; same as function CZUncertainty() in kgc R package 

    _summary_

    Args:
        lat (_type_): latitude
        lon (_type_): longitude
        size (int, optional): size of nearby pixel. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Get the RGB values of the pixel at position (x, y)
    x = round((lon+180)*(img.size[0])/360 - 0.5)
    y = round(-(lat-90)*(img.size[1])/180 - 0.5)
 
    climateZones = []
    climateZone = ''

    for i in range(x-size, x+size+1):
        for j in range(y-size, y+size+1):
            try:
                num = img.getpixel((i, j))
                # rgb_values = {'R': r, 'G': g, 'B': b}
                # Use the loc method to find the index of the row that matches the input values
                cz = kg_zoneNum_df['kg_zone'].loc[kg_zoneNum_df['zoneNum'] == num]
                climateZones.append(cz.values[0])
                if i == x and j == y:
                    climateZone = cz.values[0]
            except IndexError:
                pass
    
    climateZones_series = pd.Series(climateZones)
    climateZones_counts = climateZones_series.value_counts()
    climateZones_percentage = climateZones_counts / climateZones_counts.sum()
    uncertaintyNearbyCZ = climateZones_percentage[climateZone]

    nearbyCZ = climateZones_series.unique().tolist()
    nearbyCZ.remove(climateZone)

    return climateZone, uncertaintyNearbyCZ, nearbyCZ

def vectorized_lookupCZ(lat_array, lon_array):
    """
    This function will return the climate zone for the provided arrays of coordinates.
    
    Args:
        lat_array (np.array): Array of latitudes.
        lon_array (np.array): Array of longitudes.

    Returns:
        np.array: Array of climate zones.
    """
    # Ensure lat_array and lon_array are NumPy arrays
    lat_array = np.array(lat_array)
    lon_array = np.array(lon_array)

    # Vectorized computation of x and y coordinates from lat/lon
    x = np.round((lon_array + 180) * (img.shape[1]) / 360 - 0.5).astype(int)
    y = np.round(-(lat_array - 90) * (img.shape[0]) / 180 - 0.5).astype(int)

    # Ensure x and y are within valid ranges (clipping)
    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    # Vectorized pixel lookup from the image NumPy array
    pixel_values = img[y, x]

    # Vectorized lookup in the kg_zoneNum_df DataFrame to get climate zones
    climate_zones = kg_zoneNum_df['kg_zone'].loc[kg_zoneNum_df['zoneNum'].isin(pixel_values)].values

    # Create an array of climate zones corresponding to each input coordinate
    zone_to_climate = dict(zip(kg_zoneNum_df['zoneNum'], kg_zoneNum_df['kg_zone']))

    # Lookup the climate zone for each pixel value
    climate_zones = np.array([zone_to_climate.get(pv, np.nan) for pv in pixel_values])

    return climate_zones
