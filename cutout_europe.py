# %%
from matplotlib.gridspec import GridSpec
from cartopy.crs import PlateCarree as plate
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.geodata.wind_dataset as wds
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.plotting.plots as gplt
import geoutils.preprocessing.open_nc_file as of
import atlite
from importlib import reload
import geoutils.countries.countries as cnt
import geoutils.countries.capacities as cap


plot_dir = "/home/strnad/data/plots/extremes/"
data_dir = "/home/strnad/data/"
# %%

europe_cntrs = cnt.get_countries_in_continent('Europe')
df_europe = cnt.get_country(europe_cntrs)
shapes_europe = cnt.get_country_shape(europe_cntrs)

country_name = "Europe"
year = 2012
dataset_name = f"{country_name}-2011-2013"
dataset_file = f"{dataset_name}.nc"

lon_range_cut = [-13, 40]
lat_range_cut = [35, 72]

cutout = atlite.Cutout(
    dataset_name,
    module="era5",
    x=slice(*lon_range_cut),
    y=slice(*lat_range_cut),
    chunks={"time": 100},
    time=slice("2011", "2013"),
)
cutout.prepare()
# %%