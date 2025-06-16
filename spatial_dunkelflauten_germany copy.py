# %%
from scipy import stats
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
import geoutils.utils.file_utils as fut
import geoutils.plotting.plots as gplt
import geoutils.preprocessing.open_nc_file as of
from importlib import reload
import geoutils.countries.countries as cnt
import geoutils.countries.capacities as cap
import conextremes.detect_volumes as ce


plot_dir = "/home/strnad/plots/dunkelflauten/"
data_dir = "/home/strnad/data/dunkelflauten/"
# %%
grid_step = 1
country_name = "Germany"
savepath = f'{data_dir}/{country_name}/pv_wind_{grid_step}.nc'


# %%
reload(of)
savepath_wind = f"{data_dir}/{country_name}/cap_wind_{grid_step}.nc"
savepath_pv = f"{data_dir}/{country_name}/cap_pv_{grid_step}.nc"

cap_wind = of.open_nc_file(savepath_wind, to_dataarray=True)
cap_pv = of.open_nc_file(savepath_pv, to_dataarray=True)

# %%
# Plot the Capacity factors per grid cell
reload(gplt)
cap_wind_mean = cap_wind.mean(dim=['lat', 'lon'])
im = gplt.plot_2d(y=cap_wind_mean.data,
                  x=cap_wind_mean.time,
                  title='CF windspeed',
                  label='CF', xlabel='Day')
_ = gplt.plot_hline(y=.05, color='red', linestyle='--', ax=im['ax'])
# %%
reload(gplt)
reload(tu)

im = gplt.create_multi_plot(nrows=1, ncols=3,
                            projection='PlateCarree',
                            wspace=0.4)

titles = ['Wind', 'PV', 'Wind + PV']
for idx, cap in enumerate([cap_wind, cap_pv, (cap_pv + cap_wind)/2]):
    threshold = 0.05
    min_num_events = 1
    q_val_map, ee_map, data_quantile, rel_frac_q_map = tu.get_ee_ds(
        dataarray=cap,
        # q=q,
        threshold=threshold,
        threshold_type='lower',)

    gplt.plot_map(ee_map,
                  ax=im['ax'][idx],
                  title=titles[idx],
                  plot_borders=True,
                  vmin=3e4, vmax=5e4,
                  levels=10,
                  label='Num EEs')


# %%
reload(tu)
q = 0.95
threshold = 0.05
min_num_events = 1
evs_wind, mask_wind = tu.compute_evs(dataarray=cap_wind,
                                     threshold=threshold,
                                     q=q,
                                     threshold_type='lower',
                                     min_num_events=min_num_events,
                                     )
evs_pv, mask_pv = tu.compute_evs(dataarray=cap_pv,
                                 threshold=threshold,
                                 threshold_type='lower',
                                 min_num_events=min_num_events,
                                 )
# %%
reload(gplt)
gplt.plot_map(mask_wind, plot_borders=True,
              label='mask')
# %%
reload(ce)
labeled_data_wind, num_features_wind, comp_sizes_wind = ce.find_connected_volumes(
    cap_wind,  min_size=50)
# %%
# check for largest volume
reload(ce)
ws_minimized, df_flaute = ce.enforce_minimum_dimension_size(
    labeled_data=labeled_data, min_evts_per_step=10)
df_flaute
