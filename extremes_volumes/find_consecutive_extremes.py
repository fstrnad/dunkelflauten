# %%
import numpy as np
import xarray as xr
import geoutils.geodata.wind_dataset as wds
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.plotting.plots as cplt
import geoutils.preprocessing.open_nc_file as of

import conextremes.detect_volumes as ce
from importlib import reload

plot_dir = "/home/strnad/data/plots/extremes/"
data_dir = "/home/strnad/data/"

# %%
# Load the dataset
grid_step = 1
lon_range_europe = [-15, 40]
lat_range_europe = [35, 75]
lev = 900

# surface wind speed
reload(wds)
dataset_file_uws = f"{data_dir}/climate_data/Europe/{grid_step}/100m_u_component_of_wind_{grid_step}.nc"
dataset_file_vws = f"{data_dir}/climate_data/Europe/{grid_step}/100m_v_component_of_wind_{grid_step}.nc"
dataset_file_ws = f"{data_dir}/climate_data/Europe/10m_wind_speed_{grid_step}.nc"

# %%
reload(wds)
ds_wind = wds.Wind_Dataset(data_nc_u=dataset_file_uws,
                           data_nc_v=dataset_file_vws,
                           can=True,
                           an_types=['month', 'JJAS'],
                           compute_ws=True,
                           )
# %%
reload(tu)
q = 0.01
threshold = 1.5
min_num_events = 1
data_array = ds_wind.ds['windspeed']
evs_ws, mask = tu.compute_evs(dataarray=data_array,
                              threshold=threshold,
                              q=q,
                              reverse_treshold=True,
                              min_num_events=min_num_events,
                              )
# %%
reload(gplt)
evs_ws_mean = ds_wind.ds['u'].mean(dim=['lat', 'lon'])
gplt.plot_2d(evs_ws_mean.data, title='EVS windspeed',
             label='EVS', xlabel='Day')

# %%
reload(ce)
labeled_data, num_features, comp_sizes = ce.find_connected_volumes(
    evs_ws,  min_size=450)
# %%
# check for largest volume
reload(ce)
ws_minimized, df_flaute = ce.enforce_minimum_dimension_size(
    labeled_data=labeled_data, min_evts_per_step=100)
df_flaute
# %%
# plot the specific volume as a map
reload(gplt)
reload(sput)
reload(tu)
label_num = 4
this_df = df_flaute.loc[label_num]
lon_range = this_df['lon_range']
lat_range = this_df['lat_range']
dates = this_df['dates']

label_t = 'Windspeed [m/s]'
vmin_t2m = 0
vmax_t2m = 5
ncols = 4
nrows = int(np.ceil(len(dates)/ncols))
im = gplt.create_multi_plot(ncols=ncols,
                            nrows=nrows,
                            # end_idx=len(dates),
                            projection='PlateCarree',
                            hspace=0.1748,
                            wspace=0.3,
                            end_idx=len(dates),
                            )

for idx, date in enumerate(dates):

    event = ds_wind.ds['wind_speed'].sel(time=date)
    im_plot = gplt.plot_map(event,
                            ax=im['ax'][idx],
                            title=f'{tu.tp2str(date)}',
                            plot_type='contourf',
                            cmap='RdYlBu_r',
                            levels=30,
                            unset_label=True,
                            vmin=vmin_t2m, vmax=vmax_t2m,
                            tick_step=5,
                            orientation='horizontal',
                            plot_grid=True,
                            )

    cplt.plot_rectangle(ax=im_plot['ax'],
                        lon_range=lon_range,
                        lat_range=lat_range,
                        lw=2,)
_ = cplt.add_colorbar(im=im_plot,
                      fig=im['fig'],
                      label=label_t,
                      orientation='horizontal',
                      multi_plots=True,
                      extend='both',
                      )

# %%
# plot as a map where the specific volume is located
reload(gplt)
reload(ce)
label_value = max_low_wind
vol_data = ce.volume2ncfile(labeled_data=labeled_data, label_value=label_value)
vmin = float(vol_data.min())
vmax = float(vol_data.max())
levels = int(vmax - vmin)
_ = gplt.plot_map(vol_data,
                  title=f'{tu.tps2str(date)}',
                  plot_type='colormesh',
                  cmap='cmo.speed',
                  levels=levels,
                  label='Day',
                  vmin=float(vol_data.min()), vmax=float(vol_data.max()),
                  tick_step=2,
                  orientation='horizontal',
                  plot_grid=True,
                  )
