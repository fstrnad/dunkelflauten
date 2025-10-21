# %%
import numpy as np
import xarray as xr
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.plotting.plots as cplt
import conextremes.detect_volumes as ce
from importlib import reload

plot_dir = "/home/strnad/data/plots/extremes/"
data_dir = "/home/strnad/data/"
# %%
# Load the dataset
grid_step = 1
dataset_file = data_dir + \
    f"climate_data/{grid_step}/era5_2m_temperature_{grid_step}_ds.nc"
dataset_file2 = data_dir + \
    f"climate_data/Europe/{grid_step}/2m_temperature_{grid_step}.nc"
# Range of Europe
lon_range_cut = [-15, 40]
lat_range_cut = [35, 75]
month_range = ['Mar', 'Oct']
# %%
reload(bds)

ds_sat = bds.BaseDataset(data_nc=dataset_file2,
                         can=True,
                         an_types=['dayofyear', 'month'],
                         #  lon_range=lon_range_cut,
                         #  lat_range=lat_range_cut,
                         month_range=month_range,
                         )
# %%
reload(gplt)
reload(tu)
mean_sst = tu.compute_timemean(ds_sat.ds['2m_temperature'], timemean='all')
label_t = 'Temperature [K]'
vmin_2m_temperature = 270
vmax_2m_temperature = 300
im = gplt.plot_map(mean_sst,
                   title=f'Mean temperature',
                   plot_type='contourf',
                   cmap='RdYlBu11b',
                   levels=30,
                   label=label_t,
                   vmin=vmin_2m_temperature, vmax=vmax_2m_temperature,
                   tick_step=5,
                   orientation='horizontal',
                   )

# %%
# Create EVS dataset by tresholding the temperature by 95th percentile
reload(tu)
q = 0.99
data_array = ds_sat.ds['2m_temperature']
data_an = ds_sat.ds['2m_temperature_an_month']
evs_2m_temperature, mask = tu.compute_evs(dataarray=data_array,
                               q=q,
                               )
# %%
reload(ce)
labeled_data, num_features, comp_sizes = ce.find_connected_volumes(
    evs_2m_temperature,  min_size=200)

# %%
reload(ce)
labeled_data_minimized, df_heatwaves = ce.enforce_minimum_dimension_size(
    labeled_data=labeled_data, min_evts_per_step=250)


# %%
# plot the specific volume as a map
reload(gplt)
label_num = 6
this_df = df_heatwaves.loc[label_num]
lon_range = this_df['lon_range']
lat_range = this_df['lat_range']
dates = this_df['dates']

label_t = 'Temperature Anomalies [K]'
vmin_2m_temperature = -10
vmax_2m_temperature = -vmin_2m_temperature
ncols = 3
nrows = int(np.ceil(len(dates)/ncols))
im = gplt.create_multi_plot(ncols=ncols,
                            nrows=nrows,
                            end_idx=len(dates),
                            # title='Temperature Anomalies [K]',
                            projection='PlateCarree',
                            hspace=0.,
                            wspace=0.3,
                            # gs_lon=10,
                            # gs_lat=10
                            )

for idx, date in enumerate(dates):
    event = data_an.sel(time=date)
    im_plot = gplt.plot_map(event,
                            ax=im['ax'][idx],
                            title=f'{tu.tps2str(date)}',
                            plot_type='contourf',
                            cmap='balance',
                            levels=30,
                            #   label=label_t,
                            vmin=vmin_2m_temperature, vmax=vmax_2m_temperature,
                            tick_step=5,
                            orientation='horizontal',
                            plot_grid=True,
                            extend='both',
                            unset_label=True,
                            )
cplt.add_colorbar(im=im_plot,
                  fig=im['fig'],
                  label=label_t,
                  orientation='horizontal',
                  multi_plots=True,
                  extend='both',
                  )
# %%
# plot the specific volume as a map
reload(gplt)
label_num = 10
volume = ce.get_specific_volume(labeled_data_minimized,
                                label_value=label_num)
date_start = volume[0][0]
locs_lat1 = volume[0][1]
loc_start_lon1 = volume[0][2]
loc_start_lat2 = volume[1][1]
loc_start_lon2 = volume[1][2]

date_end = volume[-1][0]
specific_volume = volume[0]
dates = tu.get_dates_in_range(date_start, date_end)

label_t = 'Temperature [K]'
vmin_2m_temperature = -1
vmax_2m_temperature = -vmin_2m_temperature
ncols = 3
nrows = int(np.ceil(len(dates)/ncols))
im = gplt.create_multi_plot(ncols=ncols,
                            nrows=nrows,
                            end_idx=len(dates),
                            # title='Temperature Anomalies [K]',
                            projection='PlateCarree',
                            hspace=0.3,
                            wspace=0.3,
                            # gs_lon=10,
                            # gs_lat=10
                            )

for idx, date in enumerate(dates):

    event = evs_2m_temperature.sel(time=date)
    gplt.plot_map(event,
                  ax=im['ax'][idx],
                  title=f'{tu.tps2str(date)}',
                  plot_type='contourf',
                  cmap='balance',
                  levels=30,
                  unset_label=True,
                  vmin=vmin_2m_temperature, vmax=vmax_2m_temperature,
                  tick_step=5,
                  orientation='horizontal',
                  plot_grid=True,
                  )
# %%
# plot the specific volume as a time series
reload(gplt)
reload(ce)
volume = ce.get_specific_volume(labeled_data_minimized,
                                label_value=label_num)
lon_lat_pairs = gut.zip_2_lists(volume[:, 2], volume[:, 1])

# lon_lat_pairs = [[31, 67], [23, 70]]
gplt.plot_trajectory(lon_lat_pairs=lon_lat_pairs,
                     lon_range=lon_range_cut,
                     lat_range=lat_range_cut,
                     lw=2,
                     smooth_traj=True,
                     cmap='cmo.speed')

# %%
# plot as a map where the specific volume is located
reload(gplt)
reload(ce)
label_value = max_heatwave
vol_data = ce.volume2ncfile(labeled_data=labeled_data, label_value=label_value)
vmin = float(vol_data.min())
vmax = float(vol_data.max())
levels = int(vmax - vmin)
gplt.plot_map(vol_data,
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
# %%
