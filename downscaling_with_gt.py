# %%
import geoutils.utils.statistic_utils as sut
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

plot_dir = "/home/strnad/plots/dunkelflauten/downscaling"
data_dir = "/home/strnad/data/dunkelflauten/downscaling/"
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"

variables = ['tas', 'uas', 'vas', 'rsds']
europe_dir = config['europe_dir']
cmip6_dir = f'{europe_dir}/CMIP6'

gcm = 'MPI-ESM1-2-HR'
scenario = 'ssp585'
time = 'day'
country_name = 'Germany'
gs = 1.0
files_arr = []
for variable in variables:
    cmip6_folder = f"{cmip6_dir}/{country_name}/{gs}/"
    files = fut.get_files_in_folder(cmip6_folder,
                                    include_string=f'{gcm}_{scenario}_{variable}')
    if len(files) > 0:
        filename = files[0]  # always take 1st run in case of multiple
        files_arr.append(filename)

ds_cmip = of.open_nc_file(files_arr, compat='override',
                          check_vars=True
                          )
ds_cmip = gut.rename_cmip2era5(ds_cmip)
# %%
locations = {'Berlin': (13.404954, 52.520008),
             'Alps': (10, 47.2),
             'North Sea': (7.5, 54.5),
             'Munich': (11.5761, 48.1371),
             'Colonge': (6.9603, 50.9375), }
locs = np.array(list(locations.values()))
lon_range, lat_range = sput.get_lon_lat_range(ds_samples)

# %%
# Compare spatial mean values
reload(gplt)

im = gplt.create_multi_plot(nrows=1, ncols=len(variables),
                            figsize=(30, 5))

ds_dict = {'samples': ds_samples, 'gt': ds_gt}

short_time_range = ['2023-01-01', '2023-10-15']

for idx, variable in enumerate(variables):
    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        ds = tu.get_time_range_data(ds, time_range=short_time_range)
        ts_mean = ds[variable].mean(dim=['lon', 'lat'])
        if 'sample_id' in ts_mean.dims:
            ts_std = ts_mean.std(dim='sample_id')
            ts_mean = ts_mean.mean(dim='sample_id')
        else:
            ts_std = 0

        gplt.plot_2d(x=ts_mean.time.values,
                     y=ts_mean.values,
                     y_err=ts_std,
                     title=variable,
                     ax=im['ax'][idx],
                     label=ds_type,
                     color=gplt.colors[i+1],
                     rot=90)

savepath = f'{plot_dir}/{gs_gt}/spatial_mean_ts.png'
gplt.save_fig(savepath=savepath)
# %%
# Compare Correlation coefficients
variables = gut.get_vars(ds_gt)
correlation_matrices = {}
for var1 in variables:
    # Compute correlation along the 'time' dimension
    corr = xr.corr(ds_gt[var1],
                   ds_samples[var1], dim='time').compute()
    correlation_matrices[var1] = corr
# %%

nrows = len(variables)
ncols = ds_samples.dims['sample_id']
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            # figsize=(30, 30),
                            hspace=0.5,
                            projection='PlateCarree')
for idx, variable in enumerate(variables):
    for i, sample_id in enumerate(ds_samples.sample_id.values):
        corr = correlation_matrices[variable].sel(sample_id=sample_id)
        gplt.plot_map(corr,
                      ax=im['ax'][idx*ncols + i],
                      vertical_title=variable if i == 0 else None,
                      title=f'Sample {sample_id}' if idx == 0 else None,
                      vmin=0., vmax=.8,
                      cmap='viridis',
                      levels=20,
                      tick_step=5,
                      plot_borders=True,
                      label='Correlation coefficient',)

savepath = f'{plot_dir}/{gs_gt}/correlation_coefficients.png'
gplt.save_fig(savepath=savepath)

# %%
short_time_range = ['2023-12-01', '2023-12-15']

for variable in variables:
    im = gplt.create_multi_plot(nrows=2, ncols=3,
                                projection_arr=['PlateCarree', None,
                                                None, None, None, None],
                                title=f'{variable} - Single locations',
                                hspace=0.5)

    diff = ds_samples[variable] - ds_gt[variable]
    diff_mean = diff.mean(dim='sample_id').mean(dim='time')

    im_map = gplt.plot_map(diff_mean,
                           ax=im['ax'][0],
                           #    title=variable,
                           cmap='coolwarm',
                           levels=10,
                           centercolor='white',
                           tick_step=5,
                           color_bounds='absquantile',
                           plot_borders=True,
                           label='Average deviation (Sample-GT)',)

    im_map = gplt.plot_map(locs, plot_type='points',
                           ax=im['ax'][0],
                           lon_range=lon_range,
                           lat_range=lat_range,
                           plot_borders=True,
                           size=10,)

    for idx, (loc_name, loc) in enumerate(locations.items()):
        gplt.plot_text(ax=im_map['ax'],
                       text=loc_name,
                       xpos=loc[0],
                       ypos=loc[1]+.5)

        loc_data = sput.get_data_coordinate(ds_samples, loc)
        loc_data_gt = sput.get_data_coordinate(ds_gt, loc)

        data_dict = {'samples': loc_data[variable],
                     'gt': loc_data_gt[variable]}

        for i, (ds_type, ts) in enumerate(data_dict.items()):
            ts = tu.get_time_range_data(ts, time_range=short_time_range)
            if 'sample_id' in ts.dims:
                ts_std = ts.std(dim='sample_id')
                ts_mean = ts.mean(dim='sample_id')
            else:
                ts_mean = ts
                ts_std = 0
            gplt.plot_2d(x=ts_mean.time.values,
                         y=ts_mean.values,
                         y_err=ts_std,
                         title=f'{loc_name}',
                         ax=im['ax'][idx+1],
                         label=ds_type,
                         lw=1 if ds_type == 'samples' else 2,
                         color=gplt.colors[i],
                         rot=90
                         )

    savepath = f'{plot_dir}/{gs_gt}/single_locations_{variable}_{short_time_range}.png'
    gplt.save_fig(savepath=savepath)

# %%
# Plot single days
reload(gplt)

variable_dict = {
    '2m_temperature': dict(
        cmap='coolwarm',
        vmin=271, vmax=300,
        label='Temperature [K]',
        levels=20,),
    '10m_u_component_of_wind': dict(
        cmap='viridis',
        vmin=-10, vmax=10,
        label='Wind speed [m/s]',
        levels=20,),
    '10m_v_component_of_wind': dict(
        cmap='viridis',
        vmin=-10, vmax=10,
        label='Wind speed [m/s]',
        levels=20,),
    'surface_solar_radiation_downwards': dict(
        cmap='inferno',
        vmin=1e6, vmax=1e7,
        label='Solar radiation [W/m^2]',
        levels=20,),
}


num_days = 1
num_time_steps = int(24 / hourly_res) * num_days
da_obs = ds_obs[variable]
da_gt = ds_gt[variable]

start_date = '2023-06-01'
time_points = tu.get_dates_for_time_steps(start=start_date,
                                          num_steps=num_time_steps,
                                          step_size=hourly_res,
                                          freq='h')

nrows = len(ds_samples.sample_id) + 2  # +2 for mean and ground truth
ncols = len(time_points)


for variable in variables:
    im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                projection='PlateCarree',
                                hspace=0.5)
    da_samples = ds_samples[variable]
    for i, tp in enumerate(time_points):
        da_gt = ds_gt[variable].sel(time=tp)
        da_arr = [da_gt]
        label_arr = ['Ground Truth']

        da_sample_tp = da_samples.sel(time=tp)
        mean_da = da_sample_tp.mean(dim='sample_id')
        da_arr.append(mean_da)
        label_arr.append('Sample Mean')
        for sid in ds_samples.sample_id:
            da_sample = da_sample_tp.sel(sample_id=sid)
            da_arr.append(da_sample)
            label_arr.append(f'Sample {int(sid.data) + 1}')

        for j, da_sample in enumerate(da_arr):
            im_map = gplt.plot_map(da_sample,
                                   ax=im['ax'][j*ncols + i],
                                   title=f'{tu.tp2str(tp)}' if j == 0 else None,
                                   vertical_title=f'{label_arr[j]}' if i == 0 else None,
                                   cmap=variable_dict[variable]['cmap'],
                                   levels=variable_dict[variable]['levels'],
                                   label=f'{variable_dict[variable]["label"]}' if j == len(
                                       da_arr) - 1 else None,
                                   vmin=variable_dict[variable]['vmin'],
                                   vmax=variable_dict[variable]['vmax'],
                                   #    centercolor='white',
                                   tick_step=5,
                                   plot_borders=True,)

    savepath = f'{plot_dir}/{gs_gt}/trajectory_{variable}_{tu.tp2str(tp)}.png'
    gplt.save_fig(savepath=savepath)
