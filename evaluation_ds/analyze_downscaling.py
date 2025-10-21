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
data_dir = "/home/strnad/data/dunkelflauten/downscaling/test_01"
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"

# %%
reload(of)
sample_path = f'{data_dir}/samples.nc'
gt_path = f'{data_dir}/gt.nc'
obs_path = f'{data_dir}/obs.nc'
ds_samples = of.open_nc_file(sample_path)
ds_gt = of.open_nc_file(gt_path)
ds_obs = of.open_nc_file(obs_path)
gs, _, _ = sput.get_grid_step(ds_samples)
gs_obs, _, _ = sput.get_grid_step(ds_obs)
variables = gut.get_vars(ds_samples)
# %%
reload(tu)
sample_ids = ds_samples['sample_id'].values

country_name = 'Germany'
files_gt = []
hourly_res = 6
time_range = tu.get_time_range(ds_samples)
for variable in variables:
    files_gt.append(
        f'{era5_dir}/{country_name}_nn/{gs}/{variable}_{gs}_{hourly_res}h.nc')

ds_gt = of.open_nc_file(files_gt,
                        time_range=time_range)

# %%
# Compare spatial mean values
reload(gplt)

im = gplt.create_multi_plot(nrows=1, ncols=len(variables),
                            figsize=(30, 5))

ds_dict = {'samples': ds_samples, 'gt': ds_gt}

short_time_range = ['2023-10-01', '2023-10-15']

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

savepath = f'{plot_dir}/spatial_mean_ts.png'
gplt.save_fig(savepath=savepath)

# %%
# compare the distribution of the samples and the ground truth
reload(gplt)

im = gplt.create_multi_plot(nrows=1, ncols=len(variables),
                            figsize=(30, 5))
for idx, variable in enumerate(variables):
    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        plot_data = []
        if ds_type == 'samples':
            for sample_id in ds.sample_id.values:
                plot_data.append(ds.sel(sample_id=sample_id)
                                 [variable].values.flatten())
        else:
            plot_data.append(ds[variable].values.flatten())

        gplt.plot_hist(plot_data,
                       ax=im['ax'][idx],
                       title=variable,
                       color=gplt.colors[i],
                       label=ds_type,
                       nbins=100,
                       lw=1 if ds_type == 'samples' else 2,
                       alpha=0.5 if ds_type == 'samples' else 1
                       )

savepath = f'{plot_dir}/distribution_values.png'
gplt.save_fig(savepath=savepath)

# %%
# Compare the pixelwise deviation from the ground truth
reload(sut)
reload(gplt)
nrows = len(variables) // 2
ncols = np.ceil(len(variables) / nrows).astype(int)
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            hspace=0.5,)

for idx, variable in enumerate(variables):
    diff = ds_samples[variable] - ds_gt[variable]
    mean_ds = ds_samples[variable].mean(dim='time').mean(dim='sample_id')
    std_ds = ds_samples[variable].std(dim='time').mean(dim='sample_id')
    diff_mean = diff.mean(dim='sample_id').mean(dim='time') / mean_ds
    diff_std = diff.std(dim='sample_id').mean(dim='time') / std_ds

    gplt.plot_map(diff_mean,
                  ax=im['ax'][idx],
                  title=variable,
                  cmap='RdYlBu_r',
                  levels=10,
                  centercolor='white',
                  tick_step=5,
                  color_bounds='absquantile',
                  plot_borders=True,
                  label='Average deviation (Sample-GT)',)

savepath = f'{plot_dir}/mean_deviation_pixelwise.png'
gplt.save_fig(savepath=savepath)

# %%
# Single locations
reload(gplt)

locations = {'Berlin': (13.404954, 52.520008),
             'Alps': (10, 47.2),
             'North Sea': (7.5, 54.5),
             'Munich': (11.5761, 48.1371),
             'Colonge': (6.9603, 50.9375), }
locs = np.array(list(locations.values()))
lon_range, lat_range = sput.get_lon_lat_range(ds_samples)

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

    savepath = f'{plot_dir}/single_locations_{variable}_{short_time_range}.png'
    gplt.save_fig(savepath=savepath)

# %%
# Power spectral density plots
reload(sput)
reload(sput)
im = gplt.create_multi_plot(nrows=1, ncols=len(variables),
                            figsize=(30, 5),
                            ylog=True,
                            xlog=True,)
for idx, variable in enumerate(variables):
    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        plot_data = []
        if ds_type == 'samples':
            for sample_id in ds.sample_id.values:
                data = ds.sel(sample_id=sample_id)[variable]
                mean_psd, freq = sput.mean_rapsd(data)
                plot_data.append(mean_psd)
        else:
            data = ds[variable]
            mean_psd, freq = sput.mean_rapsd(data)
            plot_data.append(mean_psd)
        gplt.plot_2d(
            x=freq*111.11,
            y=plot_data,
            ax=im['ax'][idx],
            ylog=True,
            title=variable,
            color=gplt.colors[i],
            label=ds_type,
            lw=1 if ds_type == 'samples' else 2,
            alpha=0.5 if ds_type == 'samples' else 1,
            ylabel='Power spectral density [a.u.]',
            xlabel='Wavelength [km]',
        )

savepath = f'{plot_dir}/compare_rapsds.png'
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
                      vmin=0, vmax=1,
                      cmap='viridis',
                      levels=20,
                      tick_step=5,
                      plot_borders=True,
                      label='Correlation coefficient',)

savepath = f'{plot_dir}/correlation_coefficients.png'
gplt.save_fig(savepath=savepath)