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


plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_gt/"
data_dir = "/home/strnad/data/dunkelflauten/downscaling/"
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"

locations = {'Berlin': (13.404954, 52.520008),
             'Alps': (10, 47.2),
             'North Sea': (7.5, 54.5),
             'Munich': (11.5761, 48.1371),
             'Cologne': (6.9603, 50.9375), }
locs = np.array(list(locations.values()))
# %%
reload(of)
tr = '2023-01-01_2025-01-01'
N = 10
fine_res = 0.25
coarse_res = 1.0
folder_name = f'{tr}_N{N}'
use_log = False
data_str = f'2023-01-01_2025-01-01'
samples_path = f'{data_dir}/eval_with_gt/{folder_name}/samples_era5_{data_str}_{fine_res}_log_{use_log}.nc'
gt_path = f'{data_dir}/eval_with_gt/{folder_name}/gt_era5_{tr}_{fine_res}_log_{use_log}.nc'
obs_path = f'{data_dir}/eval_with_gt/{folder_name}/obs_era5_{data_str}_{coarse_res}_log_{use_log}.nc'

ds_samples = of.open_nc_file(samples_path)
ds_gt = of.open_nc_file(gt_path)
ds_obs = of.open_nc_file(obs_path)
gs, _, _ = sput.get_grid_step(ds_samples)
gs_obs, _, _ = sput.get_grid_step(ds_obs)
variables = gut.get_vars(ds_samples)
lon_range, lat_range = sput.get_lon_lat_range(ds_samples)

short_time_range = ['2023-01-01', '2023-01-31']

ds_dict = {'samples': ds_samples.load(),
           'gt': ds_gt.load(),
           'obs': ds_obs.load(), }

variable_dict = {
    '2m_temperature': dict(
        cmap='RdBu_r',
        vmin=-10, vmax=35,
        label='Temperature [°C]',
        levels=20,
        vname='Surface Air Temperature',
        offset=-271.15),
    '10m_u_component_of_wind': dict(
        cmap='plasma',
        vmin=-15, vmax=15,
        label='Wind speed [m/s]',
        offset=0,
        levels=20,
        vname='10m U Component of Wind',),
    '10m_v_component_of_wind': dict(
        cmap='viridis',
        vmin=-13, vmax=13,
        label='Wind speed [m/s]',
        offset=0,
        levels=20,
        vname='10m V Component of Wind',),
    'surface_solar_radiation_downwards': dict(
        cmap='inferno',
        vmin=0, vmax=1.5e6,
        label=r'Solar radiation [W/m$^2$]',
        offset=0,
        levels=20,
        yrange=(0, .23e-5),
        vname='Surface Solar Radiation Downwards',),
}


# %%
# compare the distribution of the samples and the ground truth
reload(gplt)

variables = gut.get_vars(ds_samples)
nrows = 2
sd_str, ed_str = tu.get_time_range(
    ds_samples, asstr=True, m=False, d=False, h=False)

tr_distr = f'{sd_str}_{ed_str}'
timemean = 'day'
timemean = None
im = gplt.create_multi_plot(nrows=nrows,
                            ncols=len(variables) // nrows,
                            figsize=(15, 10),
                            y_title=1.,
                            hspace=0.5)

colors = ['red', 'blue', 'tab:blue', 'tab:green']
sv = 'surface_solar_radiation_downwards'
for idx, variable in enumerate(variables):
    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        ds = tu.compute_timemean(
            ds, timemean=timemean)  # if variable == 'surface_solar_radiation_downwards' else ds
        plot_data = []
        offset = variable_dict[variable]['offset']
        if ds_type == 'samples':
            for sample_id in ds.sample_id.values:
                plot_data.append(ds.sel(sample_id=sample_id)
                                 [variable].values.flatten() + offset)
        else:
            plot_data.append(ds[variable].values.flatten() + offset)

        this_im = gplt.plot_hist(plot_data,
                                 ax=im['ax'][idx],
                                 title=variable_dict[variable]['vname'],
                                 color=colors[i],
                                 label=ds_type,
                                 nbins=100,
                                 lw=1 if ds_type == 'samples' else 2,
                                 alpha=0.8 if ds_type == 'samples' else 1,
                                 set_yaxis=False,
                                 ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                                 ylabel='Density',
                                 density=True,
                                 xlabel=variable_dict[variable]['label'],
                                 xlim=(variable_dict[variable]['vmin'],
                                        variable_dict[variable]['vmax']),
                                 )
        if ds_type == 'samples':
            gplt.fill_between(ax=this_im['ax'],
                              x=this_im['bc'],
                              y=this_im['hc'],
                              y2=0,
                              color=colors[i],
                              alpha=0.15,
                              )

savepath = f'{plot_dir}/distributions/distribution_values_{tr_distr}_tm{timemean}_log_{use_log}.png'
gplt.save_fig(savepath=savepath)

# %%
# Compare spatial mean values
reload(gplt)

im = gplt.create_multi_plot(nrows=1, ncols=len(variables),
                            figsize=(30, 5))


tr_str = f'{short_time_range[0]}_{short_time_range[1]}'
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

savepath = f'{plot_dir}/time_series/spatial_mean_ts_{tr_str}.png'
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

tr_dev = tu.tps2str(tu.get_time_range(ds_samples),
                    h=False, full_str=True)
for idx, variable in enumerate(variables):
    diff = ds_samples[variable] - ds_gt[variable]
    diff_mean = diff.mean(dim='sample_id').mean(dim='time')
    mean_ds = ds_samples[variable].mean(dim='time').mean(dim='sample_id')
    std_ds = ds_samples[variable].std(dim='time').mean(dim='sample_id')
    diff_mean_norm = diff_mean / np.abs(mean_ds)
    diff_std = diff_mean / std_ds

    gplt.plot_map(diff_mean_norm,
                  ax=im['ax'][idx],
                  title=variable_dict[variable]['vname'],
                  cmap='PuOr',
                  levels=10,
                  centercolor='white',
                  tick_step=5,
                  color_bounds='absquantile',
                  plot_borders=True,
                  label='Average deviation (Sample-GT)/std',)

savepath = f'{plot_dir}/mean_deviation_pixelwise_{tr_dev}.png'
gplt.save_fig(savepath=savepath)

# %%
# Single locations
reload(gplt)
tr = tu.tps2str(tu.get_time_range(ds_samples),
                h=False, full_str=True)

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

    savepath = f'{plot_dir}/single_locations/single_locations_{variable}_{tr}.png'
    gplt.save_fig(savepath=savepath)

# %%
# Power spectral density plots
reload(sput)
reload(sput)
tr_rapsd = tu.tps2str(tu.get_time_range(ds_samples),
                      h=False, full_str=True)

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
            title=variable_dict[variable]['vname'],
            color=gplt.colors[i],
            label=ds_type,
            lw=1 if ds_type == 'samples' else 2,
            alpha=0.5 if ds_type == 'samples' else 1,
            ylabel='Power spectral density [a.u.]',
            xlabel='Wavelength [km]',
        )

savepath = f'{plot_dir}/rapsd/compare_rapsds_{tr_rapsd}_mixed.png'
gplt.save_fig(savepath=savepath)

# %%
# Compare Correlation coefficients
reload(gut)
variables = gut.get_vars(ds_gt)
correlation_matrices = {}
for var1 in variables:
    # Compute correlation along the 'time' dimension
    da = ds_samples[var1]
    da = gut.add_mean_along_dim(da, dim_name='sample_id')
    corr = xr.corr(ds_gt[var1],
                   da, dim='time').compute()
    correlation_matrices[var1] = corr

tr_corr = tu.tps2str(tu.get_time_range(ds_samples),
                     h=False, full_str=True)
# %%

nrows = len(variables)
ncols = ds_samples.dims['sample_id'] + 1
ncols = 4 if ds_samples.dims['sample_id'] > 3 else ds_samples.dims['sample_id']
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            # figsize=(30, 30),
                            hspace=0.2,
                            wspace=0.1,
                            projection='PlateCarree')
for idx, variable in enumerate(variables):
    corrs = correlation_matrices[variable]
    for i, sample_id in enumerate(corrs.sample_id.values):

        corr = corrs.sel(sample_id=sample_id)
        gplt.plot_map(corr,
                      ax=im['ax'][idx*ncols + i],
                      vertical_title=variable_dict[variable]['vname'] if i == 0 else None,
                      title=f'Sample {sample_id}' if idx == 0 else None,
                      vmin=0.3, vmax=1,
                      cmap='winter',
                      levels=20,
                      tick_step=5,
                      plot_borders=True,
                      label='Correlation coefficient' if idx == len(variables)-1 else None,)
        if i >= ncols-1:
            break


savepath = f'{plot_dir}/correlations/correlation_coefficients_{tr_corr}.png'
gplt.save_fig(savepath=savepath)

# %%
reload(gplt)
ds_gt = ds_dict['gt']

num_days = 1
hourly_res = 6
start_date = '2023-07-01'
start_date = '2023-10-30'  # storm Ciearan
num_time_steps = int(24 / hourly_res) * num_days
time_points = tu.get_tps_start(start_date,
                               steps=num_time_steps,
                               time_delta=hourly_res)


plot_samples = False
nrows = 3 if not plot_samples else 3 + len(ds_samples.sample_id)
ncols = len(time_points)
variables = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_solar_radiation_downwards'
]

for variable in variables:
    im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                projection='PlateCarree',
                                rem_idx=[1, 2, 3],
                                hspace=0.3)
    da_samples = ds_samples[variable]
    da_samples = gut.add_mean_along_dim(da_samples, dim_name='sample_id')
    offset = variable_dict[variable]['offset']
    for i, tp in enumerate(time_points):
        da_obs = ds_obs[variable].sel(time=tp, method='nearest')
        da_arr = [da_obs]
        label_arr = ['Coarse Input']

        da_sample_tp = da_samples.sel(time=tp)
        if plot_samples:
            for sid in da_samples.sample_id.values:
                da_sample = da_sample_tp.sel(sample_id=sid)
                da_arr.append(da_sample)
                label_arr.append(f'Sample {sid}')
        else:
            da_arr.append(da_sample_tp.sel(sample_id='mean'))
            label_arr.append('Sample Mean')

        da_gt = ds_gt[variable].sel(time=tp, method='nearest')
        da_arr.append(da_gt)
        label_arr.append('Ground Truth')

        for j, da_sample in enumerate(da_arr):
            this_ax = im['ax'][j*ncols + i]
            if j == 0 and i > 0:
                this_ax.axis('off')
            else:
                im_map = gplt.plot_map(da_sample + offset,
                                       ax=im['ax'][j*ncols + i],
                                       title=f'{tu.tp2str(tp, h=j)}' if j <= 1 else None,
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

    savepath = f'{plot_dir}/trajectories/{variable}_{tu.tp2str(tp, h=False)}.png'
    gplt.save_fig(savepath=savepath)


# %%
