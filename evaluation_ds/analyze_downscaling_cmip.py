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

from pathlib import Path
import json
from importlib import reload
import yaml

plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_cmip6/"
data_dir = "/home/strnad/data/CMIP6/downscaling/"
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# %%
reload(of)
country_name = 'Germany'
gs_dws = 1.0
fine_res = 0.25
gcm = 'GFDL-ESM4'
gcm = 'MPI-ESM1-2-HR'
ssp = 'historical'
start_date = '2000-01-01'
end_date = '2010-01-01'
time_str = f'{start_date}_{end_date}'
gcm_str = f'{gcm}_{ssp}'
N = 3
sample_folder = f'{data_dir}/{gs_dws}/{gcm_str}/{time_str}_N{N}/'
sample_path = fut.get_files_in_folder(sample_folder,
                                      include_string=f'{fine_res}')[0]
ds_Samples = of.open_nc_file(sample_path)
sd, ed = tu.get_time_range(ds_Samples)

obs_path_bc = f'{cmip6_dir}/{country_name}/{float(gs_dws)}_bc/{gcm_str}_eval.nc'
ds_obs_bc = of.open_nc_file(obs_path_bc)
ds_obs_bc = tu.get_time_range_data(ds_obs_bc,
                                   time_range=(sd, ed))

obs_path = f'{cmip6_dir}/{country_name}/{float(gs_dws)}/{gcm_str}_eval.nc'
ds_obs = of.open_nc_file(obs_path)
ds_obs = tu.get_time_range_data(ds_obs,
                                time_range=(sd, ed))

gs, _, _ = sput.get_grid_step(ds_Samples)
variables = gut.get_vars(ds_Samples)
tr = tu.tps2str(tu.get_time_range(ds_Samples),
                h=False, full_str=True)

# %%
# ERA5 data
country_name = 'Germany'
files_gt = []
hourly_res = 6
for variable in variables:
    files_gt.append(
        f'{era5_dir}/{country_name}_nn/{gs}/{variable}_{gs}_{hourly_res}h.nc')

ds_era5 = of.open_nc_file(files_gt)


# %%
ds_dict = {
    'ERA5': ds_era5.load(),
    'CMIP6': ds_obs.load(),
    'CMIP6 bc': ds_obs_bc.load(),
    'Samples': ds_Samples.load(),
}
# %%
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
# Compare distributions of Samples and ground truth
reload(gplt)
reload(tu)
# variables = [
#     'surface_solar_radiation_downwards',
#     # '2m_temperature'
# ]
variables = gut.get_vars(ds_Samples)
nrows = 2
sd_str, ed_str = tu.get_time_range(
    ds_Samples, asstr=True, m=False, d=False, h=False)

time_range = tu.get_time_range(ds_Samples)

im = gplt.create_multi_plot(nrows=nrows,
                            ncols=len(variables) // nrows,
                            figsize=(15, 10),
                            title=f'Distribution for {gcm} {ssp} ({sd_str}-{ed_str})',
                            y_title=1.,
                            hspace=0.5)

colors = ['red', 'blue', 'tab:blue', 'tab:green']

for idx, variable in enumerate(variables):
    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        ds = tu.get_time_range_data(ds, time_range=time_range)
        ds = tu.compute_timemean(
            ds, timemean='day')  # if variable == 'surface_solar_radiation_downwards' else ds
        plot_data = []
        offset = variable_dict[variable]['offset']
        if ds_type == 'Samples':
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
                                 lw=2,
                                 alpha=0.8 if ds_type == 'Samples' else 1,
                                 set_yaxis=False,
                                 ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                                 ylabel='Density',
                                 density=True,
                                 xlabel=variable_dict[variable]['label'],
                                 xlim=(variable_dict[variable]['vmin'],
                                        variable_dict[variable]['vmax']),
                                 )
        if ds_type == 'Samples':
            gplt.fill_between(ax=this_im['ax'],
                              x=this_im['bc'],
                              y=this_im['hc'],
                              y2=0,
                              color=colors[i],
                              alpha=0.15,
                              )


savepath = f'{plot_dir}/distributions/{gcm_str}_{tr}.png'
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
        if ds_type == 'Samples':
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
            lw=1 if ds_type == 'Samples' else 2,
            alpha=0.5 if ds_type == 'Samples' else 1,
            ylabel='Power spectral density [a.u.]',
            xlabel='Wavelength [km]',
        )

savepath = f'{plot_dir}/compare_rapsds.png'
gplt.save_fig(savepath=savepath)

# %%
num_days = 1
hourly_res = 6
start_date = '2000-07-01'
num_time_steps = int(24 / hourly_res) * num_days
time_points = tu.get_tps_start(start_date,
                               steps=num_time_steps,
                               time_delta=hourly_res)
plot_samples = True
# +1 for mean and coarse inputs
nrows = len(ds_Samples.sample_id) + 2 if plot_samples else 2
ncols = len(time_points)

variables = [
    '2m_temperature',
]
for variable in variables:
    im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                projection='PlateCarree',
                                rem_idx=[1, 2, 3],
                                hspace=0.3)
    da_samples = ds_Samples[variable]
    da_samples = gut.add_mean_along_dim(da_samples, dim_name='sample_id')
    offset = variable_dict[variable]['offset']
    for i, tp in enumerate(time_points):
        da_obs = ds_obs_bc[variable].sel(time=tp, method='nearest')
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
                                       vmin=variable_dict[variable]['vmin']+10,
                                       vmax=variable_dict[variable]['vmax'],
                                       #    centercolor='white',
                                       tick_step=5,
                                       plot_borders=True,)

    savepath = f'{plot_dir}/trajectories/{fine_res}_{variable}_{tu.tp2str(tp, h=False)}.png'
    gplt.save_fig(savepath=savepath)

# %%
# plot the average dayofyear mean
reload(tu)
timemean = 'dayofyear'
window = 1
variables = variable_dict.keys()
# short_time_range = ['2020-01-01', '2024-01-31']
tr = tu.get_time_range(ds_Samples)
for variable in variables:
    da_arr = []
    label_arr = []
    offset = variable_dict[variable]['offset']
    for ds_type, ds in ds_dict.items():
        ds = tu.get_time_range_data(ds, time_range=tr)
        if ds_type != 'Samples':
            da_arr.append(ds[variable])
            label_arr.append(ds_type)
        else:
            da_samples = ds[variable]
            da_samples = gut.add_mean_along_dim(da_samples, dim_name='sample_id')

            if plot_samples:
                for sid in da_samples.sample_id.values:
                    da_arr.append(da_samples.sel(sample_id=sid))
                    label_arr.append(f'Sample {sid}')
            else:
                da_arr.append(da_samples.sel(sample_id='mean'))
                label_arr.append('Sample Mean')

    data_arr = []
    time_arr = []
    for data in da_arr:
        ts_mean = data.mean(dim='lon').mean(dim='lat')
        # ts_mean = tu.rolling_timemean(ts_mean, window=window)
        yearly_ts = tu.compute_timemean(ts_mean,
                                        timemean=timemean,
                                        groupby=True,
                                        reset_time=True,)
        data_arr.append(yearly_ts[:365])
        time_arr.append(yearly_ts.time[:365])

    sd, ed = tu.get_time_range(ds, asstr=True)
    im = gplt.plot_2d(x=time_arr,
                      y=data_arr,
                      title=f"{gcm_str} ({sd} - {ed})",
                      label_arr=label_arr,
                      color_arr=colors,
                      xlabel="Day of Year",
                      ylabel=f"{variable_dict[variable]['label']}",
                      set_grid=True,
                      loc='outside',
                      )
    savepath = f'{plot_dir}/variables/{variable}_{timemean}_{sd}_{ed}.png'
    gplt.save_fig(savepath=savepath)

