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
import sda_atmos.eval.sample_data as sda
import sda_atmos.eval.util as sda_utils
import sda_atmos.training.dataset as tds

from pathlib import Path
import json
from importlib import reload
import yaml

reload(sda)
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
N = 3

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
gcms = [
    # 'MPI-ESM1-2-HR',
    'GFDL-ESM4',
    # 'MIROC6',
    # 'HadGEM3-GC31-LL',
]

time_ranges = [
    ("2020-01-01", "2024-12-31"),
    ("2045-01-01", "2049-12-31"),
    ("2065-01-01", "2069-12-31"),
    ("2095-01-01", "2099-12-31"),
]

tr_historical = [
    ('1980-01-01', '1990-01-01'),
    ('1990-01-01', '2000-01-01'),
    ('2000-01-01', '2010-01-01'),
    ('2010-01-01', '2015-01-01')]

ssp_dict = {
    'ssp245': dict(
        label='SSP2-4.5',
        color='tab:blue'),
    'ssp585': dict(
        label='SSP5-8.5',
        color='tab:red'),
}

for gcm in gcms:
    for ssp in ssp_dict.keys():
        for start_date, end_date in time_ranges:

            time_str = f'{start_date}_{end_date}'
            gcm_str = f'{gcm}_{ssp}'
            sample_folder = f'{data_dir}/{gs_dws}/{gcm_str}/{time_str}_N{N}/'
            sample_path = fut.get_files_in_folder(sample_folder,
                                                  include_string=f'{fine_res}.nc')[0]
            ds_Samples = of.open_nc_file(sample_path)
            sd, ed = tu.get_time_range(ds_Samples)

            obs_path = f'{cmip6_dir}/{country_name}/{float(gs_dws)}/{gcm_str}_eval.nc'
            ds_obs = of.open_nc_file(obs_path)
            ds_obs = tu.get_time_range_data(ds_obs,
                                            time_range=(sd, ed))

            obs_path_bc = f'{cmip6_dir}/{country_name}/{float(gs_dws)}_bc/{gcm_str}_eval.nc'
            ds_obs_bc = of.open_nc_file(obs_path_bc)
            ds_obs_bc = tu.get_time_range_data(ds_obs_bc,
                                               time_range=(sd, ed))

            gs, _, _ = sput.get_grid_step(ds_Samples)
            variables = gut.get_vars(ds_Samples)
            tr = tu.tps2str(tu.get_time_range(ds_Samples),
                            h=False, full_str=True)

            ds_dict = {
                'Samples': ds_Samples.load(),
                'CMIP6 bc': ds_obs_bc.load(),
                'CMIP6': ds_obs.load(),
            }
            ssp_dict[ssp][time_str] = dict(
                sd=sd,
                ed=ed,
                variables=variables,
                ds_dict=ds_dict
            )

            # Compare distributions of Samples and ground truth
            reload(gplt)
            reload(tu)

            nrows = 2
            sd_str, ed_str = tu.get_time_range(
                ds_Samples, asstr=True, m=False, d=False, h=False)

            im = gplt.create_multi_plot(nrows=nrows, ncols=len(variables) // nrows,
                                        figsize=(15, 10),
                                        title=f'Distribution for {gcm} {ssp} ({sd_str}-{ed_str})',
                                        y_title=1.,
                                        hspace=0.5)

            colors = ['tab:red', 'grey', 'tab:blue']
            for idx, variable in enumerate(variables):
                for i, (ds_type, ds) in enumerate(ds_dict.items()):
                    ds = tu.compute_timemean(
                        ds, timemean='day') if variable == 'surface_solar_radiation_downwards' else ds
                    plot_data = []
                    offset = variable_dict[variable]['offset']
                    if ds_type == 'Samples':
                        for sample_id in ds.sample_id.values:
                            sdata = ds.sel(sample_id=sample_id)[
                                variable].values.flatten() + offset
                            plot_data.append(sdata)
                    else:
                        plot_data.append(
                            ds[variable].values.flatten() + offset)

                    this_im = gplt.plot_hist(plot_data,
                                             ax=im['ax'][idx],
                                             title=variable_dict[variable]['vname'],
                                             color=colors[i],
                                             label=ds_type,
                                             nbins=100,
                                             lw=1 if ds_type == 'Samples' else 2,
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
                                          alpha=0.2,
                                          )

            savepath = f'{plot_dir}/distributions/{gcm_str}_{tr}.png'
            gplt.save_fig(savepath=savepath)


# %%
# Compare distributions of SAT for different ssps and time ranges
reload(gplt)

ssp = 'ssp585'
ds_type = 'CMIP6 bc'
variable = '2m_temperature'
gcm_str = f'{gcm}_{ssp}'
im = gplt.create_multi_plot(nrows=1, ncols=1,
                            figsize=(10, 5),
                            )
for idx, tr in enumerate(time_ranges):
    start_date, end_date = tr
    time_str = f'{start_date}_{end_date}'
    sdata_dict = ssp_dict[ssp][time_str]
    sdata = sdata_dict['ds_dict'][ds_type][variable]
    if ds_type == 'Samples':
        sdata = sdata.mean(dim='sample_id')
        sdata = tu.compute_timemean(ds=sdata, timemean='day')
    sdata = sdata.values.flatten(
    ) + variable_dict[variable]['offset']
    this_im = gplt.plot_hist(sdata,
                             ax=im['ax'][0],
                             title=f'{gcm} {ssp}: {variable_dict[variable]['vname']}',
                             color=gplt.colors[idx],
                             label=tr,
                             nbins=100,
                            #  lw=1 if ds_type == 'Samples' else 2,
                             alpha=0.8 if ds_type == 'Samples' else 1,
                             set_yaxis=False,
                             #  ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                             ylabel='Density',
                             density=True,
                             xlabel=variable_dict[variable]['label'],
                             xlim=(variable_dict[variable]['vmin'],
                                   variable_dict[variable]['vmax']),
                             loc='outside',
                             )

savepath = f'{plot_dir}/distributions/{gcm_str}_{ds_type}_{variable}.png'
gplt.save_fig(savepath=savepath)

# %%
ssp = 'ssp585'
ds_type = 'CMIP6 bc'
ds_type = 'Samples'
variable = '2m_temperature'
gcm_str = f'{gcm}_{ssp}'
im = gplt.create_multi_plot(nrows=1, ncols=1,
                            figsize=(10, 5),
                            )

single_loc = (10.0, 50.0)
for idx, tr in enumerate(time_ranges):
    start_date, end_date = tr
    time_str = f'{start_date}_{end_date}'
    sdata_dict = ssp_dict[ssp][time_str]
    ds = sdata_dict['ds_dict'][ds_type]
    ds = ds.sel(lon=single_loc[0], lat=single_loc[1], method='nearest')
    sdata = ds[variable]
    if ds_type == 'Samples':
        sdata = sdata.mean(dim='sample_id')
        # sdata = tu.compute_timemean(ds=sdata, timemean='day')
    sdata = sdata.values.flatten(
    ) + variable_dict[variable]['offset']
    this_im = gplt.plot_hist(sdata,
                             ax=im['ax'][0],
                             title=f'{gcm} {ssp}: {variable_dict[variable]['vname']}',
                             color=gplt.colors[idx],
                             label=tr,
                             nbins=100,
                             #  ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                             ylabel='Density',
                             density=True,
                             xlabel=variable_dict[variable]['label'],
                             xlim=(variable_dict[variable]['vmin'],
                                   variable_dict[variable]['vmax']),
                             loc='outside',
                             )

# %%
# single time series
reload(gplt)

ds_type = 'CMIP6 bc'
im = gplt.create_multi_plot(nrows=1, ncols=1,
                            figsize=(10, 5),
                            )

single_loc = (10.0, 50.0)
for idx, tr in enumerate(time_ranges):
    start_date, end_date = tr
    time_str = f'{start_date}_{end_date}'
    sdata_dict = ssp_dict[ssp][time_str]
    ds = sdata_dict['ds_dict'][ds_type]
    ds = ds.sel(lon=single_loc[0], lat=single_loc[1], method='nearest')
    sdata = ds[variable]
    if ds_type == 'Samples':
        sdata = sdata.mean(dim='sample_id')
        # sdata = tu.compute_timemean(ds=sdata, timemean='day')
    sdata = sdata.values.flatten()[:100]
    this_im = gplt.plot_2d(sdata,
                           ax=im['ax'][0],
                           title=f'{gcm} {ssp}: {variable_dict[variable]['vname']}',
                           color=gplt.colors[idx],
                           label=tr,
                           nbins=100,
                        #    lw=1 if ds_type == 'Samples' else 2,
                           #  ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                           loc='outside',
                           )

# %%
