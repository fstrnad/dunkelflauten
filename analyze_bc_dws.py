# %%
import sys
import cmethods as cm

import cf_utils as cfu
import geoutils.utils.met_utils as mut
import geoutils.geodata.solar_radiation as sr
import workaround_fsr as wf
import geoutils.utils.statistic_utils as sut
from scipy import stats
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import atlite as at
from importlib import reload
import geoutils.countries.countries as cnt
import geoutils.countries.capacities as cap
import geoutils.cutouts.prepare_cutout as pc
import os
import yaml
# %%
if os.getenv("HOME") == '/home/ludwig/fstrnad80':
    cmip6_dir = "/mnt/lustre/work/ludwig/shared_datasets/CMIP6/"
    data_dir = f'{cmip6_dir}/downscaling/'
    era5_dir = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/Europe"
    with open('./config_cluster.yaml', 'r') as file:
        config = yaml.safe_load(file)
    plot_dir = "/mnt/lustre/home/ludwig/fstrnad80/plots/dunkelflauten/downscaling_cmip6/"

else:
    plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_cmip6/"
    data_dir = "/home/strnad/data/CMIP6/downscaling/"
    cmip6_dir = "/home/strnad/data/CMIP6/"
    era5_dir = "/home/strnad/data/climate_data/Europe"
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

# %%
reload(of)
reload(gut)
reload(fut)
variables = ['tas', 'uas', 'vas', 'rsds']

country_name = 'Germany'
gs_dws = 1.0
fine_res = 0.25
N = 3
gcms = [
    'MPI-ESM1-2-HR',
    'GFDL-ESM4',
    'MIROC6',
    'IPSL-CM6A-LR',
    'CanESM5'
]

tr_historical = [('1980-01-01', '2015-01-01')]  # full range
tr_ssp = [('2020-01-01', '2100-01-01')]  # full range

ssps = ['historical', 'ssp245', 'ssp585']

# %%

gs_cmip = 0.25
gs_era5 = 0.25  # needs to be the same for CMIP and ERA5
gs_dws = 1.0  # downscaling input resolution for the DWS data
hourly_res = 6
N = 3
used_variables = ['uas',
                  'vas',
                  'tas',
                  'rsds',
                  ]

era5_files = []
for var_cmip in used_variables:
    variable = gut.cmip2era5_dict[var_cmip]
    gs_era5 = gs_cmip  # needs to be the same for CMIP and ERA5
    era5_file = f'{era5_dir}/{country_name}_nn/{gs_era5}/{variable}_{gs_era5}_{hourly_res}h.nc'
    if not fut.exist_file(era5_file):
        gut.myprint(f'File {era5_file} does not exist!')
        continue
    else:
        era5_files.append(era5_file)
        gut.myprint(f'Using file {era5_file} for {variable}')

# opens the ERA5 Ground truth data
ds_era5 = of.open_nc_file(era5_files)
ds_era5.load()  # load the data into memory
# %%
# Open the CMIP6 data for the historical period

ssp = 'ssp245'  # or 'ssp245', 'ssp585'
gcm = 'MPI-ESM1-2-HR'  # or 'GFDL-ESM4', 'MIROC6', 'IPSL-CM6A-LR', 'CanESM5'

start_date, end_date = tr_historical[0]
tr_str = f'{start_date}_{end_date}'
gcm_str = f'{gcm}_historical'  # historical ssp
folder_hist = f'{data_dir}/{gs_dws}/{gcm_str}/{tr_str}_N{N}/'
hist_file_name = f'samples_{gcm_str}_{tr_str}_{gs_era5}_log_False.nc'
filename_hist = f'{folder_hist}/{hist_file_name}'

ds_cmip_hist = of.open_nc_file(filename_hist).mean(dim='sample_id')
ds_cmip_hist.load()  # load the data into memory
tr = tr_historical if ssp == 'historical' else tr_ssp
start_date, end_date = tr[0]
tr_str = f'{start_date}_{end_date}'

gcm_str = f'{gcm}_{ssp}'
folder_cmip6 = f'{data_dir}/{gs_dws}/{gcm_str}/{tr_str}_N{N}/'
cmip6_file = f'samples_{gcm_str}_{tr_str}_{gs_era5}_log_False'
filename_cmip6 = f'{folder_cmip6}/{cmip6_file}.nc'

ds_cmip_ssp = of.open_nc_file(filename_cmip6)
ds_cmip_ssp = ds_cmip_ssp.mean(dim='sample_id')
ds_cmip_ssp.load()  # load the data into memory
file_bc = f'{folder_cmip6}/{cmip6_file}_bc.nc'
# %%
ds_cmip_ssp_bc = of.open_nc_file(file_bc)
ds_cmip_ssp_bc.load()  # load the data into memory

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

nrows = 2
im = gplt.create_multi_plot(nrows=nrows,
                            ncols=len(used_variables) // nrows,
                            figsize=(15, 10),
                            y_title=1.,
                            title=f'{gcm} {ssp}',
                            hspace=0.5)

variables = ['uas', 'vas', 'tas', 'rsds']

for idx, variable in enumerate(variables):

    variable = gut.cmip2era5_dict[variable]
    tr_distr = tu.get_time_range(ds_cmip_ssp_bc,
                                 m=False, d=False, asstr=True)
    # this_era5, _ = tu.equalize_time_points(
    #     ds_era5[variable], ds_cmip_hist[variable])
    this_era5 = ds_era5[variable]
    ds_dict = {
        f'CMIP6 historical':  ds_cmip_hist[variable],
        'ERA5': this_era5,
        f'CMIP6 {ssp}': ds_cmip_ssp[variable],
        f'CMIP6 {ssp} BC': ds_cmip_ssp_bc[variable],
    }

    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        plot_data = ds.values.flatten()
        offset = variable_dict[variable]['offset']
        gplt.plot_hist(plot_data + offset,
                       ax=im['ax'][idx],
                       title=f'{variable_dict[variable]['vname']}',
                       color=gplt.colors[i],
                       label=ds_type if idx % nrows != 0 else None,
                       nbins=100,
                       lw=3,
                       alpha=1,
                       ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                       ylabel='Density',
                       density=True,
                       set_yaxis=False,
                       loc='outside',
                       xlabel=variable_dict[variable]['label'],
                       xlim=(variable_dict[variable]['vmin'],
                             variable_dict[variable]['vmax']),
                       )

savepath = f'{plot_dir}/bc/qdm_{gcm_str}_{tr_distr}.png'
gplt.save_fig(savepath=savepath)
# %%