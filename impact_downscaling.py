# %%
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
    era5_dir = "/mnt/lustre/work/ludwig/shared_datasets/climate_data/Europe"
    with open('./config_cluster.yaml', 'r') as file:
        config = yaml.safe_load(file)
else:
    plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_cmip6/"
    data_dir = "/home/strnad/data/CMIP6/downscaling/"
    cmip6_dir = "/home/strnad/data/CMIP6/"
    era5_dir = "/home/strnad/data/climate_data/Europe"
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
# %%
fine_res = 0.25
coarse_res = 1.0
country_name = 'Germany'
gcm = 'MPI-ESM1-2-HR'
ssp = 'ssp585'
gcm_str = f'{gcm}_{ssp}'

time_ranges = [
    ('1980-01-01', '2025-01-01')
]

time_ranges = [
    ('2023-01-01', '2025-01-01')
]

tr_idx = 0
start_date, end_date = time_ranges[tr_idx]
tr_str = f'{start_date}_{end_date}'


# %%
savepath_dict_fine_gt = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{fine_res}_{tr_str}.npy'
savepath_dict_fine = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{fine_res}_{tr_str}_dws.npy'
savepath_dict_fine_bc = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{fine_res}_{tr_str}_dws_bc.npy'
savepath_dict_coarse = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{coarse_res}_{tr_str}.npy'
savepath_dict_daily = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_daily_{fine_res}_{tr_str}.npy'

cf_era5_fine_gt = fut.load_np_dict(savepath_dict_fine_gt)
cf_era5_fine = fut.load_np_dict(savepath_dict_fine)
cf_era5_fine_bc = fut.load_np_dict(savepath_dict_fine_bc)
cf_era5_coarse = fut.load_np_dict(savepath_dict_coarse)
cf_era5_daily = fut.load_np_dict(savepath_dict_daily)
cf_dicts = {
    f'GT {fine_res}': cf_era5_fine_gt,
    f'Coarse {coarse_res}': cf_era5_coarse,
    'daily': cf_era5_daily,
    f'DS {fine_res}': cf_era5_fine,
    f'DS BC {fine_res}': cf_era5_fine_bc,
    }

def get_cf_mask(cf_dict, threshold=0.1):
    """
    Get a mask for the capacity factor data based on a threshold.
    """
    cf_onwind_solar = cfu.combined_cf_maps(cf_dict,
                                           sources=['onwind', 'solar'],)
    cf_onwind_solar = sput.rename_dims(cf_onwind_solar).mean(dim='time')

    mask = xr.where(cf_onwind_solar < threshold, 0, 1)

    return mask

# %%
cf_mask = get_cf_mask(cf_dict=cf_era5_fine_gt, threshold=0.1)
cf_mask_coarse = get_cf_mask(cf_dict=cf_era5_coarse, threshold=0.1)

# %%
# Local dunkelflauten Germany
reload(gplt)
reload(cfu)
reload(tu)
sources = ['offwind', 'solar', ]
ncols = len(cf_dicts)
nrows = len(sources) + 1
im_cfs = gplt.create_multi_plot(
    nrows=nrows, ncols=ncols,
    hspace=0.4,
    projection='PlateCarree')

threshold = 0.02

for idx, (res, cf_dict_cmip) in enumerate(cf_dicts.items()):
    hourly_res = tu.get_frequency_resolution_hours(
        cf_dict_cmip['solar']['cf_ts'])
    res = res if res != 'daily' else '0.25'
    title = f'{res}°, {hourly_res}h resolution'
    cf_onwind_solar = cfu.combined_cf_maps(cf_dict_cmip,
                                           sources=['onwind', 'solar'],)
    cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
    gs, _ ,_ = sput.get_grid_step(cf_onwind_solar)
    mask = cf_mask if gs == fine_res else cf_mask_coarse
    for s, sname in enumerate(sources):
        sd, ed = tu.get_time_range(
            cf_dict_cmip[sname]['cf_ts'], asstr=True, m=False, d=False)

        cap_fac = cf_dict_cmip[sname]['cf']

        gplt.plot_map(cap_fac,
                      ax=im_cfs['ax'][s*ncols + idx],
                      title=title if s == 0 else None,
                      vertical_title=f'{sname} capacity factor \n{sd} - {ed}' if idx == 0 else None,
                      y_title=1.2,
                      mask=mask,
                      cmap='cmo.thermal',
                      levels=25,
                      tick_step=5,
                      label='Capacity Factor [a.u.]',
                      vmin=0.08 if  sname == 'solar' else 0.0,
                      vmax=.14 if sname == 'solar' else 0.3,
                      plot_borders=True)


    num_hours = 48

    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    cf_ts_mean = tu.rolling_timemean(cf_onwind_solar, window=window)
    df_local_onwind, _ = tu.compute_evs(cf_ts_mean,
                                           threshold=threshold,
                                           threshold_type='lower',
                                           get_mask=True)

    num_years = tu.count_unique_years(df_local_onwind)
    num_dfs_cell = df_local_onwind.sum(dim='time')
    gplt.plot_map(num_dfs_cell/num_years,
                  ax=im_cfs['ax'][ncols*len(sources) + idx],
                  plot_borders=True,
                  vmin=3,
                  vmax=18,
                  label='No. of Dunkelflauten / Year',
                #   title=title,
                  vertical_title=f'No. of Dunkelflaute events \n{sd} - {ed}' if idx == 0 else None,
                  cmap='cmo.amp',
                  leftcolor='white',
                  mask=mask,
                  levels=15,
                  tick_step=5
                  )

savepath_cfs = f"{config['plot_dir']}/impact_downscaling/compare_res_ERA5_{tr_str}.png"

gplt.save_fig(savepath_cfs, fig=im_cfs['fig'])
# %%