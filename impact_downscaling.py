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
savepath_dict_coarse = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{coarse_res}_{tr_str}.npy'
savepath_dict_daily = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_daily_{fine_res}_{tr_str}.npy'

cf_era5_fine_gt = fut.load_np_dict(savepath_dict_fine_gt)
cf_era5_fine = fut.load_np_dict(savepath_dict_fine)
cf_era5_coarse = fut.load_np_dict(savepath_dict_coarse)
cf_era5_daily = fut.load_np_dict(savepath_dict_daily)
cf_dicts = {
    f'GT {fine_res}': cf_era5_fine_gt,
    fine_res: cf_era5_fine,
    coarse_res: cf_era5_coarse,
    'daily': cf_era5_daily
}

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


for idx, (res, cf_dict_cmip) in enumerate(cf_dicts.items()):
    hourly_res = tu.get_frequency_resolution_hours(
        cf_dict_cmip['solar']['cf_ts'])
    res = res if res != 'daily' else '0.25'
    title = f'{res}°, {hourly_res}h resolution'

    for s, sname in enumerate(sources):
        sd, ed = tu.get_time_range(
            cf_dict_cmip[sname]['cf_ts'], asstr=True, m=False, d=False)

        cap_fac = cf_dict_cmip[sname]['cf']
        gplt.plot_map(cap_fac,
                      ax=im_cfs['ax'][s*ncols + idx],
                      title=title if s == 0 else None,
                      vertical_title=f'{sname} capacity factor \n{sd} - {ed}' if idx == 0 else None,
                      y_title=1.2,
                      vmin=0.1,
                      cmap='cmo.thermal',
                      label='Capacity Factor [a.u.]',
                      vmax=.15 if sname == 'solar' else 0.35,
                      plot_borders=True)

    cf_onwind_solar = cfu.combined_cf_maps(cf_dict_cmip,
                                           sources=['onwind', 'solar'],)
    cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
    num_hours = 48

    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    cf_ts_mean = tu.rolling_timemean(cf_onwind_solar, window=window)
    threshold = 0.02
    df_local_onwind, mask = tu.compute_evs(cf_ts_mean,
                                           threshold=threshold,
                                           threshold_type='lower',
                                           #    max_rel_share=0.02,
                                           get_mask=True)

    num_years = tu.count_unique_years(df_local_onwind)
    num_dfs_cell = df_local_onwind.sum(dim='time')
    gplt.plot_map(num_dfs_cell/num_years,
                  ax=im_cfs['ax'][ncols*len(sources) + idx],
                  plot_borders=True,
                  significance_mask=xr.where(mask, 0, 1),
                  vmin=3,
                  vmax=18,
                  label='No. of Dunkelflauten / Year',
                #   title=title,
                  vertical_title=f'No. of Dunkelflaute events \n{sd} - {ed}' if idx == 0 else None,
                  cmap='cmo.amp',
                  leftcolor='white',
                  levels=15,
                  tick_step=5
                  )

savepath_cfs = f"{config['plot_dir']}/impact_downscaling/compare_res_ERA5_{tr_str}_{res}.png"

gplt.save_fig(savepath_cfs, fig=im_cfs['fig'])
