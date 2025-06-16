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

import yaml
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# %%
plot_dir = "/home/strnad/plots/dunkelflauten/downscaling_cmip6/"
data_dir = "/home/strnad/data/CMIP6/downscaling/"
cmip6_dir = "/home/strnad/data/CMIP6/"
era5_dir = "/home/strnad/data/climate_data/Europe"
country_name = 'Germany'
# %%
fine_res = 0.25
coarse_res = 1.0

gcm = 'MPI-ESM1-2-HR'
ssp = 'ssp585'
gcm_str = f'{gcm}_{ssp}'

time_ranges = [
    ('2020-01-01', '2029-12-31'),
    ('2050-01-01', '2059-12-31'),
    ("2070-01-01", "2079-12-31"),
    ('2090-01-01', '2099-12-31')
]

tr_idx = 0
start_date, end_date = time_ranges[tr_idx]
tr_str = f'{start_date}_{end_date}'
# %%
savepath_dict_fine = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{fine_res}_{tr_str}.npy'
savepath_dict_coarse = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{coarse_res}_{tr_str}.npy'

cf_dict_fine = fut.load_np_dict(savepath_dict_fine)
cf_dict_coarse = fut.load_np_dict(savepath_dict_coarse)
cf_dicts = {
    fine_res: cf_dict_fine,
    coarse_res: cf_dict_coarse
}

# %%
# Local dunkelflauten Germany
reload(gplt)
reload(cfu)
reload(tu)
sources = ['onwind', 'solar', ]
n_cols = len(sources)

im_cfs = gplt.create_multi_plot(
    nrows=2, ncols=n_cols,
    hspace=0.4,
    projection='PlateCarree')

im_dfs = gplt.create_multi_plot(
    nrows=1, ncols=2,
    projection='PlateCarree')

for idx, (res, cf_dict_cmip) in enumerate(cf_dicts.items()):
    hourly_res = tu.get_frequency_resolution_hours(
        cf_dict_cmip['solar']['cf_ts'])
    title = f'{res}°, {hourly_res}h resolution'

    for s, sname in enumerate(sources):
        sd, ed = tu.get_time_range(cf_dict_cmip[sname]['cf_ts'], asstr=True)

        cap_fac = cf_dict_cmip[sname]['cf']
        gplt.plot_map(cap_fac,
                      ax=im_cfs['ax'][s*n_cols + idx],
                      title=title if s == 0 else None,
                      vertical_title=f'{sname} \n{sd} - {ed}' if idx == 0 else None,
                      vmin=0,
                      cmap='cmo.solar',
                      label='Capacity Factor [a.u.]',
                      vmax=.2 if sname == 'solar' else 0.4,
                      plot_borders=True)

    savepath_cfs = f"{config['plot_dir']}/capacities_cmip6/compare_res_{gcm_str}_{tr_str}_{res}.png"

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
                  ax=im_dfs['ax'][idx],
                  plot_borders=True,
                  significance_mask=xr.where(mask, 0, 1),
                  vmin=0,
                  #   vmax=25,
                  label='No. of Dunkelflauten / Year',
                  title=title,
                  vertical_title=f'{sd} - {ed}' if idx == 0 else None,
                  cmap='Reds',
                  leftcolor='white',
                  levels=10,
                  )

    savepath_dfs = f"{config['plot_dir']}/local_risks/CMIP6/compare_res_df_local_{gcm_str}_{sd}_{ed}.png"

gplt.save_fig(savepath_dfs, fig=im_dfs['fig'])
gplt.save_fig(savepath_cfs, fig=im_cfs['fig'])
# %%
