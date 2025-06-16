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
reload(of)
reload(gut)
reload(fut)
variables = ['tas', 'uas', 'vas', 'rsds']
europe_dir = config['europe_dir']
time = 'day'

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

tr_historical = [
    ('1980-01-01', '1990-01-01'),
    ('1990-01-01', '2000-01-01'),
    ('2000-01-01', '2010-01-01'),
    ('2010-01-01', '2015-01-01')]

tr_ssp = [
    ('2020-01-01', '2030-01-01'),
    ('2030-01-01', '2040-01-01'),
    ('2040-01-01', '2050-01-01'),
    ('2050-01-01', '2060-01-01'),
    ('2060-01-01', '2070-01-01'),
    ('2070-01-01', '2080-01-01'),
    ('2080-01-01', '2090-01-01'),
    ('2090-01-01', '2100-01-01')
]
ssps = ['historical', 'ssp245', 'ssp585']

gcm_ssp_cf_dict = {}
for gcm in gcms:
    ssp_cf_dict = {}
    for ssp in ssps:
        gcm_str = f'{gcm}_{ssp}'
        cf_tr_dict = {}
        time_ranges = tr_ssp if ssp != 'historical' else tr_historical
        for tr_idx, (start_date, end_date) in enumerate(time_ranges):
            start_date, end_date = time_ranges[tr_idx]
            tr_str = f'{start_date}_{end_date}'

            reload(cfu)
            savepath_dict = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{fine_res}_{tr_str}.npy'
            if fut.exist_file(savepath_dict):
                cf_dict_cmip = fut.load_np_dict(savepath_dict)
                cf_tr_dict[tr_str] = cf_dict_cmip
            else:
                print(f'{gcm} {ssp} {tr_str} not found')
        ssp_cf_dict[ssp] = cf_tr_dict
    gcm_ssp_cf_dict[gcm] = ssp_cf_dict

# %%
# For comparison with ERA5
gs_era5 = 0.25
cf_dict_path_era5 = f'{config['data_dir']}/{country_name}/era5/cf_dict_{gs_era5}.npy'

cf_dict_era5 = fut.load_np_dict(cf_dict_path_era5)
# %%
# Risk per year
reload(tu)
gcm = 'MPI-ESM1-2-HR'
ssp_cf_dict = gcm_ssp_cf_dict['MPI-ESM1-2-HR']

df_type = 'all'
for ssp, cf_tr_dict in ssp_cf_dict.items():
    gcm_str = f'{gcm}_{ssp}'
    for tr_str, cf_dict_cmip in cf_tr_dict.items():
        ts_df = cf_dict_cmip[df_type]['ts']
        sd, ed = tu.get_time_range(ts_df, asstr=True)
        hourly_res = tu.get_frequency_resolution_hours(ts_df)
        num_years = tu.count_unique_years(ts_df)
        num_hours = 48
        window = int(num_hours / hourly_res)  # 8*6 = 48 hours
        ts_df_mean = tu.rolling_timemean(ts_df, window=window)

        threshold = 0.06
        ts_evs = tu.compute_evs(ts_df_mean,
                                threshold=threshold,
                                threshold_type='lower',
                                get_mask=False)
        tps_dfl_raw = ts_df_mean.where(ts_evs == 1).dropna('time')
        min_consecutive = 1
        tps_dfl = tu.find_consecutive_ones(ts_evs,
                                           min_consecutive=min_consecutive)
        tps_dfl = tu.get_sel_tps_ds(ts_df_mean, tps_dfl)
        dfl_per_year = tu.count_time_points(tps_dfl)

        dfl_per_month = tu.count_time_points(tps_dfl, counter='month')
        dfl_per_month = sut.relative_share(dfl_per_month)
        dfl_per_week = tu.count_time_points(tps_dfl, counter='week')

        reload(gplt)
        im = gplt.create_multi_plot(nrows=2, ncols=2,
                                    figsize=(22, 10),
                                    hspace=0.4,
                                    wspace=0.1,
                                    )
        gplt.plot_2d(x=[ts_df_mean.time.data, tps_dfl.time.data],
                     y=[ts_df_mean, tps_dfl],
                     ax=im['ax'][0],
                     title=f'Identify Dunkelflauten {country_name}',
                     label_arr=['CF time series', 'Selected Dunkelflauten   '],
                     ls_arr=['-', ''],
                     mk_arr=['', 'x'],
                     #  xlabel='Time',
                     ylabel='CF [a.u]',
                     ylim=(0, 0.6),
                     rot=90,
                     )
        gplt.plot_hline(ax=im['ax'][0], y=threshold, color='r', ls='--',
                        label=f'Threshold = {threshold}',
                        lw=1)

        gplt.plot_2d(x=dfl_per_month.time.data,
                     y=dfl_per_month.data,
                     ax=im['ax'][1],
                     plot_type='bar',
                     title=f'Dunkelflaute per month {country_name}',
                     label_arr=['Dunkelflaute per month'],
                     ls_arr=['-'],
                     mk_arr=[''],
                     #  xlabel='Year',
                     ylabel='Rel. Frac. Dunkelflaute events',
                     rot=90,
                     #  ylim=(0, 26),
                     )

        gplt.plot_2d(x=tu.get_year(dfl_per_year.time),
                     y=dfl_per_year.data,
                     ax=im['ax'][2],
                     #  plot_type='bar',
                     title=f'Dunkelflaute per year {country_name}',
                     label_arr=['Dunkelflaute per year'],
                     ls_arr=['-'],
                     color='tab:blue',
                     mk_arr=[''],
                     #  xlabel='Year',
                     ylabel='Number of Dunkelflaute events',
                     rot=90,
                     ylim=(0, 9),
                     )

        gplt.plot_2d(x=dfl_per_week.time.data,
                     ax=im['ax'][3],
                     y=dfl_per_week.data,
                     plot_type='bar',
                     title=f'Dunkelflaute per week of year {country_name}',
                     label_arr=['Dunkelflaute per week'],
                     ls_arr=['-'],
                     mk_arr=[''],
                     #  xlabel='Year',
                     ylabel='Number of Dunkelflaute events',
                     rot=90,
                     ylim=(0, 9)
                     )

        savepath = f'{config['plot_dir']}/dunkelflauten_cmip6/ts_{df_type}_df_{gcm_str}_{gs_dws}_{sd}_{ed}.png'
        gplt.save_fig(savepath)
# %%
# Local dunkelflauten Germany
reload(gplt)
reload(cfu)
reload(tu)
gcm = 'MPI-ESM1-2-HR'
ssp_cf_dict = gcm_ssp_cf_dict['MPI-ESM1-2-HR']

for ssp, cf_tr_dict in ssp_cf_dict.items():
    gcm_str = f'{gcm}_{ssp}'
    im_df = gplt.create_multi_plot(
        nrows=1, ncols=len(cf_tr_dict),
        projection='PlateCarree',
    )

    for idx, (tr_str, cf_dict_cmip) in enumerate(cf_tr_dict.items()):
        cf_onwind_solar = cfu.combined_cf_maps(cf_dict_cmip,
                                               sources=['onwind', 'solar'],)
        cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
        num_hours = 48
        hourly_res = 6
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
        sd, ed = tu.get_time_range(df_local_onwind, asstr=True)
        gplt.plot_map(num_dfs_cell/num_years,
                      ax=im_df['ax'][idx],
                      plot_borders=True,
                      significance_mask=xr.where(mask, 0, 1),
                      vmin=0,
                      vmax=25,
                      label='No. of Dunkelflauten / Year',
                      title=f'{sd} - {ed}',
                      vertical_title=f'{gcm} {ssp}' if idx == 0 else None,
                      cmap='Reds',
                      leftcolor='white',
                      levels=10,
                      )

    savepath = f"{config['plot_dir']}/local_risks/CMIP6/df_local_2020_2100_{gcm_str}_{gs_dws}.png"
    gplt.save_fig(savepath)
# %%
# compare to historical period
reload(gplt)

avs_dfs_per_year = 0

for ssp, cf_tr_dict in ssp_cf_dict.items():
    gcm_str = f'{gcm}_{ssp}'
    if ssp != 'historical':
        im_df = gplt.create_multi_plot(
            nrows=1, ncols=len(cf_tr_dict),
            projection='PlateCarree',
        )
    for idx, (tr_str, cf_dict_cmip) in enumerate(cf_tr_dict.items()):
        cf_onwind_solar = cfu.combined_cf_maps(cf_dict_cmip,
                                               sources=['onwind', 'solar'],)
        cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
        num_hours = 48
        hourly_res = 6
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
        sd, ed = tu.get_time_range(df_local_onwind, asstr=True)
        dfs_per_year = num_dfs_cell / num_years

        if ssp == 'historical':
            avs_dfs_per_year += dfs_per_year
        else:
            gplt.plot_map(dfs_per_year - avs_dfs_per_year/len(tr_historical),
                          ax=im_df['ax'][idx],
                          plot_borders=True,
                          significance_mask=xr.where(mask, 0, 1),
                          vmin=-5,
                          vmax=5,
                          label='No. of Dunkelflauten / Year - Historical',
                          title=f'{sd} - {ed}',
                          vertical_title=f'{gcm} {ssp}' if idx == 0 else None,
                          cmap='cmo.balance',
                          centercolor='white',
                          levels=10,
                          )

    savepath = f"{config['plot_dir']}/local_risks/CMIP6/df_local_compare_historical_{gcm_str}_{gs_dws}.png"
    gplt.save_fig(savepath)

# %%