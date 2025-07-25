# %%
from copy import deepcopy as dcp
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


tr_historical = [('1980-01-01', '2015-01-01')]  # full range
tr_ssp = [('2020-01-01', '2100-01-01')]

ssps = [
    'historical',
    'ssp245',
    'ssp585'
]
use_bc = True  # use bias correction for downscaling
gcm_ssp_cf_dict = {}
for gcm in gcms:
    ssp_cf_dict = {}
    for ssp in ssps:
        gcm_str = f'{gcm}_{ssp}'
        time_ranges = tr_ssp if ssp != 'historical' else tr_historical
        start_date, end_date = time_ranges[0]
        tr_str = f'{start_date}_{end_date}'

        reload(cfu)
        if use_bc:
            savepath_dict = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{fine_res}_{tr_str}_bc.npy'
        else:
            savepath_dict = f'{config['data_dir']}/{country_name}/CMIP6/cf/cf_dict_{gcm_str}_{fine_res}_{tr_str}.npy'
        if fut.exist_file(savepath_dict):
            cf_dict_cmip = fut.load_np_dict(savepath_dict)
        else:
            print(f'{gcm} {ssp} {tr_str} not found')
        ssp_cf_dict[ssp] = cf_dict_cmip
    gcm_ssp_cf_dict[gcm] = ssp_cf_dict

# %%
# For comparison with ERA5
gs_era5 = 0.25
tr_str = '1980-01-01_2025-01-01'
cf_dict_path_era5 = f'{config['data_dir']}/{country_name}/ERA5/cf/cf_dict_{gs_era5}_{tr_str}.npy'
cf_dict_era5 = fut.load_np_dict(cf_dict_path_era5)

# %%  Compute all local dunkelflaute events
reload(tu)


def local_dfs(cf_dict, num_hours=48, hourly_res=6, threshold=0.02):
    cf_onwind_solar = cfu.combined_cf_maps(cf_dict,
                                           sources=['onwind', 'solar'],)
    cf_onwind_solar = sput.rename_dims(cf_onwind_solar)

    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    cf_ts_mean = tu.rolling_timemean(cf_onwind_solar, window=window)
    df_local_onwind, _ = tu.compute_evs(cf_ts_mean,
                                        threshold=threshold,
                                        threshold_type='lower',
                                        #    max_rel_share=0.02,
                                        get_mask=True)

    return df_local_onwind


def local_dfs_per_year(df_local_onwind):
    num_years = tu.count_unique_years(df_local_onwind)
    num_dfs_cell = df_local_onwind.sum(dim='time')  # done later
    dfs_per_year = num_dfs_cell / num_years

    return dfs_per_year


def country_df_ts(cf_dict, df_type='all',
                  min_consecutive=1,
                  num_hours=48, hourly_res=6, threshold=0.06):
    ts_df = cf_dict[df_type]['ts']
    hourly_res = tu.get_frequency_resolution_hours(ts_df)
    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    ts_df_mean = tu.rolling_timemean(ts_df, window=window)

    ts_evs = tu.compute_evs(ts_df_mean,
                            threshold=threshold,
                            threshold_type='lower',
                            get_mask=False)

    tps_dfl = tu.find_consecutive_ones(ts_evs,
                                       min_consecutive=min_consecutive)
    tps_dfl = tu.get_sel_tps_ds(ts_df_mean, tps_dfl)
    dfl_per_year = tu.count_time_points(tps_dfl,
                                        counter='year')

    return dfl_per_year


# %%
df_dict_local = {}
df_dict_country = {}
num_hours = 48
hourly_res = 6
threshold = 0.02

df_dict_local['ERA5'] = local_dfs(cf_dict_era5,
                                  num_hours=num_hours,
                                  hourly_res=hourly_res,
                                  threshold=threshold)

for gcm in gcms:
    ssp_df_dict_local = {}
    for ssp in ssps:
        gcm_str = f'{gcm}_{ssp}'
        df_tr_dict_local = {}
        time_ranges = tr_ssp if ssp != 'historical' else tr_historical
        print(f'Processing {gcm} {ssp}')
        time_ranges = tr_ssp if ssp != 'historical' else tr_historical
        start_date, end_date = time_ranges[0]
        tr_str = f'{start_date}_{end_date}'
        cf_dict_tr = gcm_ssp_cf_dict[gcm][ssp]
        dfs_per_year_local = local_dfs(cf_dict_tr,
                                       num_hours=num_hours,
                                       hourly_res=hourly_res,
                                       threshold=threshold)

        ssp_df_dict_local[ssp] = dfs_per_year_local
    df_dict_local[gcm] = ssp_df_dict_local
# %%
threshold_country = 0.06
df_dict_country['ERA5'] = country_df_ts(
    cf_dict_era5,
    num_hours=num_hours,
    hourly_res=hourly_res,
    threshold=threshold_country
)

for gcm in gcms:
    ssp_df_dict_country = {}
    for ssp in ssps:
        df_tr_dict_country = {}
        time_ranges = tr_ssp if ssp != 'historical' else tr_historical
        print(f'Processing {gcm} {ssp}')
        time_ranges = tr_ssp if ssp != 'historical' else tr_historical
        start_date, end_date = time_ranges[0]
        tr_str = f'{start_date}_{end_date}'
        cf_dict_tr = gcm_ssp_cf_dict[gcm][ssp]
        if cf_dict_tr:
            dfs_per_year_country = country_df_ts(
                cf_dict_tr,
                num_hours=num_hours,
                hourly_res=hourly_res,
                threshold=threshold_country
            )
            ssp_df_dict_country[ssp] = dfs_per_year_country
    df_dict_country[gcm] = ssp_df_dict_country
# %%
# Risk per year
reload(tu)
gcm = 'MPI-ESM1-2-HR'
ssp_cf_dict = gcm_ssp_cf_dict['MPI-ESM1-2-HR']
df_type = 'all'

for ssp, cf_dict_cmip in ssp_cf_dict.items():
    gcm_str = f'{gcm}_{ssp}'

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

    savepath = f'{config['plot_dir']}/dunkelflauten_cmip6/ts_df_{gcm_str}_{gs_dws}_{sd}_{ed}.png'
    gplt.save_fig(savepath)
# %%
# Time Series for all different GCMS:
reload(sut)
reload(gplt)
ssp = 'ssp245'
im = gplt.create_multi_plot(nrows=1, ncols=2,
                            figsize=(15, 5))
for idx, ssp in enumerate(['ssp245', 'ssp585']):
    for idx_gcm, gcm in enumerate(gcms):

        dfl_per_year = df_dict_country[gcm][ssp]
        fit_vals, slope, p = sut.linear_regression_xarray(dfl_per_year)
        slope *= 365*24
        gplt.plot_2d(x=tu.get_year(dfl_per_year.time),
                     y=[dfl_per_year.data, fit_vals],
                     ax=im['ax'][idx],
                     #  plot_type='bar',
                     title=ssp,
                     label=[None,
                            f'{gcm} (sl {slope:.2f})'],
                     ls_arr=['dotted', '-'],
                     color=gplt.colors[idx_gcm+1],
                     mk_arr=['.', ''],
                     lw_arr=[0.5, 2],
                     alpha_arr=[0.5, 1],
                     loc='under',
                     #  xlabel='Year',
                     ylabel='Number of Dunkelflaute events',
                     rot=90,
                     #  ylim=(0, 9),
                     )
savepath = f'{config['plot_dir']}/dunkelflauten_cmip6/ts_df_year_reg.png'
gplt.save_fig(savepath)

# %%
# Compare historical ssps vs ERA5
ssp = 'historical'
im = gplt.create_multi_plot(nrows=3, ncols=1,
                            figsize=(10, 13),
                            hspace=0.6)
for idx, ssp in enumerate(ssps):
    for idx_gcm, gcm in enumerate(['ERA5'] + gcms):
        if gcm != 'ERA5':
            dfl_per_year = df_dict_country[gcm][ssp]
        else:
            if ssp != 'historical':
                continue
            dfl_per_year_era5, _ = tu.equalize_time_points(
                df_dict_country[gcm],
                df_dict_country['MPI-ESM1-2-HR'][ssp]
            )
            tr = tu.get_time_range(dfl_per_year_era5, asstr=True)
            dfl_per_year = dfl_per_year_era5

        fit_vals, slope, p = sut.linear_regression_xarray(dfl_per_year)
        slope *= 365
        ls = '--' if gcm == 'ERA5' else '-'
        lw = 3 if gcm == 'ERA5' else 2
        gplt.plot_2d(x=tu.get_year(dfl_per_year.time),
                     y=[dfl_per_year.data, fit_vals],
                     ax=im['ax'][idx],
                     #  plot_type='bar',
                     title=ssp,
                     label=[None,
                            f'{gcm} (r={slope:.3f}, p={p:.3f})'],
                     ls_arr=[ls, ls],
                     color=gplt.colors[idx_gcm],
                     mk_arr=['x', ''],
                     lw_arr=[0.5, lw],
                     alpha_arr=[0.5, 1],
                     ncol_legend=1,
                     loc='outside',
                     #  xlabel='Year',
                     ylabel='No. of Dunkelflaute events',
                     rot=45,
                     #  ylim=(0, 9),
                     #  box_loc=(-0.05, -0.2)
                     )
savepath = f'{config['plot_dir']}/dunkelflauten_cmip6/ts_df_year_reg_all_ssps.png'
gplt.save_fig(savepath)

# %%
# Local dunkelflauten Germany
reload(gplt)
reload(cfu)
reload(tu)
gcm = 'CanESM5'
ssp_df_dict_local = df_dict_local[gcm]

im_df = gplt.create_multi_plot(
    nrows=1, ncols=len(ssps),
    projection='PlateCarree',
)
for idx, ssp in enumerate(ssps):
    dfs_per_year = local_dfs_per_year(df_dict_local[gcm][ssp])
    sd, ed = tu.get_time_range(
        gcm_ssp_cf_dict[gcm][ssp]['all']['ts'], asstr=True, m=False, d=False)
    gplt.plot_map(dfs_per_year,
                  ax=im_df['ax'][idx],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=0,
                  vmax=25,
                  label='No. of Dunkelflauten / Year',
                  title=f'{ssp} ({sd}-{ed})',
                  vertical_title=f'{gcm}' if idx == 0 else None,
                  cmap='cmo.amp',
                  leftcolor='white',
                  tick_step=5,
                  levels=25,
                  )

savepath = f"{config['plot_dir']}/local_risks/CMIP6/df_local_ssps_{gcm}_{gs_dws}_{threshold}.png"
gplt.save_fig(savepath)
# %%
# compare to historical period the local dunkelflauten risk
reload(gplt)

use_era5 = True
gcm = 'MPI-ESM1-2-HR'

if use_era5:
    dfs_per_year_era5 = local_dfs_per_year(df_dict_local['ERA5'])

avs_dfs_per_year_hist = 0
vmin = -15
for ssp, dfs in df_dict_local[gcm].items():
    gcm_str = f'{gcm}_{ssp}'
    im_df = gplt.create_multi_plot(
        nrows=1, ncols=1,
        projection='PlateCarree',
    )

    dfs_per_year = local_dfs_per_year(dfs)

    if use_era5:
        diff = dfs_per_year - dfs_per_year_era5

    gplt.plot_map(diff,
                  ax=im_df['ax'][0],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=vmin,
                  vmax=-vmin,
                  label='No. of Dunkelflauten / Year - Historical (ERA5)',
                  title=f'{gcm} {ssp}',
                  cmap='cmo.balance',
                  centercolor='white',
                  levels=20,
                  )

    savepath = f"{config['plot_dir']}/local_risks/CMIP6/df_local_compare_era5_{gcm_str}_{gs_dws}__{threshold}.png"

    gplt.save_fig(savepath)

# %%
# Average of dunkelflaute per year over multiple GCMs


def get_gcm_average(ssp, df_dict_local, av_type='mean'):
    gcm_arr = []
    for gcm in gcms:
        dfs_per_year = local_dfs_per_year(df_dict_local[gcm][ssp])
        gcm_arr.append(dfs_per_year)
    if av_type == 'mean':
        av_df_gcm = xr.concat(
            gcm_arr, dim='gcm').mean(dim='gcm')
    elif av_type == 'median':
        av_df_gcm = xr.concat(
            gcm_arr, dim='gcm').median(dim='gcm')
    elif av_type == 'max':
        av_df_gcm = xr.concat(
            gcm_arr, dim='gcm').max(dim='gcm')
    elif av_type == 'min':
        av_df_gcm = xr.concat(
            gcm_arr, dim='gcm').min(dim='gcm')
    else:
        raise ValueError(f'Unknown average type: {av_type}')
    std_df_gcm = xr.concat(
        gcm_arr, dim='gcm').std(dim='gcm')

    return av_df_gcm, std_df_gcm


# %%
reload(gplt)

ssp = 'ssp245'
ssp = 'historical'

av_dfs, av_std_dfs = get_gcm_average(ssp=ssp, df_dict_local=df_dict_local)

im_dfs = gplt.create_multi_plot(
    nrows=2, ncols=1,
    projection='PlateCarree',
    hspace=0.5
)

im = gplt.plot_map(av_dfs,
                   ax=im_dfs['ax'][0],
                   plot_borders=True,
                   vmin=0,
                   vmax=25,
                   label='No. of Dunkelflauten / Year',
                   title=f'{tr_str}',
                   vertical_title=f'{ssp}',
                   cmap='cmo.amp',
                   leftcolor='white',
                   tick_step=5,
                   levels=25,
                   )
im_df_std = gplt.plot_map(av_std_dfs,
                          ax=im_dfs['ax'][1],
                          plot_borders=True,
                          vmin=0,
                          vmax=25,
                          label='Std No. of Dunkelflauten / Year',
                          #   title=f'{tr_str}',
                          vertical_title=f'{ssp} Ensemble std',
                          cmap='cmo.amp',
                          leftcolor='white',
                          tick_step=5,
                          levels=25,
                          )

savepath = f"{config['plot_dir']}/local_risks/CMIP6/ensemble_gcms_{ssp}_{gs_dws}_{threshold}.png"
gplt.save_fig(savepath, fig=im_df['fig'])

# %%
dfs_per_year_era5 = local_dfs_per_year(df_dict_local['ERA5'])

vmin = -20
ncols = len(ssps)
im_df = gplt.create_multi_plot(
    nrows=4, ncols=ncols,
    projection='PlateCarree',
    hspace=0.5,
)

for idx_ssp, ssp in enumerate(ssps):
    av_dfs, std_dfs = get_gcm_average(ssp=ssp, df_dict_local=df_dict_local)
    max_dfs, _ = get_gcm_average(ssp=ssp,
                                 df_dict_local=df_dict_local,
                                 av_type='max')
    min_dfs, _ = get_gcm_average(ssp=ssp,
                                 df_dict_local=df_dict_local,
                                 av_type='min')

    sd, ed = tu.get_time_range(gcm_ssp_cf_dict[gcm][ssp]['all']['ts'],
                               asstr=True, m=False, d=False)
    tr = f'{sd}-{ed}'

    diff = av_dfs - dfs_per_year_era5
    diff_max = max_dfs - dfs_per_year_era5
    diff_min = min_dfs - dfs_per_year_era5

    gplt.plot_map(diff,
                  ax=im_df['ax'][idx_ssp],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=vmin,
                  vmax=-vmin,
                  label='No. of Dunkelflauten / Year - Historical (ERA5)',
                  title=f'{ssp} ({tr})',
                  vertical_title=f'Mean (Diff. to ERA5 1980-2015)' if idx_ssp == 0 else None,
                  cmap='cmo.balance',
                  centercolor='white',
                  levels=20,
                  tick_step=4,
                  y_title=1.2,
                  )

    gplt.plot_map(diff_max,
                  ax=im_df['ax'][idx_ssp + ncols*1],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=vmin,
                  vmax=-vmin,
                  label='No. of Dunkelflauten / Year - Historical (ERA5)',
                  vertical_title=f'Max( Diff. to ERA5 1980-2015)' if idx_ssp == 0 else None,
                  cmap='cmo.balance',
                  centercolor='white',
                  levels=20,
                  tick_step=4,
                  )

    gplt.plot_map(diff_min,
                  ax=im_df['ax'][idx_ssp + ncols*2],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=vmin,
                  vmax=-vmin,
                  label='No. of Dunkelflauten / Year - Historical (ERA5)',
                  vertical_title=f'Min( Diff. to ERA5 1980-2015)' if idx_ssp == 0 else None,
                  cmap='cmo.balance',
                  centercolor='white',
                  levels=20,
                  tick_step=4,
                  )

    gplt.plot_map(std_dfs,
                  ax=im_df['ax'][idx_ssp + ncols*3],
                  plot_borders=True,
                  vmin=0,
                  vmax=15,
                  label='Std No. of Dunkelflauten / Year',
                  #   title=f'{tr_str}',
                  vertical_title=f'Ensemble std' if idx_ssp == 0 else None,
                  cmap='cmo.amp',
                  leftcolor='white',
                  tick_step=5,
                  levels=25,
                  )
savepath = f"{config['plot_dir']}/local_risks/CMIP6/df_local_compare_ensemble_era5_{gs_dws}_{threshold}.png"

gplt.save_fig(savepath)
# %%
# Time Series in steps of 10 years
ref_gcm = 'MPI-ESM1-2-HR'
ref_ssp = 'ssp245'
ref_df = df_dict_local[ref_gcm][ref_ssp]


def get_gcm_average_time(ssp, df_dict_local,
                         start_year=2020, end_year=2099,
                         span_years=20):

    time_ranges = tu.split_time_by_year_span(ref_df,
                                             start_year=start_year,
                                             end_year=end_year,
                                             span_years=span_years)
    av_df_tr_dict = {}
    std_df_tr_dict = {}
    for time_range in time_ranges:
        gcm_arr = []
        for gcm in gcms:
            dfs_per_year = local_dfs_per_year(
                tu.get_time_range_data(df_dict_local[gcm][ssp],
                                       time_range=time_range)
            )
            gcm_arr.append(dfs_per_year)
        av_df_gcm = xr.concat(
            gcm_arr, dim='gcm').mean(dim='gcm')
        std_df_gcm = xr.concat(
            gcm_arr, dim='gcm').std(dim='gcm')
        av_df_tr_dict[time_range] = av_df_gcm
        std_df_tr_dict[time_range] = std_df_gcm

    return av_df_tr_dict, std_df_tr_dict


# %%
# Time Series in steps of 10 years
reload(gplt)
reload(tu)

nrows = 2
ncols = 4
im = gplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            figsize=(25, 15),
                            hspace=0.4,
                            wspace=0.3,
                            projection='PlateCarree',
                            )
for idx_ssp, ssp in enumerate(['ssp245', 'ssp585']):

    av_df_tr, std_df_tr = get_gcm_average_time(ssp=ssp,
                                               df_dict_local=df_dict_local,)

    for idx, ((sd, ed), av_dfs) in enumerate(av_df_tr.items()):
        gplt.plot_map(av_dfs,
                      ax=im['ax'][idx + ncols * idx_ssp],
                      plot_borders=True,
                      vmin=0,
                      vmax=15,
                      label='No. of Dunkelflauten / Year',
                      vertical_title=f'Ensemble Mean {ssp}' if idx == 0 else None,
                      title=f'{sd} - {ed}',
                      cmap='cmo.amp',
                      leftcolor='white',
                      levels=25,
                      tick_step=5,
                      )

savepath = f'{config['plot_dir']}/local_risks/CMIP6/dunkelflauten_local_era5_time_series_{sd}_{ed}.png'
gplt.save_fig(savepath, fig=im['fig'])
# %%
