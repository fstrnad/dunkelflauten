# %%
from tqdm import tqdm
import sys
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

gcm = 'MPI-ESM1-2-HR'
gcm = 'GFDL-ESM4'
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
tr_ssp = [('2020-01-01', '2100-01-01')]  # full range

ssps = ['historical', 'ssp245', 'ssp585']

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
cf_dict_era5['all']['ts_uncorr'] = config['technology']['solar']['weight']*cf_dict_era5['solar']['ts_uncorr'] + config['technology']['offwind']['weight'] * \
    cf_dict_era5['offwind']['ts_uncorr'] + \
    config['technology']['onwind']['weight'] * \
    cf_dict_era5['onwind']['ts_uncorr']


# %%
# lsm file
lsm_file_country = f'{era5_dir}/{country_name}_nn/{gs_era5}/lsm.nc'
reload(of)
lsm = of.open_nc_file(lsm_file_country)
# %%
# GCM average of capacity factors


def get_gcm_average(ssp, cf_dicts,
                    source_types=['onwind', 'offwind',
                                  'solar', 'all'],
                    var_type='cf'):

    # cf_dicts = dcp(cf_dicts)
    source_cf_gcm_dict = {}
    for source in source_types:
        tr_arr = []
        for gcm in gcms:
            if cf_dicts[gcm][ssp]:
                tr_arr.append(cf_dicts[gcm][ssp][source][var_type])
        av_cf_gcm = xr.concat(
            tr_arr, dim='gcm').mean(dim='gcm')
        std_cf_gcm = xr.concat(
            tr_arr, dim='gcm').std(dim='gcm')
        source_cf_gcm_dict[source] = (av_cf_gcm, std_cf_gcm)

    return source_cf_gcm_dict


def get_gcm_quantile(ssp, cf_dicts,
                     source_types=['onwind', 'offwind',
                                   'solar', 'all'],
                     var_type='cf_ts',
                     q=0.1):

    # cf_dicts = dcp(cf_dicts)
    source_cf_gcm_dict = {}
    source_cf_gcm_max_dict = {}
    source_cf_gcm_min_dict = {}

    for source in source_types:
        tr_arr = []
        for gcm in gcms:
            if cf_dicts[gcm][ssp]:
                data = cf_dicts[gcm][ssp][source][var_type]
                data = sput.check_dimensions(data)
                tr_arr.append(data.quantile(q, dim='time'))
        q_cf_gcm = xr.concat(
            tr_arr, dim='gcm').mean(dim='gcm')
        q_cf_gcm_max = xr.concat(
            tr_arr, dim='gcm').max(dim='gcm')
        q_cf_gcm_min = xr.concat(
            tr_arr, dim='gcm').min(dim='gcm')

        source_cf_gcm_dict[source] = q_cf_gcm
        source_cf_gcm_max_dict[source] = q_cf_gcm_max
        source_cf_gcm_min_dict[source] = q_cf_gcm_min

    return source_cf_gcm_dict, source_cf_gcm_max_dict, source_cf_gcm_min_dict


# %%
reload(gplt)
# Plot differences to ERA5 in CF for 1 GCM
ssp = 'historical'
gcm = 'MPI-ESM1-2-HR'
ssp_cf_dict = gcm_ssp_cf_dict[gcm]
gcm_str = f'{gcm}_{ssp}'
cf_dict_cmip = ssp_cf_dict[ssp]
ncols = len(cf_dict_cmip)
nrows = 1
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2)
im_diff = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                 projection='PlateCarree',
                                 wspace=0.2)

cf_0_dict = cf_dict_era5
sd, ed = tu.get_time_range(
    cf_0_dict['onwind']['ts'], asstr=True, m=False, d=False)
sd_tr, ed_tr = tu.get_time_range(cf_dict_cmip['onwind']['ts'], asstr=True)
vertical_title = f'{gcm} - {ssp} \n({sd_tr} - {ed_tr})'
vertical_title_diff = f'Difference to ERA5 \n({sd} - {ed})'
for idx, (sname, source) in enumerate(cf_dict_cmip.items()):

    cap_fac = source['cf']
    cf_0 = cf_0_dict[sname]['cf']
    gplt.plot_map(cap_fac,
                  ax=im['ax'][idx],
                  title=sname,
                  vertical_title=vertical_title if idx == 0 else None,
                  cmap='cmo.haline',
                  vmin=0,
                  vmax=.15 if sname == 'solar' else 0.4,
                  label='Capacity Factor [a.u.]',
                  plot_borders=True)

    # Plot the difference to the first time period
    vmin = -0.03 if sname == 'solar' else -0.05
    gplt.plot_map(cap_fac - cf_0,
                  ax=im_diff['ax'][idx],
                  title=sname,
                  vertical_title=vertical_title_diff if idx == 0 else None,
                  vmin=vmin,
                  vmax=-vmin,
                  cmap='cmo.tarn_r',
                  centercolor='white',
                  label=f'Difference CF ({sd} - {ed}) [a.u.]',
                  plot_borders=True)

savepath = f"{config['plot_dir']}/capacities_cmip6/cf_types_full_{gcm_str}_{fine_res}.png"
gplt.save_fig(savepath, fig=im['fig'])
savepath_diff = f"{config['plot_dir']}/capacities_cmip6/cf_types_diff_{gcm_str}_{fine_res}.png"
gplt.save_fig(savepath_diff, fig=im_diff['fig'])


# %%
# Plot differences to ERA5 in CF for ensemble mean
ssp = 'historical'
source_cf_gcm_dict = get_gcm_average(ssp=ssp,
                                     cf_dicts=gcm_ssp_cf_dict,)
nrows = 1
ncols = len(source_cf_gcm_dict) - 1
im_diff = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                 projection='PlateCarree',
                                 wspace=0.2)

cf_0_dict = cf_dict_era5
sd, ed = tu.get_time_range(
    cf_0_dict['onwind']['ts'], asstr=True, m=False, d=False)

for idx, (sname, source) in enumerate(source_cf_gcm_dict.items()):
    source_avs, source_stds = source
    if sname != 'all':
        sd_tr, ed_tr = tu.get_time_range(
            gcm_ssp_cf_dict[gcm][ssp]['offwind']['ts'], asstr=True)
        vertical_title = f'Ensbemble Mean {ssp} \n({sd_tr} - {ed_tr})'
        cap_fac = source_avs
        cf_0 = tu.get_time_range_data(cf_0_dict[sname]['cf_ts'],
                                      time_range=(sd_tr, ed_tr),
                                      ).mean(dim='time')

        vmin = -0.04 if sname == 'solar' else -0.05
        gplt.plot_map(cap_fac - cf_0,
                      ax=im_diff['ax'][idx],
                      title=sname,
                      vertical_title=vertical_title if idx == 0 else None,
                      vmin=vmin,
                      vmax=-vmin,
                      levels=20,
                      tick_step=5,
                      cmap='cmo.tarn_r',
                      centercolor='white',
                      label=f'Diff. CF ERA5 (1980-2015)',
                      plot_borders=True)

savepath_diff = f"{config['plot_dir']}/capacities_cmip6/cf_types_diff_ensemble_mean_{ssp}.png"
gplt.save_fig(savepath_diff, fig=im_diff['fig'])

# %%
# plot the single time_series and spatial ensemble mean for all GCMs
reload(tu)
reload(gplt)

sources = ['onwind', 'offwind', 'solar', 'all']

source_im_dict = {
    'onwind': {'ls': 'dashed', 'y_range': (0.27, 0.33)},
    'offwind': {'ls': 'dashdot', 'y_range': (0.5, 0.57)},
    'solar': {'ls': 'dotted', 'y_range': (0.1075, 0.12)},
    'all': {'ls': 'solid', 'y_range': (0.19, 0.215)},
}

ncols = len(sources)
im_pic = gplt.create_multi_plot(
    nrows=3, ncols=ncols,
    projection_arr=[
        None, None, None, None,
        None, None, None, None,
        'PlateCarree', 'PlateCarree', 'PlateCarree', 'PlateCarree',],
    hspace=0.5, wspace=0.29,
    figsize=(13, 11)
)

lon_range_ger = [5, 16]
lat_range_ger = [47, 55.5]

subdevide = 4  # 4 tps per day
num_years = 10
window = int(365 * subdevide * num_years)
yearly_ts_all = 0
x_lin_reg = np.linspace(0, 1, 100)
time_range = None

ssp = 'historical'

all_gcms = ['ERA5'] + gcms
tr_ref = list(gcm_ssp_cf_dict[all_gcms[1]][ssp].keys())[
    0]  # use the first time range as reference
ref_ts_cmip = gcm_ssp_cf_dict[all_gcms[1]][ssp][sources[0]]['ts']
sd, ed = tu.get_time_range(ref_ts_cmip)
sd_tr, ed_tr = tu.get_time_range(ref_ts_cmip, asstr=True, m=False, d=False)
cf_0_dict = cf_dict_era5

for idx, source in enumerate(sources):
    for idx_gcm, gcm in enumerate(all_gcms):
        source_cf_gcm_dict = get_gcm_average(ssp=ssp,
                                             cf_dicts=gcm_ssp_cf_dict,)
        if gcm == 'ERA5':
            cf_dict_cmip = cf_dict_era5
        else:
            ssp_cf_dict = gcm_ssp_cf_dict[gcm]
            cf_dict_cmip = ssp_cf_dict[ssp]
        if not cf_dict_cmip:
            continue

        vertical_title = f'Ensbemble Mean {ssp} \n({sd_tr} - {ed_tr})'
        vertical_title_diff = f'Diff. Ensb. Mean - ERA5 \n({sd_tr} - {ed_tr})'

        label = f'{gcm}'
        data_arr = []
        trend_arr = []
        cap = cf_dict_cmip[source]
        da = cap['ts_uncorr'] if gcm == 'ERA5' else cap['ts']
        if gcm == 'ERA5':
            da = tu.get_time_range_data(da, time_range=(sd, ed))

        data_arr.append(da)

        if window > 1:
            ds_std = tu.rolling_timemean(da, window=window,
                                         center=True,
                                         fill_lims=False,
                                         rolling_std=True,
                                         method_std='std',
                                         )
            da_plot = ds_std['cf']
            da_std = ds_std['cf_std']
        else:
            da_plot = da
            da_std = None

        ts_std = da_plot.std(dim='time')

        time = da_plot.time
        # compute the trend
        num_tps = (time.values - time.values[0]
                   ).astype('timedelta64[h]').astype(int)
        coeffs = np.polyfit(num_tps, da_plot.values, deg=1)
        trend = np.polyval(coeffs, num_tps)
        trend = xr.DataArray(
            trend, coords={'time': da_plot.time}, dims='time')

        vertical_title = f'Decadal Spatial Mean'
        title = f"{source} CF" if source != 'all' else 'combined CF'
        gplt.plot_2d(x=da_plot.time,
                     y=da_plot.values,
                     ax=im_pic['ax'][idx],
                     #  power_pv.values,
                     title=title,
                     vertical_title=vertical_title if idx == 0 and idx_gcm == 0 else None,
                     x_title_offset=-0.55,
                     color=gplt.colors[idx_gcm],
                     ylabel="Capacity Factor [a.u.]" if idx == 0 else None,
                     set_grid=True,
                     ylim=source_im_dict[source]['y_range'],
                     alpha=1.,
                     rot=45,
                     lw=3. if gcm == 'ERA5' else 1.5,
                     ls='-' if gcm == 'ERA5' else '-',
                     zorder=10 if gcm == 'ERA5' else 1,
                     )

        vertical_title = f'Decadal Standard Deviation'
        gplt.plot_2d(x=da_std.time,
                     y=da_std.values,
                     ax=im_pic['ax'][idx+ncols],
                     vertical_title=vertical_title if idx == 0 and idx_gcm == 0 else None,
                     x_title_offset=-0.55,
                     color=gplt.colors[idx_gcm],

                     ylabel="Std Capacity Factor [a.u.]" if idx == 0 else None,
                     set_grid=True,
                     rot=45,
                     lw=3. if gcm == 'ERA5' else 1.5,
                     ls='-' if gcm == 'ERA5' else '-',
                     label=label if idx == len(sources) - 1 else None,
                     loc='under',
                     box_loc=(-4.25, -0.24),
                     ncol_legend=6,
                     )

    cf, std_cf = source_cf_gcm_dict[source]
    cf_0 = cf_dict_era5[source]['cf']
    diff = cf - cf_0
    diff = sput.check_dimensions(diff)
    if source == 'onwind' or source == 'solar':
        diff = xr.where(lsm > 0.5, diff, np.nan)
    elif source == 'offwind':
        diff = xr.where(lsm < 0.5, diff, np.nan)
        diff_off = diff.copy()
    elif source == 'all':
        diff = xr.where(lsm > 0.5, diff, diff_off)

    vmin = -0.01
    im_map = gplt.plot_map(diff,
                           ax=im_pic['ax'][ncols*2 + idx],
                           vertical_title=vertical_title_diff if idx == 0 else None,
                           vmin=vmin,
                           vmax=-vmin,
                           levels=20,
                           tick_step=5,
                           cmap='cmo.delta',
                           centercolor='white',
                           label=f'Diff. CMIP6-ERA5' if idx == ncols - 1 else None,
                           orientation='vertical',
                           extend='both',
                           lon_range=lon_range_ger,
                           lat_range=lat_range_ger,
                           x_title_offset=-0.55,
                           plot_borders=True)


savepath = f"{config['plot_dir']}/capacities_cmip6/all_gcms_{ssp}_cf_time_period.png"
gplt.save_fig(savepath, fig=im_pic['fig'])

# %%

gcm = 'MPI-ESM1-2-HR'
ssp = 'historical'
source = 'onwind'


data = gcm_ssp_cf_dict[gcm][ssp][source]['cf_winter_ts']
data_era5 = cf_dict_era5[source]['cf_winter_ts']
q = 0.2
data_q = data.quantile(q, dim='time')
data_era5_q = data_era5.quantile(q, dim='time')

im = gplt.plot_map(data_q,
                   title=f'{gcm} {ssp} {source} {q} quantile',
                   vmin=-0.0,
                   vmax=0.1,
                   cmap='cmo.haline',
                   label='Capacity Factor [a.u.]',
                   lon_range=lon_range_ger,
                   lat_range=lat_range_ger,
                   levels=20,
                   plot_borders=True)

# %%
var_type = 'cf_winter_ts'
q = 0.2
ensemble_mean, ensemble_max, ensemble_min = get_gcm_quantile(
    ssp=ssp,
    cf_dicts=gcm_ssp_cf_dict,
    var_type=var_type,
    q=q)

# %%
data_ensemble = ensemble_mean[source]

im = gplt.plot_map(data_ensemble,
                   title=f'CMIP6 {ssp} {source} {q} quantile',
                   vmin=-0.0,
                   vmax=0.1,
                   cmap='cmo.haline',
                   label='Capacity Factor [a.u.]',
                   lon_range=lon_range_ger,
                   lat_range=lat_range_ger,
                   #   centercolor='white',
                   levels=20,
                   plot_borders=True)
# %%
# %%
# Compare ERA5 to historical, ssp245, ssp585 all scenarios
fine_res = 0.25
lsm_file_country = f'{era5_dir}/{country_name}_nn/{fine_res}/lsm.nc'
lsm_fine = of.open_nc_file(lsm_file_country)
lsm_fine = xr.where(lsm_fine > 0.5, 1, np.nan)

var_type = 'cf_winter_ts'
this_source = 'onwind'
this_sources = [this_source]
data_era5 = cf_dict_era5[this_source][var_type]

q = 0.2

data_era5_q = data_era5.quantile(q, dim='time')

time_future = ('2080-01-01', '2100-12-31')

vmin = -0.1
vmax = 0.1

vmin_rel = -0.2
vmax_rel = 0.2

ncols = len(ssps)
nrows = 4
im_df = gplt.create_multi_plot(
    nrows=nrows, ncols=ncols,
    projection='PlateCarree',
    hspace=0.,
    figsize=(16, 18),
)

label = 'Diff. events/year (CMIP6-ERA5)'

for idx_ssp, ssp in enumerate(ssps):
    ensemble_mean, ensemble_max, ensemble_min = get_gcm_quantile(
        ssp=ssp,
        cf_dicts=gcm_ssp_cf_dict,
        source_types=this_sources,
        var_type=var_type,
        q=q)

    ref_ts = gcm_ssp_cf_dict[gcms[0]][ssp][this_source]['ts']
    sd, ed = tu.get_time_range(ref_ts,
                               asstr=True, m=False, d=False)
    tr = f'{sd}-{ed}'

    diff = 1 - (ensemble_mean[this_source] / data_era5_q)
    diff_max = 1 - (ensemble_max[this_source] / data_era5_q)
    diff_min = 1 - (ensemble_min[this_source] / data_era5_q)

    gplt.plot_map(ensemble_mean[this_source] * lsm_fine,
                  ax=im_df['ax'][idx_ssp],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=0,
                  vmax=vmax,
                  label=label if idx_ssp == ncols - 1 else None,
                  orientation='vertical',
                  title=f'{ssp} ({tr})',
                  vertical_title=f'Ensemble Mean' if idx_ssp == 0 else None,
                  cmap='cmo.haline',
                  levels=20,
                  tick_step=4,
                  y_title=1.2,
                  #   mask=mask,
                  lon_range=lon_range_ger,
                  lat_range=lat_range_ger,
                  )

    gplt.plot_map(ensemble_max[this_source] * lsm_fine,
                  ax=im_df['ax'][idx_ssp + ncols*1],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=0,
                  vmax=vmax,
                  label=label if idx_ssp == ncols - 1 else None,
                  orientation='vertical',
                  vertical_title=f'Ensemble Max' if idx_ssp == 0 else None,
                  cmap='cmo.haline',
                  levels=20,
                  tick_step=4,
                  #   mask=mask,
                  lon_range=lon_range_ger,
                  lat_range=lat_range_ger,
                  )

    gplt.plot_map(ensemble_min[this_source] * lsm_fine,
                  ax=im_df['ax'][idx_ssp + ncols*2],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=0,
                  vmax=vmax,
                  label=label if idx_ssp == ncols - 1 else None,
                  orientation='vertical',
                  vertical_title=f'Ensemble Min' if idx_ssp == 0 else None,
                  cmap='cmo.haline',
                  levels=20,
                  tick_step=4,
                  #   mask=mask,
                  lon_range=lon_range_ger,
                  lat_range=lat_range_ger,
                  )

    gplt.plot_map(ensemble_min[this_source] * lsm_fine,
                  ax=im_df['ax'][idx_ssp + ncols*3],
                  plot_borders=True,
                  #   significance_mask=xr.where(mask, 0, 1),
                  vmin=vmin_rel,
                  vmax=vmax_rel,
                  label=label if idx_ssp == ncols - 1 else None,
                  orientation='vertical',
                  vertical_title=f'Relative Diff to ERA5' if idx_ssp == 0 else None,
                  cmap='cmo.balance',
                  centercolor='white',
                  levels=20,
                  tick_step=4,
                  #   mask=mask,
                  lon_range=lon_range_ger,
                  lat_range=lat_range_ger,
                  )


savepath = f"{config['plot_dir']}/local_risks/CMIP6/compare_ensemble_era5_{var_type}.png"

# gplt.save_fig(savepath)
# %%
# Use again time points below threshold and count relative to all days


def get_event_counts(data, num_hours=48, hourly_res=6, threshold=0.02):
    data = sput.rename_dims(data)

    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    data_roll = tu.rolling_timemean(data, window=window)
    df_local, _ = tu.compute_evs(data_roll,
                                 threshold=threshold,
                                 threshold_type='lower',
                                 #    max_rel_share=0.02,
                                 get_mask=True)
    # count all events per cell
    rel_share_events = df_local.sum(dim='time') / data_roll.time.size * 100
    rel_share_events = rel_share_events.rename('rel_share_events')

    return rel_share_events


def gcm_rel_shares(ssp, cf_dicts,
                   source='onwind',
                   var_type='cf_winter_ts',
                   num_hours=48,
                   hourly_res=6,
                   threshold=0.02,
                   time_range=None):

    tr_arr = []
    for gcm in tqdm(gcms):
        if cf_dicts[gcm][ssp]:
            data = cf_dicts[gcm][ssp][source][var_type]
            data = sput.check_dimensions(data)
            data = tu.get_time_range_data(data, time_range=time_range)
            rel_shares = get_event_counts(data,
                                          num_hours=num_hours,
                                          hourly_res=hourly_res,
                                          threshold=threshold)
            tr_arr.append(rel_shares)
    rel_shares_gcm = xr.concat(
        tr_arr, dim='gcm')

    return rel_shares_gcm

# %%


threshold = 0.02

rel_shares_era5 = get_event_counts(cf_dict_era5['all']['cf_winter_ts'])
rel_shares_ensemble_hist = gcm_rel_shares(ssp='historical',
                                          cf_dicts=gcm_ssp_cf_dict,
                                          source='all',
                                          var_type='cf_winter_ts',
                                          num_hours=48,
                                          hourly_res=6,
                                          threshold=threshold)
# %%
time_range = ('2020-01-01', '2050-12-31')
rel_shares_ensemble_ssp245 = gcm_rel_shares(ssp='ssp245',
                                            cf_dicts=gcm_ssp_cf_dict,
                                            source='all',
                                            var_type='cf_winter_ts',
                                            num_hours=48,
                                            hourly_res=6,
                                            threshold=threshold,
                                            time_range=time_range)
rel_shares_ensemble_ssp585 = gcm_rel_shares(ssp='ssp585',
                                            cf_dicts=gcm_ssp_cf_dict,
                                            source='all',
                                            var_type='cf_winter_ts',
                                            num_hours=48,
                                            hourly_res=6,
                                            threshold=threshold,
                                            time_range=time_range)
# %%
rel_shares_dict = {
    'historical': rel_shares_ensemble_hist,
    'ssp245': rel_shares_ensemble_ssp245,
    'ssp585': rel_shares_ensemble_ssp585,
}

# %%
reload(gplt)
ncols = 3
nrows = 3
lon_range_ger = [5, 16]
lat_range_ger = [47, 55.5]
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2,
                            hspace=0.3,
                            rem_idx=[3, 6],
                            figsize=(14, 10),
                            )

gplt.plot_map(rel_shares_era5,
              label='Occurrences per season [%]',
              ax=im['ax'][0],
              vertical_title='historical (1985-2015)',
              title='ERA5',
              vmin=0,
              vmax=2,
              lon_range=lon_range_ger,
              lat_range=lat_range_ger,
              plot_borders=True,
              cmap='cmo.tempo',
              leftcolor='white',
              levels=20,
              orientation='horizontal',
              tick_step=4,)

for idx_ssp, (ssp_name, rel_shares_ensemble) in enumerate(rel_shares_dict.items()):
    rel_share = rel_shares_ensemble.max(dim='gcm')

    label = 'Occurrences per season [%]'
    gplt.plot_map(rel_share,
                  label=label if idx_ssp == ncols - 1 else None,
                  ax=im['ax'][idx_ssp*ncols + 1],
                  title='Ensemble Max CMIP6' if idx_ssp == 0 else None,
                  vertical_title=f'{ssp_name} (2020-2050)' if idx_ssp > 0 else None,
                  vmin=0,
                  vmax=2,
                  lon_range=lon_range_ger,
                  lat_range=lat_range_ger,
                  plot_borders=True,
                  cmap='cmo.tempo',
                  leftcolor='white',
                  levels=20,
                  orientation='horizontal',
                  tick_step=4,)

    diff = rel_share - rel_shares_era5
    rel_diff = (diff / rel_share) * 100
    label = 'Diff. occurrences per season [%]'
    vmin_rel = -2.
    vmax_rel = 2.
    gplt.plot_map(diff,
                  label=label if idx_ssp == ncols - 1 else None,
                  title='Diff. CMIP6 Max - ERA5' if idx_ssp == 0 else None,
                  ax=im['ax'][idx_ssp*ncols + 2],
                  vmin=vmin_rel,
                  vmax=vmax_rel,
                  lon_range=lon_range_ger,
                  lat_range=lat_range_ger,
                  plot_borders=True,
                  cmap='cmo.diff',
                  centercolor='white',
                  levels=10,
                  tick_step=2,
                  orientation='horizontal',)

savepath = f"{config['plot_dir']}/local_risks/CMIP6/local_compare_ensemble_era5_{threshold}.png"

gplt.save_fig(savepath, fig=im['fig'])

# %%
# plot the standard deviation of the ensemble for historical, ssp245, ssp585
nrows = 1
ncols = 3
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.3,
                            hspace=0.3,
                            # figsize=(14, 10),
                            )


for idx_ssp, (ssp_name, rel_shares_ensemble) in enumerate(rel_shares_dict.items()):
    rel_share_std = rel_shares_ensemble.std(dim='gcm')

    label = 'Std occ./season [%]'
    gplt.plot_map(rel_share_std,
                  label=label if idx_ssp == ncols - 1 else None,
                  ax=im['ax'][idx_ssp],
                  vertical_title='Ensemble Std CMIP6' if idx_ssp == 0 else None,
                  title=f'{ssp_name} (2020-2050)' if idx_ssp > 0 else f'{ssp_name} (1985-2015)',
                  vmin=0,
                  vmax=1,
                  lon_range=lon_range_ger,
                  lat_range=lat_range_ger,
                  plot_borders=True,
                  cmap='cmo.tempo',
                  leftcolor='white',
                  levels=20,
                  orientation='vertical',
                  tick_step=4,)


savepath = f"{config['plot_dir']}/local_risks/CMIP6/local_compare_ensemble_era5_std_{threshold}.png"

gplt.save_fig(savepath, fig=im['fig'])