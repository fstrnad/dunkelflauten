# %%
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
# %%
reload(gplt)
# Plot differences to ERA5 in CF for 1 GCM
ssp = 'historical'
gcm = 'MPI-ESM1-2-HR'
ssp_cf_dict = gcm_ssp_cf_dict[gcm]
gcm_str = f'{gcm}_{ssp}'
cf_dict_cmip = ssp_cf_dict[ssp]
ncols = len(cf_dict_cmip)-1
nrows = 1
im = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                            projection='PlateCarree',
                            wspace=0.2)
im_diff = gplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                 projection='PlateCarree',
                                 wspace=0.2)

cf_0_dict = cf_dict_era5
sd, ed = tu.get_time_range(
    cf_0_dict['wind']['ts'], asstr=True, m=False, d=False)
sd_tr, ed_tr = tu.get_time_range(cf_dict_cmip['wind']['ts'], asstr=True)
vertical_title = f'{gcm} - {ssp} \n({sd_tr} - {ed_tr})'
vertical_title_diff = f'Difference to ERA5 \n({sd} - {ed})'
for idx, (sname, source) in enumerate(cf_dict_cmip.items()):
    if sname != 'wind':
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
                      vertical_title=vertical_title if idx == 0 else None,
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


def get_gcm_average(ssp, cf_dicts,
                    source_types=['onwind', 'offwind',
                                  'solar', 'all']):

    # cf_dicts = dcp(cf_dicts)
    source_cf_gcm_dict = {}
    for source in source_types:
        tr_arr = []
        for gcm in gcms:
            if cf_dicts[gcm][ssp]:
                tr_arr.append(cf_dicts[gcm][ssp][source]['cf'])
        av_cf_gcm = xr.concat(
            tr_arr, dim='gcm').mean(dim='gcm')
        std_cf_gcm = xr.concat(
            tr_arr, dim='gcm').std(dim='gcm')
        source_cf_gcm_dict[source] = (av_cf_gcm, std_cf_gcm)

    return source_cf_gcm_dict


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
    cf_0_dict['wind']['ts'], asstr=True, m=False, d=False)

for idx, (sname, source) in enumerate(source_cf_gcm_dict.items()):
    source_avs, source_stds = source
    if sname != 'wind' and sname != 'all':
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
# Plot the time series

# %%
# Plot Capacity Factors
timemean = 'dayofyear'
ssp = 'ssp245'
gcm = 'MPI-ESM1-2-HR'
ssp_cf_dict = gcm_ssp_cf_dict[gcm]
gcm_str = f'{gcm}_{ssp}'
window = 8
yearly_ts_all = 0
for ssp, cf_dict_cmip in ssp_cf_dict.items():
    gcm_str = f'{gcm}_{ssp}'
    data_arr = []
    for cap_name, cap in cf_dict_cmip.items():
        if cap_name == 'wind':
            continue
        data = cap['ts']
        ts_mean = tu.rolling_timemean(data, window=window)
        yearly_ts = tu.compute_timemean(ts_mean,
                                        timemean=timemean,
                                        groupby=True,
                                        reset_time=True,)
        data_arr.append(yearly_ts)

    colors = ['blue', 'tab:blue', 'orange', 'black']

    sd, ed = tu.get_time_range(data, asstr=True)
    im = gplt.plot_2d(x=data_arr[0].time,
                      y=data_arr,
                      title=f"Contributions to capacity factor ({sd} - {ed})",
                      vertical_title=f'{gcm} - {ssp}',
                      label_arr=list(cf_dict_cmip.keys()) + ['All'],
                      color_arr=colors,
                      xlabel="Day of Year",
                      ylabel="Capacity Factor [a.u.]",
                      set_grid=True,
                      ylim=(0, .9),
                      loc='outside',
                      )
    savepath = f"{config['plot_dir']}/capacities_cmip6/single_years/{country_name}_{gcm_str}_{tr_str}_{gs_dws}_capacity_factor_per_day.png"
    gplt.save_fig(savepath, fig=im['fig'])

# %%
# plot the single time_series
reload(tu)
reload(gplt)

source_im_dict = {
    'onwind': {'ls': 'dashed', 'y_range': (0.25, 0.35)},
    'offwind': {'ls': 'dashdot', 'y_range': (0.49, 0.59)},
    'solar': {'ls': 'dotted', 'y_range': (0.1, 0.13)},
    'all': {'ls': 'solid', 'y_range': (0.2, 0.28)},
}


subdevide = 12  # 4 tps per day
window = int(365*subdevide)
yearly_ts_all = 0
x_lin_reg = np.linspace(0, 1, 100)
time_range = None

ssp = 'historical'

sources = ['onwind', 'offwind', 'solar', 'all']
im_pic = gplt.create_multi_plot(nrows=2, ncols=len(sources),
                                projection_arr=[
                                    None, None, None, None,
    'PlateCarree', 'PlateCarree', 'PlateCarree', 'PlateCarree',],
    hspace=0.5, wspace=0.2,
    fig_size=(22, 6))

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
        label = f'{gcm}'
        data_arr = []
        trend_arr = []
        cap = cf_dict_cmip[source]
        da = cap['ts'] if gcm != 'ERA5' or source == 'all' else cap['ts_uncorr']
        if gcm == 'ERA5':
            da = tu.get_time_range_data(da, time_range=(sd, ed))
            if source == 'all':
                da -= 0.03  # use the uncorrect capacity factor
        data_arr.append(da)
        if window > 1:
            da = tu.rolling_timemean(da, window=window,
                                     center=False,
                                     )
        time = da.time
        # compute the trend
        num_tps = (time.values - time.values[0]
                   ).astype('timedelta64[h]').astype(int)
        coeffs = np.polyfit(num_tps, da.values, deg=1)
        trend = np.polyval(coeffs, num_tps)
        trend = xr.DataArray(
            trend, coords={'time': da.time}, dims='time')
        im = gplt.plot_2d(x=da.time,
                          y=da.values,
                          ax=im_pic['ax'][idx],
                          #  power_pv.values,
                          title=f"{source} CF",
                          color=gplt.colors[idx_gcm+1],
                          xlabel="Year",
                          ls=source_im_dict[source]['ls'],
                          ylabel="Capacity Factor [a.u.]" if idx == 0 else None,
                          set_grid=True,
                          ylim=source_im_dict[source]['y_range'],
                          alpha=0.3,
                          rot=45,
                          )
        gplt.plot_2d(x=trend.time,
                     y=trend.values,
                     ax=im['ax'],
                     #  loc='outside',
                     ncol_legend=2,
                     label=label if idx == len(sources) - 1 else None,
                     lw=3,
                     color=gplt.colors[idx_gcm+1],
                     set_axis=False,
                     )

    cf, std_cf = source_cf_gcm_dict[source]
    cf_0 = cf_0_dict[source]['cf']

    vmin = -0.01 if source == 'solar' else -0.05
    gplt.plot_map(cf - cf_0,
                  ax=im_pic['ax'][len(sources) + idx],
                  vertical_title=vertical_title if idx == 0 else None,
                  vmin=vmin,
                  vmax=-vmin,
                  levels=20,
                  tick_step=5,
                  cmap='cmo.tarn_r',
                  centercolor='white',
                  label=f'Diff. CF ERA5 (1980-2015)',
                  plot_borders=True)


savepath = f"{config['plot_dir']}/capacities_cmip6/all_gcms_{ssp}_{gs_dws}_cf_time_period.png"
gplt.save_fig(savepath, fig=im['fig'])


