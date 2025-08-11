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
cf_dict_era5['all']['ts_uncorr'] = config['technology']['solar']['weight']*cf_dict_era5['solar']['ts_uncorr'] + config['technology']['offwind']['weight']*cf_dict_era5['offwind']['ts_uncorr'] + config['technology']['onwind']['weight']*cf_dict_era5['onwind']['ts_uncorr']


# %%
# lsm file
lsm_file_country = f'{era5_dir}/{country_name}_nn/{gs_era5}/lsm.nc'
reload(of)
lsm = of.open_nc_file(lsm_file_country)
# %%
# GCM average of capacity factors
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
# plot the single time_series and spatial ensemble mean for all GCMs
reload(tu)
reload(gplt)

source_im_dict = {
    'onwind': {'ls': 'dashed', 'y_range': (0.225, 0.375)},
    'offwind': {'ls': 'dashdot', 'y_range': (0.45, 0.625)},
    'solar': {'ls': 'dotted', 'y_range': (0.1, 0.125)},
    'all': {'ls': 'solid', 'y_range': (0.18, 0.23)},
}


subdevide = 12  # 4 tps per day
window = int(365 * subdevide)
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
        vertical_title_diff = f'Diff. Ensbemble Mean to ERA5 \n({sd_tr} - {ed_tr})'

        label = f'{gcm}'
        data_arr = []
        trend_arr = []
        cap = cf_dict_cmip[source]
        da = cap['ts_uncorr'] if gcm == 'ERA5' else cap['ts']
        if gcm == 'ERA5':
            da = tu.get_time_range_data(da, time_range=(sd, ed))
            if source == 'all':
                # use the uncorrect capacity factor
                print(da)
        data_arr.append(da)

        if window > 1:
            da = tu.rolling_timemean(da, window=window,
                                     center=True,
                                     fill_lims=False,
                                     )

        ts_std = da.std(dim='time')

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
                          color=gplt.colors[idx_gcm],
                          ls=source_im_dict[source]['ls'],
                          ylabel="Capacity Factor [a.u.]" if idx == 0 else None,
                          set_grid=True,
                          ylim=source_im_dict[source]['y_range'],
                          alpha=1.,
                          rot=45,
                          lw=4 if gcm == 'ERA5' else 1.5,
                          y_err=ts_std.data,
                          label=label if idx == len(sources) - 1 else None,
                          loc='under',
                          box_loc=(-2.95, -0.2),
                          ncol_legend=6,
                          zorder=10 if gcm == 'ERA5' else 1,
                          )


    cf, std_cf = source_cf_gcm_dict[source]
    cf_0 = cf_0_dict[source]['cf']
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
    gplt.plot_map(diff,
                  ax=im_pic['ax'][len(sources) + idx],
                  vertical_title=vertical_title_diff if idx == 0 else None,
                  vmin=vmin,
                  vmax=-vmin,
                  levels=20,
                  tick_step=5,
                  cmap='cmo.tarn_r',
                  centercolor='white',
                  label=f'Diff. CMIP6-ERA5',
                  plot_borders=True)


savepath = f"{config['plot_dir']}/capacities_cmip6/all_gcms_{ssp}_{gs_dws}_cf_time_period.png"
gplt.save_fig(savepath, fig=im['fig'])

# %%


