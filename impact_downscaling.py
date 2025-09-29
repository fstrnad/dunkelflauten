# %%
import cf_utils as cfu
import geoutils.utils.met_utils as mut
import geoutils.geodata.solar_radiation as sr
import pre_processing.workaround_fsr as wf
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
# lsm file
reload(of)
lsm_file_country = f'{era5_dir}/{country_name}_nn/{fine_res}/lsm.nc'
lsm_fine = of.open_nc_file(lsm_file_country)
lsm_file_coarse = f'{era5_dir}/{country_name}_av/{coarse_res}/lsm.nc'
lsm_coarse = of.open_nc_file(lsm_file_coarse)


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
sources = ['onwind', 'solar', ]
ncols = len(cf_dicts)
nrows = len(sources) + 1
im_cfs = gplt.create_multi_plot(
    nrows=nrows, ncols=ncols,
    hspace=-0.2,
    wspace=0.17,
    projection='PlateCarree')

threshold = 0.02

lon_range_ger = [5, 16]
lat_range_ger = [47, 55.5]

for idx, (res, cf_dict_era5) in enumerate(cf_dicts.items()):
    hourly_res = tu.get_frequency_resolution_hours(
        cf_dict_era5['solar']['cf_ts'])
    res = res if res != 'daily' else '0.25'
    title = f'{res}°, {hourly_res}h resolution'
    cf_onwind_solar = cf_dict_era5['all']['cf_ts']
    cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
    gs, _, _ = sput.get_grid_step(cf_onwind_solar)
    mask = cf_mask if gs == fine_res else cf_mask_coarse
    for s, sname in enumerate(sources):
        sd, ed = tu.get_time_range(
            cf_dict_era5[sname]['cf_ts'], asstr=True, m=False, d=False)

        cap_fac = cf_dict_era5[sname]['cf']
        cap_fac = sput.check_dimensions(cap_fac)

        gplt.plot_map(cap_fac,
                      ax=im_cfs['ax'][s*ncols + idx],
                      title=title if s == 0 else None,
                      vertical_title=f'{sname} capacity factor \n{sd} - {ed}' if idx == 0 else None,
                      y_title=1.2,
                      #   mask=mask,
                      cmap='cmo.thermal',
                      #   levels=25,
                      tick_step=5,
                      vmin=0.08 if sname == 'solar' else 0.0,
                      vmax=.14 if sname == 'solar' else 0.4,
                      lon_range=lon_range_ger,
                      lat_range=lat_range_ger,
                      plot_borders=True,
                      orientation='vertical',
                      label='Capacity Factor [a.u.]' if idx == ncols -
                      1 else None,
                      )

    # window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    # cf_ts_mean = tu.rolling_timemean(cf_onwind_solar, window=window)
    # df_local_onwind, _ = tu.compute_evs(cf_ts_mean,
    #                                     threshold=threshold,
    #                                     threshold_type='lower',
    #                                     get_mask=True)

    # num_years = tu.count_unique_years(df_local_onwind)
    # num_dfs_cell = df_local_onwind.sum(dim='time')
    # gplt.plot_map(num_dfs_cell/num_years,
    #               ax=im_cfs['ax'][ncols*len(sources) + idx],
    #               plot_borders=True,
    #               vmin=3,
    #               vmax=18,
    #               label='No. of Dunkelflauten / Year',
    #               #   title=title,
    #               vertical_title=f'No. of Dunkelflaute events \n{sd} - {ed}' if idx == 0 else None,
    #               cmap='cmo.amp',
    #               leftcolor='white',
    #               mask=mask,
    #               levels=15,
    #               tick_step=5,
    #               lon_range=lon_range_ger,
    #               lat_range=lat_range_ger,
    #               )

    quantile = 0.03
    num_hours = 48
    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    ts = cf_dict_era5['all']['cf_winter_ts'] if res != 'DS BC 0.25' else cf_dicts['GT 0.25']['all']['cf_winter_ts']
    cf_ts_mean = tu.rolling_timemean(
        ts, window=window)
    cap_fac_qt = cf_ts_mean.quantile(q=quantile, dim='time')
    cap_fac_qt = sput.check_dimensions(cap_fac_qt)
    risk = cap_fac_qt
    im = gplt.plot_map(
        risk,
        ax=im_cfs['ax'][ncols*len(sources) + idx],
        vertical_title=r'Low capacity factors (Q$_{{0.05}}$)' +
        f'\nNov - Jan {sd}-{ed}' if idx == 0 else None,
        y_title=1.2,
        cmap='cmo.oxy_r',
        # mask=mask,
        # levels=25,
        tick_step=5,
        vmin=0.02,
        vmax=0.06,
        lon_range=lon_range_ger,
        lat_range=lat_range_ger,
        plot_borders=True,
        orientation='vertical',
        label=r'Quantile$_{0.05}$(CF) [a.u.]' if idx == ncols - 1 else None,
    )
    # break

savepath_cfs = f"{config['plot_dir']}/impact_downscaling/compare_res_ERA5_{tr_str}.png"

gplt.save_fig(savepath_cfs, fig=im_cfs['fig'])


# %%
# Plot the respective cf time series of the single different resolutions
reload(gplt)
cf_dicts_ts = {
    f'GT {fine_res}': cf_era5_fine_gt['all']['ts'],
    f'Coarse {coarse_res}': cf_era5_fine['all']['ts'],
    'daily': cf_era5_daily['all']['ts'],
    f'DS {fine_res}': cf_era5_fine_bc['all']['ts'],
    f'DS BC {fine_res}': cf_era5_coarse['all']['ts'],
}

time_range = ['2024-11-01', '2024-11-30']
im = gplt.create_multi_plot(
    nrows=1, ncols=1,

    figsize=(8, 4),
    projection=None)

gt = f'GT {fine_res}'
for idx, (res, cf_dict_ts) in enumerate(cf_dicts_ts.items()):
    hourly_res = tu.get_frequency_resolution_hours(cf_dict_ts)
    res = res if res != 'daily' else '0.25'
    window = int(48/hourly_res)  # 8*6 = 48 hours

    title = f'{res}°, {hourly_res}h resolution'

    ts_country = tu.rolling_timemean(cf_dict_ts,
                                     window=window,
                                     center=True,
                                     fill_lims=False,
                                     )
    ts_country = tu.get_time_range_data(ds=ts_country,
                                        time_range=time_range)

    gplt.plot_2d(x=ts_country['time'], y=ts_country.data,
                 ax=im['ax'][0],
                 label=title,
                 lw=4 if res == gt else 2.5,
                 ls='--' if res == gt else '-',
                 color='black' if res == gt else gplt.colors[idx+5],
                 ylabel='Capacity Factor [a.u.]',
                 alpha=1.0 if res == gt else 0.8,
                 rot=45,)

gplt.plot_hline(y=0.06,
                ax=im['ax'][0],
                lw=2,
                ls='dotted',
                color='grey',
                label='Threshold 0.06',
                loc='outside')

savepath_cfs = f"{config['plot_dir']}/impact_downscaling/compare_res_time_series.png"

gplt.save_fig(savepath_cfs, fig=im['fig'])
# %%
# %%
# Local dunkelflauten Germany  3 rows
reload(gplt)
reload(cfu)
reload(tu)
sources = ['onwind', 'solar', ]
ncols = len(cf_dicts)
nrows = len(sources) + 2
im_cfs = gplt.create_multi_plot(
    nrows=nrows, ncols=ncols,
    hspace=-0,
    wspace=0.19,
    projection_arr=['PlateCarree']*ncols*(nrows-1) + [None],
    end_idx=(nrows-1)*ncols+1,
    full_length_row=True,
    end_row_idx=ncols - 2,
    pos_x=[-0.12]*(nrows-1)*ncols + [-0.02],
)
last_ax = im_cfs['ax'][-1]
last_ax.set_position([0.12, 0.05, 0.5, 0.2])

threshold = 0.02

lon_range_ger = [5, 16]
lat_range_ger = [47, 55.5]

for idx, (res, cf_dict_era5) in enumerate(cf_dicts.items()):
    hourly_res = tu.get_frequency_resolution_hours(
        cf_dict_era5['solar']['cf_ts'])
    res = res if res != 'daily' else '0.25'
    title = f'{res}°, {hourly_res}h resolution'
    cf_onwind_solar = cf_dict_era5['all']['cf_ts']
    cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
    gs, _, _ = sput.get_grid_step(cf_onwind_solar)
    mask = cf_mask if gs == fine_res else cf_mask_coarse
    for s, sname in enumerate(sources):
        sd, ed = tu.get_time_range(
            cf_dict_era5[sname]['cf_ts'], asstr=True, m=False, d=False)

        cap_fac = cf_dict_era5[sname]['cf']
        cap_fac = sput.check_dimensions(cap_fac)

        gplt.plot_map(cap_fac,
                      ax=im_cfs['ax'][s*ncols + idx],
                      title=title if s == 0 else None,
                      vertical_title=f'{sname} capacity factor \n{sd} - {ed}' if idx == 0 else None,
                      y_title=1.2,
                      #   mask=mask,
                      cmap='cmo.thermal',
                      #   levels=25,
                      tick_step=5,
                      vmin=0.08 if sname == 'solar' else 0.0,
                      vmax=.14 if sname == 'solar' else 0.4,
                      lon_range=lon_range_ger,
                      lat_range=lat_range_ger,
                      plot_borders=True,
                      orientation='vertical',
                      label='Capacity Factor [a.u.]' if idx == ncols -
                      1 else None,
                      )
    quantile = 0.03
    num_hours = 48
    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    ts = cf_dict_era5['all']['cf_winter_ts'] if res != 'DS BC 0.25' else cf_dicts['GT 0.25']['all']['cf_winter_ts']
    cf_ts_mean = tu.rolling_timemean(
        ts, window=window)
    cap_fac_qt = cf_ts_mean.quantile(q=quantile, dim='time')
    cap_fac_qt = sput.check_dimensions(cap_fac_qt)
    risk = cap_fac_qt
    im = gplt.plot_map(
        risk,
        ax=im_cfs['ax'][ncols*len(sources) + idx],
        vertical_title=r'Low capacity factors (Q$_{{0.05}}$)' +
        f'\nNov - Jan {sd}-{ed}' if idx == 0 else None,
        y_title=1.2,
        cmap='cmo.oxy_r',
        # mask=mask,
        # levels=25,
        tick_step=5,
        vmin=0.02,
        vmax=0.06,
        lon_range=lon_range_ger,
        lat_range=lat_range_ger,
        plot_borders=True,
        orientation='vertical',
        label=r'Quantile$_{0.05}$(CF) [a.u.]' if idx == ncols - 1 else None,
    )

gt = f'GT {fine_res}'
for idx, (res, cf_dict_ts) in enumerate(cf_dicts_ts.items()):
    hourly_res = tu.get_frequency_resolution_hours(cf_dict_ts)
    res = res if res != 'daily' else '0.25'
    window = int(48/hourly_res)  # 8*6 = 48 hours

    title = f'{res}°, {hourly_res}h resolution'

    ts_country = tu.rolling_timemean(cf_dict_ts,
                                     window=window,
                                     center=True,
                                     fill_lims=False,
                                     )
    ts_country = tu.get_time_range_data(ds=ts_country,
                                        time_range=time_range)

    gplt.plot_2d(x=ts_country['time'], y=ts_country.data,
                 ax=im_cfs['ax'][-1],
                 label=title,
                 lw=4 if res == gt else 2.5,
                 ls='--' if res == gt else '-',
                 color='black' if res == gt else gplt.colors[idx+5],
                 ylabel='Capacity Factor [a.u.]',
                 alpha=1.0 if res == gt else 0.8,
                 rot=45,)

gplt.plot_hline(y=0.06,
                ax=im_cfs['ax'][-1],
                lw=2,
                ls='dotted',
                color='grey',
                label='Threshold 0.06',
                loc='outside')


savepath_cfs = f"{config['plot_dir']}/impact_downscaling/compare_res.png"

gplt.save_fig(savepath_cfs, fig=im_cfs['fig'])


# %%
# Local dunkelflauten Germany  3 columns
reload(gplt)
reload(cfu)
reload(tu)
sources = ['onwind', 'solar', ]
nrows = len(cf_dicts)+1
nrows = len(cf_dicts)
ncols = len(sources) + 1
im_cfs = gplt.create_multi_plot(
    nrows=nrows, ncols=ncols,
    wspace=0.1,
    hspace=0.25,
    projection='PlateCarree',
    # projection_arr=['PlateCarree']*ncols*(nrows-1) + [None],
    # end_idx=(nrows-1)*ncols+1,
    full_length_row=True,
    # end_row_idx=ncols - 1,
    # figsize=()
    )
# last_ax = im_cfs['ax'][-1]
# last_ax.set_position([0.15, 0.07, 0.5, 0.1])
# threshold = 0.02
lon_range_ger = [5, 16]
lat_range_ger = [47, 55.5]

for idx, (res, cf_dict_era5) in enumerate(cf_dicts.items()):
    hourly_res = tu.get_frequency_resolution_hours(
        cf_dict_era5['solar']['cf_ts'])
    res = res if res != 'daily' else '0.25'
    title = f'{res}°, {hourly_res}h resolution'
    cf_onwind_solar = cf_dict_era5['all']['cf_ts']
    cf_onwind_solar = sput.rename_dims(cf_onwind_solar)
    gs, _, _ = sput.get_grid_step(cf_onwind_solar)
    mask = cf_mask if gs == fine_res else cf_mask_coarse
    for s, sname in enumerate(sources):
        sd, ed = tu.get_time_range(
            cf_dict_era5[sname]['cf_ts'], asstr=True, m=False, d=False)

        cap_fac = cf_dict_era5[sname]['cf']
        cap_fac = sput.check_dimensions(cap_fac)

        gplt.plot_map(cap_fac,
                      ax=im_cfs['ax'][idx*ncols + s],
                      vertical_title=title if s == 0 else None,
                      title=f'{sname} capacity factor \n{sd} - {ed}' if idx == 0 else None,
                      y_title=1.2,
                      #   mask=mask,
                      cmap='cmo.thermal',
                      #   levels=25,
                      tick_step=5,
                      vmin=0.08 if sname == 'solar' else 0.0,
                      vmax=.14 if sname == 'solar' else 0.4,
                      lon_range=lon_range_ger,
                      lat_range=lat_range_ger,
                      plot_borders=True,
                      orientation='horizontal',
                      label='CF [a.u.]' if idx == nrows -
                      1 else None,
                      )

    quantile = 0.03
    num_hours = 48
    window = int(num_hours / hourly_res)  # 8*6 = 48 hours
    ts = cf_dict_era5['all']['cf_winter_ts'] if res != 'DS BC 0.25' else cf_dicts['GT 0.25']['all']['cf_winter_ts']
    cf_ts_mean = tu.rolling_timemean(
        ts, window=window)
    cap_fac_qt = cf_ts_mean.quantile(q=quantile, dim='time')
    cap_fac_qt = sput.check_dimensions(cap_fac_qt)
    risk = cap_fac_qt
    im = gplt.plot_map(
        risk,
        ax=im_cfs['ax'][idx*ncols + ncols - 1],
        title=r'Low capacity factors (Q$_{{0.05}}$)' +
        f'\nNov - Jan {sd}-{ed}' if idx == 0 else None,
        y_title=1.2,
        cmap='cmo.oxy_r',
        # mask=mask,
        # levels=25,
        tick_step=5,
        vmin=0.02,
        vmax=0.06,
        lon_range=lon_range_ger,
        lat_range=lat_range_ger,
        plot_borders=True,
        orientation='horizontal',
        label=r'Q$_{0.05}$ (CF) [a.u.]' if idx == nrows - 1 else None,
    )

# gt = f'GT {fine_res}'
# for idx, (res, cf_dict_ts) in enumerate(cf_dicts_ts.items()):
#     hourly_res = tu.get_frequency_resolution_hours(cf_dict_ts)
#     res = res if res != 'daily' else '0.25'
#     window = int(48/hourly_res)  # 8*6 = 48 hours

#     title = f'{res}°, {hourly_res}h resolution'

#     ts_country = tu.rolling_timemean(cf_dict_ts,
#                                      window=window,
#                                      center=True,
#                                      fill_lims=False,
#                                      )
#     ts_country = tu.get_time_range_data(ds=ts_country,
#                                         time_range=time_range)

#     gplt.plot_2d(x=ts_country['time'], y=ts_country.data,
#                  ax=im_cfs['ax'][-1],
#                  label=title,
#                  lw=4 if res == gt else 2.5,
#                  ls='--' if res == gt else '-',
#                  color='black' if res == gt else gplt.colors[idx+5],
#                  ylabel='Capacity Factor [a.u.]',
#                  alpha=1.0 if res == gt else 0.8,
#                  rot=45,)

# gplt.plot_hline(y=0.06,
#                 ax=im_cfs['ax'][-1],
#                 lw=2,
#                 ls='dotted',
#                 color='grey',
#                 label='Threshold 0.06',
#                 loc='outside')


savepath_cfs = f"{config['plot_dir']}/impact_downscaling/compare_res.png"

gplt.save_fig(savepath_cfs, fig=im_cfs['fig'])
