# %%
"""
Variables to analzye are the following:
2m_temperature t2m
forecast_surface_roughness fsr -> roughness
surface_net_solar_radiation ssr -> albedo
surface_solar_radiation_downwards ssrd -> albedo, influx_diffuse (=ssrd-fdir)
toa_incident_solar_radiation tisr -> influx_toa
total_sky_direct_solar_radiation_at_surface fdir -> influx_direct
"""
import pre_processing.workaround_fsr as wf
import geoutils.geodata.solar_radiation as sr
import geoutils.utils.met_utils as mut
import geoutils.cutouts.prepare_cutout as pc
import atlite.datasets.era5 as ald
import geoutils.bias_correction.quantile_mapping as qm
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
import os
from importlib import reload
import geoutils.countries.countries as cnt
import geoutils.countries.capacities as cap
import yaml
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if os.getenv("HOME") == '/home/ludwig/fstrnad80':
    data_dir = "/mnt/qb/work/ludwig/fstrnad80/data/processed_data/Europe/"
    output_dir = "/mnt/lustre/work/ludwig/shared_datasets/era5/"
    output_dir = "/mnt/qb/goswami/data/Europe/"
else:
    data_dir = "/home/strnad/data/climate_data/Europe"
    output_dir = "/home/strnad/data/climate_data/Europe"
    plot_dir = "/home/strnad/plots/dunkelflauten/variables"


requested_era5_flux = [
    "surface_net_solar_radiation",
    "surface_solar_radiation_downwards",
    "toa_incident_solar_radiation",
    "total_sky_direct_solar_radiation_at_surface",
]

requested_era5_temp = [
    "2m_temperature",
    "soil_temperature_level_4",
    "2m_dewpoint_temperature",
]

requested_era5_wind = [
    # "runoff",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    # "100m_u_component_of_wind",
    # "100m_v_component_of_wind",
]

requested_static_era5 = [
    'geopotential',
]

used_variables = [
    '2m_temperature',
    "forecast_surface_roughness"
] + requested_era5_flux + requested_era5_wind
gs = 0.5

country_name = 'Germany'
hourly_res = 6.0
files = []
for variable in used_variables:
    gut.myprint(f'Variable: {variable}')
    wb_file = f'{output_dir}/{country_name}_av/{gs}/{variable}_{gs}_{hourly_res}h.nc'
    if not fut.exist_file(wb_file):
        gut.myprint(f'File {wb_file} does not exist!')
        continue
    files.append(wb_file)

ds = of.open_nc_file(files,
                     verbose=False
                     )
# %%
locations = {'Berlin': (13.404954, 52.520008),
             'Alps': (10, 47.2),
             'North Sea': (7.5, 54.5),
             'Munich': (11.5761, 48.1371),
             'Colonge': (6.9603, 50.9375), }
locs = np.array(list(locations.values()))
lon_range, lat_range = sput.get_lon_lat_range(ds)
# %%
# Mean FSR and std FSR
short_time_range = ['1979-01-01', '2023-12-15']
variable = 'forecast_surface_roughness'

da_roughness = tu.get_time_range_data(ds[variable],
                                      time_range=short_time_range)

im = gplt.create_multi_plot(nrows=2, ncols=2,
                            projection_arr=['PlateCarree', 'PlateCarree',
                                            None, None],
                            hspace=0.4)
gplt.plot_map(da_roughness.mean(dim='time'),
              ax=im['ax'][0],
              title='Mean',
              cmap='Reds',
              levels=10,
              tick_step=5,
              vmin=0, vmax=1.2,
              plot_borders=True,
              label=f'Average {variable}',)

gplt.plot_map(da_roughness.std(dim='time'),
              ax=im['ax'][1],
              title='Standard deviation',
              cmap='Reds',
              levels=10,
              tick_step=5,
              vmin=0, vmax=.1,
              plot_borders=True,
              label=f'Std {variable}',)

ts_mean = da_roughness.mean(dim=['lon', 'lat'])
ts_std = da_roughness.std(dim=['lon', 'lat'])


gplt.plot_2d(x=ts_mean.time.values,
             y=ts_mean.values,
             ax=im['ax'][2],
             #  y_err=ts_std,
             title='Mean over locations',
             label='surface roughness',
             ylim=(0, 1),
             rot=90)

plot_data = ds[variable].values.flatten()

gplt.plot_hist(plot_data,
               ax=im['ax'][3],
               title='Distribution of values',
               color=gplt.colors[1],
               nbins=100,
               lw=1,
               alpha=1
               )


savepath = f'{plot_dir}/mean_{variable}_ts.png'
gplt.save_fig(savepath=savepath)
# %%
# FSR for different locations
short_time_range = ['2000-01-01', '2024-01-01']
fsr = tu.get_time_range_data(ds['forecast_surface_roughness'],
                             time_range=short_time_range)
im = gplt.create_multi_plot(nrows=2, ncols=3,
                            projection_arr=['PlateCarree', None,
                                            None, None, None, None],
                            hspace=0.5)

short_var_name = f'fsr'
gplt.plot_map(fsr.mean(dim='time'),
              ax=im['ax'][0],
              title=f'Mean FSR analytical',
              cmap='Greens',
              levels=15,
              tick_step=5,
              vmin=0, vmax=1.3,
              plot_borders=True,
              label=short_var_name,)
im_map = gplt.plot_map(locs, plot_type='points',
                       ax=im['ax'][0],
                       lon_range=lon_range,
                       lat_range=lat_range,
                       plot_borders=True,
                       size=10,)
for idx, (loc_name, loc) in enumerate(locations.items()):
    gplt.plot_text(ax=im_map['ax'],
                   text=loc_name,
                   xpos=loc[0],
                   ypos=loc[1]+.5)

    loc_data = sput.get_data_coordinate(fsr, loc)
    mean = loc_data.mean()
    gplt.plot_2d(x=[loc_data.time.values],
                 y=[loc_data.values],
                 title=f'{loc_name} - FSR',
                 ax=im['ax'][idx+1],
                 ylim=(-0.1, 1.3),
                 lw=1,
                 label='fsr',
                 #  color=gplt.colors[1],
                 rot=90
                 )
    gplt.plot_hline(ax=im['ax'][idx+1], y=mean,
                    label='mean', color='red')

savepath = f'{plot_dir}/single_locations_fsr_{short_time_range}.png'
gplt.save_fig(savepath=savepath)


# %%
# Analyze the fluxes

ds_fluxes = ds[requested_era5_flux]

im = gplt.create_multi_plot(nrows=2, ncols=2,
                            projection='PlateCarree',
                            hspace=0.5)

for idx, variable in enumerate(requested_era5_flux):
    da = ds_fluxes[variable].mean(dim='time')

    gplt.plot_map(da,
                  ax=im['ax'][idx],
                  title=f'Mean {variable}',
                  cmap='viridis',
                  levels=10,
                  tick_step=5,
                  #   vmin=0, vmax=1.2,
                  plot_borders=True,
                  label=f'{ds_fluxes[variable].attrs['short_name']}',)

savepath = f'{plot_dir}/compare_{requested_era5_flux}_spatial_mean.png'
gplt.save_fig(savepath=savepath)

im = gplt.create_multi_plot(nrows=2, ncols=2,
                            projection='PlateCarree',
                            hspace=0.5)

for idx, variable in enumerate(requested_era5_flux):
    da = ds_fluxes[variable].std(dim='time')
    gplt.plot_map(da,
                  ax=im['ax'][idx],
                  title=f'Std {variable}',
                  cmap='viridis',
                  levels=10,
                  tick_step=5,
                  #   vmin=0, vmax=1.2,
                  plot_borders=True,
                  label=f'{ds_fluxes[variable].attrs['short_name']}',)

savepath = f'{plot_dir}/compare_{requested_era5_flux}_spatial_std.png'
gplt.save_fig(savepath=savepath)

# %%
# Analyze the variables for different locations

for idx, variable in enumerate(requested_era5_flux):

    im = gplt.create_multi_plot(nrows=2, ncols=3,
                                projection_arr=['PlateCarree', None,
                                                None, None, None, None],
                                hspace=0.5)
    data = tu.get_time_range_data(ds[variable],
                                  time_range=short_time_range)
    short_var_name = f'{ds_fluxes[variable].attrs['short_name']}'
    gplt.plot_map(data.mean(dim='time'),
                  ax=im['ax'][0],
                  title=f'Mean {variable}',
                  cmap='Reds',
                  levels=15,
                  tick_step=5,
                  #   vmin=0, vmax=1.2,
                  plot_borders=True,
                  label=short_var_name,)
    im_map = gplt.plot_map(locs, plot_type='points',
                           ax=im['ax'][0],
                           lon_range=lon_range,
                           lat_range=lat_range,
                           plot_borders=True,
                           size=10,)
    for idx, (loc_name, loc) in enumerate(locations.items()):
        gplt.plot_text(ax=im_map['ax'],
                       text=loc_name,
                       xpos=loc[0],
                       ypos=loc[1]+.5)

        loc_data = sput.get_data_coordinate(data, loc)

        gplt.plot_2d(x=loc_data.time.values,
                     y=loc_data.values,
                     title=f'{loc_name} - {variable}',
                     ax=im['ax'][idx+1],
                     label=short_var_name,
                     lw=1,
                     color=gplt.colors[1],
                     rot=90
                     )

    savepath = f'{plot_dir}/single_locations_{variable}_{short_time_range}.png'
    gplt.save_fig(savepath=savepath)


# %%
# Compute TOA solar radiation analytically
reload(sr)

toa = sr.get_toa_incident_solar_radiation_for_xarray(
    data_array_like=ds['toa_incident_solar_radiation'],)
# %%
reload(gplt)
short_time_range = ['2023-01-01', '2024-01-01']

im = gplt.create_multi_plot(nrows=2, ncols=3,
                            projection_arr=['PlateCarree', None,
                                            None, None, None, None],
                            hspace=0.5)
data = tu.get_time_range_data(toa,
                              time_range=short_time_range)
data_era5 = tu.get_time_range_data(ds['toa_incident_solar_radiation'],
                                   time_range=short_time_range)
short_var_name = f'toa'
gplt.plot_map(data.mean(dim='time'),
              ax=im['ax'][0],
              title=f'Mean TOA analytical',
              cmap='Reds',
              levels=15,
              tick_step=5,
              #   vmin=0, vmax=1.2,
              plot_borders=True,
              label=short_var_name,)
im_map = gplt.plot_map(locs, plot_type='points',
                       ax=im['ax'][0],
                       lon_range=lon_range,
                       lat_range=lat_range,
                       plot_borders=True,
                       size=10,)
for idx, (loc_name, loc) in enumerate(locations.items()):
    gplt.plot_text(ax=im_map['ax'],
                   text=loc_name,
                   xpos=loc[0],
                   ypos=loc[1]+.5)

    loc_data = sput.get_data_coordinate(data, loc)
    loc_data_era5 = sput.get_data_coordinate(data_era5, loc)
    gplt.plot_2d(x=[loc_data.time.values, loc_data_era5.time.values],
                 y=[loc_data.values, loc_data_era5.values],
                 title=f'{loc_name} - TOA analytical',
                 ax=im['ax'][idx+1],
                 label=['analytical', 'era5'],
                 lw=1,
                 #  color=gplt.colors[1],
                 rot=90
                 )

savepath = f'{plot_dir}/single_locations_analytical_toa_{short_time_range}.png'
gplt.save_fig(savepath=savepath)


# %%
# Create correlation matrix

ds_fluxes = ds[requested_era5_flux]
correlation_matrices = {}
for var1 in ds_fluxes.data_vars:
    for var2 in ds_fluxes.data_vars:
        if var1 != var2:
            # Compute correlation along the 'time' dimension
            corr = xr.corr(ds[var1], ds[var2], dim='time')
            correlation_matrices[(var1, var2)] = corr
        else:
            correlation_matrices[(var1, var2)] = xr.ones_like(
                ds[var1].isel(time=0))

# %%
im = gplt.create_multi_plot(nrows=len(ds_fluxes.data_vars),
                            ncols=len(ds_fluxes.data_vars),
                            projection='PlateCarree',
                            hspace=0.5)
for idx, ((var1, var2), corr) in enumerate(correlation_matrices.items()):
    gplt.plot_map(
        corr,
        ax=im['ax'][idx],
        title=f'{var2}' if idx < len(ds_fluxes.data_vars) else None,
        vertical_title=f'{var1}' if idx % len(
            ds_fluxes.data_vars) == 0 else None,
        cmap='copper',
        levels=20,
        # leftcolor='white',
        tick_step=5,
        vmin=0.5, vmax=1,
        plot_borders=True,
        label='Correlation',)


savepath = f'{plot_dir}/correlation_fluxes.png'
gplt.save_fig(savepath=savepath)
# %%
# Analyze how the flux variables are used in atlite
reload(pc)
ds_influx = ds[requested_era5_flux + ['2m_temperature']]
ds_at = pc.prepare_cutout(ds=ds_influx, features=['influx'])
ds_at.load()
# %%
reload(at)
cutout_path = f'{data_dir}/{country_name}/pv_cutout.nc'
cutout_germany = at.Cutout(
    cutout_path,
    data=ds_at)
# %%
reload(sput)
pv_vars = ['influx', 'albedo', 'influx_diffuse',
           'influx_direct', 'influx_toa',]

short_time_range = ['2023-01-01', '2023-02-01']

for idx, variable in enumerate(pv_vars):

    im = gplt.create_multi_plot(nrows=2, ncols=3,
                                projection_arr=['PlateCarree', None,
                                                None, None, None, None],
                                hspace=0.5)
    data = tu.get_time_range_data(cutout_germany.data[variable],
                                  time_range=short_time_range)
    short_var_name = variable
    gplt.plot_map(data.mean(dim='time'),
                  ax=im['ax'][0],
                  title=f'Mean {variable}',
                  cmap='Reds',
                  levels=15,
                  tick_step=5,
                  #   vmin=0, vmax=1.2,
                  plot_borders=True,
                  label=short_var_name,)
    im_map = gplt.plot_map(locs, plot_type='points',
                           ax=im['ax'][0],
                           lon_range=lon_range,
                           lat_range=lat_range,
                           plot_borders=True,
                           size=10,)
    for idx, (loc_name, loc) in enumerate(locations.items()):
        gplt.plot_text(ax=im_map['ax'],
                       text=loc_name,
                       xpos=loc[0],
                       ypos=loc[1]+.5)

        loc_data = sput.get_data_coordinate(data, loc,
                                            lon_name='x',
                                            lat_name='y')

        gplt.plot_2d(x=loc_data.time.values,
                     y=loc_data.values,
                     title=f'{loc_name} - {variable}',
                     ax=im['ax'][idx+1],
                     label=short_var_name,
                     lw=1,
                     color=gplt.colors[1],
                     rot=90
                     )

    savepath = f'{plot_dir}/single_locations_pv_{variable}_{short_time_range}.png'
    gplt.save_fig(savepath=savepath)

# %%
panel = config['technology']['solar']['resource']['panel']
orientation = config['technology']['solar']['resource']['orientation']

cf_mean_influx_direct_diffuse = cutout_germany.pv(
    panel=panel,
    orientation=orientation,
    capacity_factor=True,
    use_influx=False,
    capacity_factor_timeseries=True)

cf_mean_influx = cutout_germany.pv(
    panel=panel,
    orientation=orientation,
    capacity_factor=True,
    use_influx=True,
    use_ground=True,
    capacity_factor_timeseries=True)


# %%
# compare capacity factors for different influx
reload(gplt)
diff = cf_mean_influx - cf_mean_influx_direct_diffuse

short_time_range = ['2023-01-01', '2024-02-01']


im = gplt.create_multi_plot(nrows=2, ncols=3,
                            projection_arr=[
                                'PlateCarree', None, None,
                                'PlateCarree', None, None],
                            hspace=0.5)
data = tu.get_time_range_data(diff,
                              time_range=short_time_range)
short_var_name = variable
gplt.plot_map(data.mean(dim='time'),
              ax=im['ax'][0],
              title=f'Mean Difference PV-CF',
              cmap='RdYlBu',
              levels=15,
              tick_step=5,
              vmin=-0.01, vmax=0.01,
              centercolor='white',
              plot_borders=True,
              label='Mean Diff CF',)
gplt.plot_map(data.std(dim='time'),
              ax=im['ax'][3],
              title=f'Std Difference PV-CF',
              cmap='Reds',
              levels=15,
              tick_step=5,
              vmin=0.0, vmax=0.03,
              leftcolor='white',
              plot_borders=True,
              label='Standard Deviation Diff CF',)

im_map = gplt.plot_map(locs, plot_type='points',
                       ax=im['ax'][0],
                       lon_range=lon_range,
                       lat_range=lat_range,
                       plot_borders=True,
                       size=10,)
for idx, (loc_name, loc) in enumerate(locations.items()):
    gplt.plot_text(ax=im_map['ax'],
                   text=loc_name,
                   xpos=loc[0],
                   ypos=loc[1]+.5)

    loc_data = sput.get_data_coordinate(data, loc,
                                        lon_name='x',
                                        lat_name='y')

    gplt.plot_2d(x=loc_data.time.values,
                 y=loc_data.values,
                 title=f'{loc_name} - Difference PV-CF',
                 ax=im['ax'][idx+1] if idx < 2 else im['ax'][idx+2],
                 label=short_var_name,
                 lw=1,
                 color=gplt.colors[1],
                 rot=90,
                 ylim=(-0.1, 0.1)
                 )
    if idx > 2:
        break
savepath = f'{plot_dir}/diff_cf_pv_{short_time_range}.png'
gplt.save_fig(savepath=savepath)


# %%
# Analzye the influence of constant fsr values on the capacity factor for wind
reload(wf)
ds_full_wind = ds[requested_era5_wind + ['forecast_surface_roughness']]
ds_small_wind = ds[requested_era5_wind]
ds_at_full = pc.prepare_cutout(ds=ds_full_wind,
                               features=['wind'])
fsr_const = wf.create_pseudo_fsr_data(gs=gs,
                                      data_array_like=ds_small_wind)
ds_small_wind['forecast_surface_roughness'] = fsr_const
ds_at_small = pc.prepare_cutout(ds=ds_small_wind,
                                features=['wind'])
# %%
ds_at_full.load()
ds_at_small.load()
# %%
reload(at)
cutout_path = f'{data_dir}/{country_name}/wind_cutout.nc'
cutout_germany_wind = at.Cutout(
    cutout_path,
    data=ds_at_full)
cutout_germany_wind_small = at.Cutout(
    cutout_path,
    data=ds_at_small)
# %%
wind = 'onwind'
turbine = config['technology'][wind]['resource']['turbine']

cap_wind_fsr_full = cutout_germany_wind.wind(
    turbine=turbine,
    capacity_factor=True,
    capacity_factor_timeseries=True,)

cap_wind_fsr_small = cutout_germany_wind_small.wind(
    turbine=turbine,
    capacity_factor=True,
    capacity_factor_timeseries=True,)
# %%
diff = cap_wind_fsr_full - cap_wind_fsr_small

short_time_range = ['2023-01-01', '2024-02-01']


im = gplt.create_multi_plot(nrows=2, ncols=3,
                            projection_arr=[
                                'PlateCarree', None, None,
                                'PlateCarree', None, None],
                            hspace=0.5)
data = tu.get_time_range_data(diff,
                              time_range=short_time_range)
short_var_name = variable
gplt.plot_map(data.mean(dim='time'),
              ax=im['ax'][0],
              title=f'Mean Difference PV-CF',
              cmap='RdYlBu',
              levels=15,
              tick_step=5,
              vmin=-0.02, vmax=0.02,
              centercolor='white',
              plot_borders=True,
              label='Mean Diff CF',)
gplt.plot_map(data.std(dim='time'),
              ax=im['ax'][3],
              title=f'Std Difference PV-CF',
              cmap='Reds',
              levels=15,
              tick_step=5,
              vmin=0.0, vmax=0.03,
              leftcolor='white',
              plot_borders=True,
              label='Standard Deviation Diff CF',)

im_map = gplt.plot_map(locs, plot_type='points',
                       ax=im['ax'][0],
                       lon_range=lon_range,
                       lat_range=lat_range,
                       plot_borders=True,
                       size=10,)
for idx, (loc_name, loc) in enumerate(locations.items()):
    gplt.plot_text(ax=im_map['ax'],
                   text=loc_name,
                   xpos=loc[0],
                   ypos=loc[1]+.5)

    loc_data = sput.get_data_coordinate(data, loc,
                                        lon_name='x',
                                        lat_name='y')

    gplt.plot_2d(x=loc_data.time.values,
                 y=loc_data.values,
                 title=f'{loc_name} - Difference PV-CF',
                 ax=im['ax'][idx+1] if idx < 2 else im['ax'][idx+2],
                 label=short_var_name,
                 lw=1,
                 color=gplt.colors[1],
                 rot=90,
                 ylim=(-0.05, 0.05)
                 )
    if idx > 2:
        break
savepath = f'{plot_dir}/diff_cf_wind_fsr_{short_time_range}.png'
gplt.save_fig(savepath=savepath)