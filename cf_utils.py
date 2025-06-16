import cmethods as cm
import geoutils.bias_correction.quantile_mapping as qm
import geoutils.geodata.solar_radiation as sr
import workaround_fsr as wf
import geoutils.utils.statistic_utils as sut
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.preprocessing.open_nc_file as of
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import atlite as at
from importlib import reload
import geoutils.countries.countries as cnt
import geoutils.cutouts.prepare_cutout as pc

# %%
reload(cnt)

reduced_era5_flux = [
    "surface_solar_radiation_downwards",
]

full_requested_era5_flux = [
    "surface_net_solar_radiation",
    "surface_solar_radiation_downwards",
    "toa_incident_solar_radiation",
    "total_sky_direct_solar_radiation_at_surface",
]

requested_era5_temp = [
    "2m_temperature",
    # "soil_temperature_level_4",
    # "2m_dewpoint_temperature",
]

requested_era5_wind = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    # "100m_u_component_of_wind",
    # "100m_v_component_of_wind",
    "forecast_surface_roughness",  # fsr
]

requested_static_era5 = [
    'geopotential',

]

feature_dict = {
    "wind": requested_era5_wind,
    "temperature": requested_era5_temp,
    "influx": reduced_era5_flux,
    "static": requested_static_era5,
    # 'runoff': ['runoff']
}


def compute_cutouts(ds, simple=True,
                    features=['influx', 'wind'],
                    savepath=None):

    ds_vars = gut.get_vars(ds)
    gs = sput.get_grid_step(ds)[0]
    ntimes = ds.sizes['time']

    if 'toa_incident_solar_radiation' not in ds_vars:
        reload(sr)
        gut.myprint(
            f'Computing toa_incident_solar_radiation  with grid step {gs} and time steps {ntimes}')
        toa = sr.get_toa_incident_solar_radiation_for_xarray(
            data_array_like=ds)
        ds['toa_incident_solar_radiation'] = toa

    if 'forecast_surface_roughness' not in ds_vars:
        reload(wf)
        gut.myprint(
            f'Computing fsr with grid step {gs} and time steps {ntimes}')
        fsr_const = wf.create_pseudo_fsr_data(gs=gs,
                                              data_array_like=ds)
        ds['forecast_surface_roughness'] = fsr_const

    ds = gut.add_era5_units(ds)
    ds_vars = gut.get_vars(ds)

    requested_era5_flux = reduced_era5_flux if simple else full_requested_era5_flux
    requested_vars = []
    for feature in features:
        if feature not in feature_dict:
            raise ValueError(
                f'Feature {feature} not in {feature_dict.keys()}!'
            )
        variables = feature_dict[feature]
        if feature == 'influx':
            requested_vars.extend(requested_era5_flux)
        else:
            requested_vars.extend(variables)

    for var in requested_vars:
        if var not in ds_vars:
            raise ValueError(
                f'Variable {var} requested variable but not in {ds_vars}!'
            )

    cutout = pc.prepare_cutout(
        ds=ds,
        features=features,)

    return cutout


def prepare_country_shapes(country_name):
    country_shape = cnt.get_country_shape(country_name)
    country_shape_off = cnt.get_country_shape(country_name, onshore=False)
    return country_shape, country_shape_off


def prepare_country_matrix(cutout, country_name, onshore=True,
                           as_xr=False):
    country_shape, country_shape_off = prepare_country_shapes(country_name)
    ind_matrix = cnt.cutout_country_cells(cutout,
                                          country_name=country_name,
                                          onshore=onshore,
                                          as_xr=as_xr)
    if onshore:
        return ind_matrix, country_shape
    else:
        country_shape_off = cnt.get_country_shape(country_name, onshore=False)
        return ind_matrix, country_shape_off

# Capacity factors Wind on shore


def compute_cf_dict(cutout_country, config,
                    sources=['offwind', 'onwind', 'solar'],
                    country_name='Germany',
                    correct_qm=True, qm_method='quantile_mapping',
                    ):
    cf_dict = dict()
    for source in sources:
        if source not in config['technology']:
            raise ValueError(
                f'Source {source} not in {config["technology"].keys()}!')

        cf_dict[source] = dict()

    for source in cf_dict.keys():
        gut.myprint(f'Computing capacity factors for {source}')
        ind_matrix, country_shape = prepare_country_matrix(
            cutout=cutout_country,
            country_name=country_name,
            onshore=False if source == 'offwind' else True)
        ind_matrix_xr, _ = prepare_country_matrix(
            cutout=cutout_country,
            country_name=country_name,
            onshore=False if source == 'offwind' else True,
            as_xr=True)

        # Capacity factors PV
        if source == 'solar':
            panel = config['technology']['solar']['resource']['panel']
            orientation = config['technology']['solar']['resource']['orientation']
            cf = cutout_country.pv(
                panel=panel,
                orientation=orientation,
                capacity_factor=True,
                capacity_factor_timeseries=False,
            )
            cf_ts = cutout_country.pv(
                panel=panel,
                orientation=orientation,
                capacity_factor=True,
                capacity_factor_timeseries=True,)

            power_generation_source = cutout_country.pv(
                shapes=country_shape,
                panel=panel,
                orientation=orientation,
                capacity_factor=True)

            cf_average_ts = cutout_country.pv(panel=panel,
                                              orientation=orientation,
                                              capacity_factor=False,
                                              layout=cf,
                                              per_unit=True,
                                              matrix=ind_matrix,
                                              )

        # Capacity factors Wind
        else:
            turbine = config['technology'][source]['resource']['turbine']
            cf = cutout_country.wind(
                turbine=turbine,
                capacity_factor=True)
            cf_ts = cutout_country.wind(
                turbine=turbine,
                capacity_factor=True,
                capacity_factor_timeseries=True)
            power_generation_source = cutout_country.wind(
                turbine=turbine, shapes=country_shape,
                capacity_factor=False)
            cf_average_ts = cutout_country.wind(turbine=turbine,
                                                capacity_factor=False,
                                                layout=cf,
                                                per_unit=True,
                                                matrix=ind_matrix
                                                )
        power_source = sput.remove_single_dim(power_generation_source,)
        cf_average_ts = sput.remove_single_dim(ds=cf_average_ts)
        cf_dict[source]['cf'] = cf
        cf_dict[source]['cf_ts'] = cf_ts
        cf_dict[source]['ts'] = xr.DataArray(cf_average_ts, name='cf')
        cf_dict[source]['power'] = power_source
        cf_dict[source]['matrix'] = ind_matrix
        cf_dict[source]['matrix_xr'] = ind_matrix_xr
        cf_dict[source]['shape'] = country_shape
        cf_dict[source]['weight'] = config['technology'][source]['weight']

    if correct_qm:
        gut.myprint(f'Correcting capacity factors with quantile mapping')
        country_initials = cnt.get_country_iso(country_name)
        opsd_dict = get_opsd(config, country=country_initials)
        cf_dict = correct_cf_dict_opsd(cf_dict, opsd_dict, method=qm_method)

    weight_arr = []
    ts_all = 0
    cap_all = 0
    for source in cf_dict.keys():
        weight = cf_dict[source]['weight']
        ts = cf_dict[source]['ts']
        cap_all += cf_dict[source]['cf'] * weight
        ts_all += ts * weight
        weight_arr.append(weight)

    sum_weights = np.sum(weight_arr)
    print(f'Sum of weights: {sum_weights}')
    ts_all = ts_all / sum_weights
    cap_all = cap_all / sum_weights
    cf_dict['all'] = {'ts': ts_all}
    cf_dict['all']['cf'] = cap_all
    cf_dict['all']['weights'] = weight_arr

    if 'offwind' in cf_dict.keys() and 'onwind' in cf_dict.keys():
        # Combine onshore and offshore wind
        gut.myprint('Combining onshore and offshore wind')
        ts_offwind = cf_dict['offwind']['ts']
        ts_onwind = cf_dict['onwind']['ts']
        ts_wind = config['technology']['offwind']['weight'] * ts_offwind + \
            config['technology']['onwind']['weight'] * ts_onwind

        cf_dict['wind'] = {'ts': ts_wind,
                           'weights':
                           [config['technology']['offwind']['weight'],
                            config['technology']['onwind']['weight']],
                           }

    return cf_dict


def get_opsd(config, country='DE', hourly_res=6):
    # OPSD Data
    opsd_df = pd.read_csv(config['data']['opsd'])

    opsd_times = opsd_df['cet_cest_timestamp']
    opsd_onwind_cap = opsd_df[f'{country}_wind_onshore_capacity']
    opsd_onwind_gen = opsd_df[f'{country}_wind_onshore_generation_actual']
    opsd_offwind_cap = opsd_df[f'{country}_wind_offshore_capacity']
    opsd_offwind_gen = opsd_df[f'{country}_wind_offshore_generation_actual']
    opsd_pv_cap = opsd_df[f'{country}_solar_capacity']
    opsd_pv_gen = opsd_df[f'{country}_solar_generation_actual']

    # Transform to xarray
    reload(tu)
    opsd_onwind_cap_xr = tu.pandas2xr_ts(opsd_onwind_cap, times=opsd_times)
    opsd_onwind_gen_xr = tu.pandas2xr_ts(opsd_onwind_gen, times=opsd_times)
    opsd_offwind_cap_xr = tu.pandas2xr_ts(opsd_offwind_cap, times=opsd_times)
    opsd_offwind_gen_xr = tu.pandas2xr_ts(opsd_offwind_gen, times=opsd_times)
    opsd_pv_cap_xr = tu.pandas2xr_ts(opsd_pv_cap, times=opsd_times)
    opsd_pv_gen_xr = tu.pandas2xr_ts(opsd_pv_gen, times=opsd_times)

    opsd_onwind_cf = opsd_onwind_gen_xr / opsd_onwind_cap_xr
    opsd_offwind_cf = sut.normalize(opsd_offwind_gen_xr / opsd_offwind_cap_xr)
    opsd_pv_cf = opsd_pv_gen_xr / opsd_pv_cap_xr

    opsd_onwind_cf_6h = tu.convert_time_resolution(
        opsd_onwind_cf, keep_time_points=hourly_res,
        freq='h', average=False,
        start_index=1)
    opsd_offwind_cf_6h = tu.convert_time_resolution(
        opsd_offwind_cf, keep_time_points=hourly_res,
        freq='h', average=False,
        start_index=1)
    opsd_pv_cf_6h = tu.convert_time_resolution(
        opsd_pv_cf, keep_time_points=hourly_res,
        freq='h', average=False,
        start_index=1)

    opsd_onwind_cf_6h = tu.remove_nans(opsd_onwind_cf_6h)
    opsd_offwind_cf_6h = tu.remove_nans(opsd_offwind_cf_6h)
    opsd_pv_cf_6h = tu.remove_nans(opsd_pv_cf_6h)

    opsd_dict = {'onwind': xr.DataArray(opsd_onwind_cf_6h, name='cf'),
                 'offwind': xr.DataArray(opsd_offwind_cf_6h, name='cf'),
                 'solar': xr.DataArray(opsd_pv_cf_6h, name='cf')}

    return opsd_dict


# def cf_qm(ts_model, ts_obs, method='basic_quantile'):
#     gut.myprint(f'Quantile mapping {method} for time series')
#     QM = qm.BiasCorrection(obs_data=ts_obs,
#                            mod_data=ts_model)
#     ts_model_corr = QM.correct(method=method)
#     return ts_model_corr


def cf_qm(ts_model, ts_obs,
          method='quantile_mapping', kind='+',
          name='cf'):
    gut.myprint(f'Quantile mapping {method} for time series')
    ts_obs = xr.DataArray(ts_obs, name=name)
    ts_full = xr.DataArray(ts_model, name=name)
    ts_obs, hist = tu.equalize_time_points(ts_obs, ts_full)
    if len(ts_obs) == 0:
        return None
    ts_model_corr = cm.adjust(
        method=method,
        obs=ts_obs,
        simh=hist,
        simp=ts_full,
        n_quantiles=1000,
        kind=kind,
    )
    return ts_model_corr[name]


def correct_cf_dict_opsd(cf_dict, opsd_dict,
                         method='quantile_mapping'):
    cf_dict_corr = cf_dict.copy()
    for source in cf_dict.keys():
        if source == 'all':
            continue
        ts_model = cf_dict[source]['ts']
        ts_obs = opsd_dict[source]
        cf_dict_corr[source]['opsd'] = ts_obs
        ts_model_corr = cf_qm(ts_model, ts_obs, method=method)
        if ts_model_corr is None:
            gut.myprint(
                f'No common time points!')
            continue
        else:
            cf_dict_corr[source]['ts_uncorr'] = ts_model
            cf_dict_corr[source]['ts'] = ts_model_corr
    return cf_dict_corr


def combined_cf_maps(cf_dict, sources=['onwind', 'solar']):
    sum_weights = 0
    for source in sources:
        if source not in cf_dict:
            raise ValueError(
                f'Source {source} not in {cf_dict.keys()}!')
        sum_weights += cf_dict[source]['weight']
    cf_combined = 0
    for source in sources:
        cf_combined += cf_dict[source]['weight'] * cf_dict[source]['cf_ts']
    cf_combined = cf_combined / sum_weights

    return cf_combined