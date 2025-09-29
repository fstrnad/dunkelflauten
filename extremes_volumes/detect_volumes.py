from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
from scipy.ndimage import label
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
from importlib import reload
reload(tu)
reload(gut)


def find_connected_volumes(data_array, min_size=2):
    """
    Identifies all connected volumes of 1 values in a 3D xarray.DataArray based on sharing two axes.
    Connections are defined by being adjacent along two of the three axes (lon, lat, time).

    Parameters:
    data_array (xr.DataArray): A 3D xarray.DataArray with dimensions (lon, lat, time) containing 0s and 1s.

    Returns:
    xr.DataArray: A new DataArray where each connected volume of 1s is assigned a unique label.
    """
    # Ensure the input array contains only 0 and 1 values
    assert set(np.unique(data_array)) <= {
        0, 1}, "DataArray must contain only 0 and 1 values"

    # Define the structure for the 26-neighborhood (3x3x3 cube with all 1s)
    structuring_element = np.ones((3, 3, 3), dtype=int)

    # Use scipy.ndimage's label function to identify connected components
    labeled_array, num_features = label(
        data_array, structure=structuring_element)

    gut.myprint(f"Number of connected components: {num_features}.")
    # Find the size of each labeled component
    component_sizes = np.bincount(labeled_array.ravel())
    sizes, counts = np.unique(component_sizes,
                              return_counts=True,
                              )
    gut.myprint(f'Ignore sizes < {min_size}')

    # Find the components that are larger than 1 voxel (ignore size == 1)
    if min_size < 2:
        raise ValueError(
            f"min_size should be at least 2 but is set to: {min_size}")
    valid_labels = np.where(component_sizes >= min_size)[0][1:]

    # Create a new labeled array where we relabel only valid components
    relabeled_array = np.zeros_like(labeled_array)

    # Relabel the valid components starting from 1
    new_label = 1
    for valid_label in tqdm(valid_labels):
        if valid_label == 0:
            continue  # Skip background
        relabeled_array[labeled_array == valid_label] = new_label
        new_label += 1

    # Convert the relabeled array back to a DataArray with the same coordinates and dimensions as the input
    relabeled_dataarray = xr.DataArray(relabeled_array,
                                       coords=data_array.coords,
                                       dims=data_array.dims)

    # Update the number of features to reflect the number of valid components
    num_valid_features = new_label - 1

    gut.myprint(f"Number of valid connected components: {num_valid_features}")
    rel_frac_valid = num_valid_features / num_features
    gut.myprint(f"Fraction of valid components: {rel_frac_valid:.2f}")

    return relabeled_dataarray, num_valid_features, component_sizes


def enforce_minimum_dimension_size(labeled_data: xr.DataArray,
                                   min_evts_per_step: int = 2,  # min number per time step
                                   summary: bool = True,
                                   verbose: bool = False) -> xr.DataArray:
    # Get the unique labels in the array (excluding background label 0)
    unique_labels = np.unique(labeled_data.values)
    unique_labels = unique_labels[unique_labels > 0]
    coords = gut.get_coords(labeled_data)
    timepoints = coords['time']
    if gut.get_dims(labeled_data) != ['time', 'lat', 'lon']:
        raise ValueError(
            "The DataArray dimension should be a 'time', 'lat', 'lon' dimension.")

    # Create a new array to store valid components
    valid_array = np.zeros_like(labeled_data.values)

    if summary:
        df = pd.DataFrame(columns=['label', 'num_events', 'start', 'end'])

    # Loop over each label and check its dimensions
    new_label = 1
    for label_val in tqdm(unique_labels):
        # Get the indices where the component is located
        component_mask = labeled_data.values == label_val
        time_indices, lat_indices, lon_indices = np.where(component_mask)
        tind, tind_cnts = np.unique(time_indices, return_counts=True)
        if len(tind) == 0:
            raise ValueError(f'No time indices found for label {label_val}')
        if len(tind) == 1:
            tp = tu.tp2str(timepoints[time_indices[0]])
            gut.myprint(
                f"Removing label {label_val} at for single time {tp}!")
            labeled_data.values[time_indices, lat_indices, lon_indices] = 0
        if len(tind) - 1 != tind[-1]-tind[0]:
            raise ValueError(f'Broken time indices: {tind}')

        # Check if every time point has at least min_size events, otherwise remove all days that do not have min_size events
        if np.any(tind_cnts < min_evts_per_step):

            # time indices with at least min_size events
            t_minsize = np.where(tind_cnts >= min_evts_per_step,
                                 True, False)
            if not np.any(t_minsize):
                continue

            # find longest connected stretch of indices
            connected = longest_connected_indices(
                array=t_minsize,
                ind_val=True
            )
            tind_new = tind[connected[0]: connected[-1]]

            # keep only tind_new components
            isin = np.isin(time_indices, tind_new)
            num_events = np.count_nonzero(isin)
            time_indices_new = time_indices[isin]
            lat_indices_new = lat_indices[isin]
            lon_indices_new = lon_indices[isin]
            valid_array[time_indices_new,
                        lat_indices_new,
                        lon_indices_new] = new_label

            tpstart = tu.tp2str(timepoints[time_indices_new[0]])
            tpend = tu.tp2str(timepoints[time_indices_new[-1]])
            lon_range = gut.get_min_max(coords['lon'][lon_indices_new])
            lat_range = gut.get_min_max(coords['lat'][lat_indices_new])
            td = tu.get_frequency_resolution(labeled_data)
            dates = tu.get_dates_in_range(tpstart, tpend,
                                          time_delta=td)
            if summary:
                gut.myprint(
                    f"label {
                        label_val}->{new_label} for {num_events} events for {tpstart} and {tpend}",
                    verbose=verbose)
                df = pd.concat(
                    [df, pd.DataFrame({'label': [new_label],
                                       'old_label': [int(label_val)],
                                       'num_events': [num_events],
                                       'start': [tpstart],
                                       'end': [tpend],
                                       'lon_range': [lon_range],
                                       'lat_range': [lat_range],
                                       'num_tps': [len(dates)],
                                       'dates': [dates]})],
                    ignore_index=True)

        else:
            valid_array[labeled_data == label_val] = new_label
        new_label += 1

    # Update the number of features to reflect the number of valid components
    num_valid_features = new_label - 1

    print(f"Number of valid connected components: {num_valid_features}")
    return_array = xr.DataArray(valid_array,
                                coords=labeled_data.coords,
                                dims=labeled_data.dims)
    if summary:
        df = df.sort_values(by='num_events', ascending=False)
        df = df.reset_index(drop=True)
        return return_array, df
    else:
        return return_array


def get_specific_volume(labeled_data, label_value):
    """
    Extracts the indices (lat, lon, time) of the specific connected volume based on the given label value.

    Parameters:
    labeled_data (xr.DataArray): The labeled DataArray where connected volumes have unique labels.
    label_value (int): The label value corresponding to the connected volume you want to extract.

    Returns:
    list of tuples: A list of (lat, lon, time) indices where the connected volume exists.
    """
    # Check if the label_value is valid (should be between 1 and the number of features)
    check_labeled_data(labeled_data, label_value)

    dims, time_dim, indices = get_indices(labeled_data, label_value)
    data_coords = gut.get_coords(labeled_data)
    # Convert the indices to coordinates
    coords = []
    for idx in indices:
        dim1 = data_coords[dims[0]][idx[0]]
        dim2 = data_coords[dims[1]][idx[1]]
        dim3 = data_coords[dims[2]][idx[2]]
        this_coords = [dim1, dim2, dim3]
        this_coords[time_dim] = tu.tp2str(this_coords[time_dim])
        coords.append(this_coords)

    return np.array(coords, dtype=object)


def get_all_volumes(labeled_array):

    # Flatten the labeled_data for easier analysis
    flattened = labeled_array.values.flatten()
    dims = gut.get_dims(labeled_array)
    # Count the volume for each label
    unique_labels, counts = np.unique(flattened, return_counts=True)

    # Create a DataFrame from the counts
    df = pd.DataFrame({'elements': counts}, index=unique_labels)
    # Remove the background label (usually 0)
    df = df[df.index > 0]
    unique_labels = unique_labels[unique_labels > 0]
    for index in tqdm(unique_labels):
        volume = get_specific_volume(labeled_array, index)
        df.at[index, dims[0]] = volume[0][0]
        df.at[index, dims[1]] = volume[0][1]
        df.at[index, dims[2]] = volume[0][2]

    return df


def get_indices(labeled_data, label_value):
    dims = gut.get_dims(labeled_data)
    if len(dims) != 3:
        raise ValueError(
            "The input DataArray should have exactly 3 dimensions.")
    if 'time' not in dims:
        raise ValueError("The input DataArray should have a 'time' dimension.")
    time_dim = gut.get_index_array(dims, 'time')

    # Get a boolean mask where the labeled_data equals the label_value
    mask = labeled_data == label_value

    # Extract the lat, lon, and time indices where the mask is True
    indices = np.argwhere(mask.values)
    return dims, time_dim, indices


def check_labeled_data(labeled_data, label_value):
    if label_value < 1 or label_value > labeled_data.max().item():
        raise ValueError(
            # ojbect to allow for mixed types
            f"Invalid label_value: {label_value}. It should be between 1 and {labeled_data.max().item()}.")


def volume2ncfile(labeled_data, label_value,
                  collapse_time=True):
    """
    Save the labeled data to a netCDF file.

    Parameters:
    labeled_data (xr.DataArray): The labeled data to save.
    ncfile (str): The path to the netCDF file to save the data to.
    """
    check_labeled_data(labeled_data, label_value)
    volume = get_specific_volume(labeled_data, label_value)
    dims = gut.get_dims(labeled_data)
    time_dim = gut.get_index_array(dims, 'time')

    time_start, time_end = volume[0][time_dim], volume[-1][time_dim]

    vol_data = tu.get_time_range_data(ds=labeled_data,
                                      time_range=[time_start, time_end]
                                      )
    days = []
    this_day = vol_data.isel(time=0)
    this_day = xr.where(this_day == label_value, 0, np.nan)
    for day, date in enumerate(vol_data['time'].data):
        tmp_day = vol_data.sel(time=date)
        if collapse_time:
            tmp_day = xr.where(tmp_day == label_value, day, np.nan)
            this_day = xr.where(np.isnan(this_day), tmp_day, this_day)
        else:
            this_day = xr.where(tmp_day == label_value, day, np.nan)
            days.append(this_day)

    if collapse_time:
        vol_data = this_day
    else:
        vol_data = tu.merge_time_arrays(days)

    return vol_data


def longest_connected_indices(array, ind_val=True):
    """
    Find the indices of the longest contiguous stretch of ind_valt in a boolean numpy array.

    Parameters:
        array (numpy.ndarray): 1D array of boolean values (True/False).

    Returns:
        tuple: (start_index, end_index) for the longest period of False values.
               If no False values exist, returns (-1, -1).
    """
    if not isinstance(array, np.ndarray) or array.dtype != np.bool_:
        raise ValueError("Input must be a NumPy array of boolean values.")

    # check if any ind_val is in the array
    if not np.any(array == ind_val):
        gut.myprint(f"WARNING! No {ind_val} in array",
                    color='yellow')
        return None

    max_length = 0
    start_index = -1
    end_index = -1

    current_start = None

    for i, value in enumerate(array):
        if value == ind_val:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                current_length = i - current_start
                if current_length > max_length:
                    max_length = current_length
                    start_index = current_start
                    end_index = i
                current_start = None

    # Check at the end in case the longest False sequence ends at the last element
    if current_start is not None:
        current_length = len(array) - current_start
        if current_length > max_length:
            start_index = current_start
            end_index = len(array)

    return (start_index, end_index)
