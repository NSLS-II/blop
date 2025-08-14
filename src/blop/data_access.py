from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr


class DataAccess(ABC):
    @abstractmethod
    def get_data(self, data):
        """Retrieve data from a data source."""
        pass

    def data_validation(self, data):
        """Validate the retrieved data. Confirm that all key/value pairs are the same length."""
        data_length = {len(v) for v in data.values()}
        if len(data_length) > 1:
            raise ValueError(f"Data validation failed: The data has key value pairs of different lengths: {data_length}")
        pass

    @abstractmethod
    def convert_to_dictonary(self, data):
        """Convert the retrieved data into a dictionary format."""
        pass

    @abstractmethod
    def convert_back_from_dictonary(self, data_dict):
        """Convert the dictionary back to the original data format."""
        pass


class TiledDataAccess(DataAccess):
    def get_data(self, data, digestion_kwargs):
        """Retrieve data from a tiled database"""
        external_data = {}
        internal_data = data["streams"]["primary"].read()
        if "image_key" in digestion_kwargs:
            external_data = {key: data["streams"]["primary"][str(key)].read() for key in list(digestion_kwargs.values())}
        return internal_data, external_data

    def convert_to_dictonary(self, data, digestion_kwargs):
        """Convert xarray to a dictionary format."""
        internal_data, external_data = self.get_data(data, digestion_kwargs)
        dictionary = {}
        for var_name, data_array in internal_data.data_vars.items():
            dictionary[var_name] = data_array.values.flatten().tolist()
        if digestion_kwargs is not None:
            for key, value in external_data.items():
                dictionary[key] = value.astype(float)
        self.data_validation(dictionary)
        return dictionary

    def convert_back_from_dictonary(self, data_dict):
        """Convert a dictionary back to xarray format."""
        data_vars = {}
        updated_data = {key: value for key, value in data_dict.items() if "ts" not in key}
        for key, value in updated_data.items():
            if key in ["time"]:
                converted_value = pd.to_datetime(value, unit="s", origin="unix")
                data_vars[key] = xr.DataArray(converted_value)
            elif isinstance(value, (list, np.ndarray)) and isinstance(value[0], np.ndarray):
                value = np.array(value)
                var_dims = ("dim_0",) + tuple(f"{key}_dim{i}" for i in range(1, len(value.shape)))
                data_vars[key] = xr.DataArray(value, dims=var_dims)
            else:
                data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars)


class DatabrokerDataAccess(DataAccess):
    def get_data(self, data):
        """Retrieve data from a database based"""
        return data.table(fill=True)

    def convert_to_dictonary(self, data):
        """Convert database data to a dictionary format."""
        data = self.get_data(data)
        dictionary = {key: data[key].to_list() for key in data}
        self.data_validation(dictionary)
        return dictionary

    def convert_back_from_dictonary(self, data_dict):
        """Convert a dictionary back to database data format."""
        return pd.DataFrame(data_dict)
