import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

warnings.warn(
    "The data_access module is deprecated and will be removed in Blop v1.0.0. "
    "Data access should be done within your custom ``blop.protocols.EvaluationFunction``.",
    DeprecationWarning,
    stacklevel=2,
)


class DataAccess(ABC):
    @abstractmethod
    def get_data(self, uid: str) -> Any:
        """Retrieve data from the data source."""
        pass

    def data_validation(self, data: dict[str, Any]) -> None:
        """Validate the retrieved data. Confirm that all key/value pairs are the same length."""
        data_length = {len(v) for v in data.values()}
        if len(data_length) > 1:
            raise ValueError(f"Data validation failed: The data has key value pairs of different lengths: {data_length}")

    @abstractmethod
    def convert_data(self, data_dict: dict[str, Any]) -> Any:
        """Convert the dictionary back to the original data format."""
        pass


class TiledDataAccess(DataAccess):
    def __init__(self, data):
        self.data = data

    def get_data(self, uid: str) -> dict[str, Any]:
        """Retrieve data from a tiled database"""
        external_data = {}
        internal_data = self.data[uid]["primary"].base["internal"].read()
        for index in list(self.data[uid]["primary"].get_contents().keys())[1:]:
            external_data[index] = self.data[uid]["primary"][index].read()
        return self._convert_to_dictonary(internal_data, external_data)

    def _convert_to_dictonary(self, internal_data: dict[str, Any], external_data: dict[str, Any]) -> dict[str, Any]:
        """Convert xarray to a dictionary format."""
        dictionary = {**internal_data, **external_data}
        self.data_validation(dictionary)
        return dictionary

    def convert_data(self, new_data: dict[str, Any]) -> xr.Dataset:
        """Convert a dictionary back to xarray format."""
        data_vars = {}
        updated_data = {key: value for key, value in new_data.items() if "ts" not in key and "time" not in key}
        for key, value in updated_data.items():
            if isinstance(value, list | np.ndarray) and isinstance(value[0], np.ndarray):
                if len(value.shape) == 2:
                    value = np.expand_dims(value, axis=0)
                var_dims = ("dim_0",) + tuple(f"{key}_dim{i}" for i in range(1, len(value.shape)))
                data_vars[key] = xr.DataArray(value, dims=var_dims)
            else:
                data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars)


class DatabrokerDataAccess(DataAccess):
    def __init__(self, data):
        self.data = data

    def get_data(self, uid: str) -> dict[str, Any]:
        """Retrieve data from a database based"""
        table = self.data[uid].table(fill=True)
        return self._convert_to_dictonary(table)

    def _convert_to_dictonary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Convert database data to a dictionary format."""
        dictionary = {key: data[key].to_list() for key in data}
        self.data_validation(dictionary)
        return dictionary

    def convert_data(self, new_data: dict[str, Any]) -> pd.DataFrame:
        """Convert a dictionary back to a pandas DataFrame, handling scalars and arrays."""
        dataframe = pd.DataFrame()
        for key, value in new_data.items():
            if np.isscalar(value) or (isinstance(value, list | np.ndarray) and isinstance(value[0], np.ndarray)):
                dataframe[key] = pd.Series([value])
            else:
                dataframe[key] = pd.Series(value)
        return dataframe
