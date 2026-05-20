#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mgomezd1
"""

# %% Imports
import dask.array as da
from dask_ml import preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import warnings
import xarray as xr
from dask import optimize
import dask.array as da
import zarr as zr

# %% Classes
## TODO: Add docstrings, add type hints, add error handling
## TODO: add logging
## TODO: implement other scaling methods (e.g., min-max)


## TODO: Properly parallelize the standard scaler
class AI_StandardScaler(preprocessing.StandardScaler):
    # The fit function should expect dask arrays
    def fit(self, X, y=None, **kwargs):
        assert isinstance(X, da.Array), "X should be a dask array"
        num_workers = kwargs.get("num_workers", 1)
        mean = X.mean(axis=(0, -1, -2)).compute(
            num_workers=num_workers, scheduler="threads"
        )
        std = X.std(axis=(0, -1, -2)).compute(
            num_workers=num_workers, scheduler="threads"
        )
        self.mean_ = np.tile(mean.reshape(-1, 1, 1), (1, 241, 241))
        self.scale_ = np.tile(std.reshape(-1, 1, 1), (1, 241, 241))
        return self


class DaskDataset(Dataset):
    def __init__(self, AI_X, AI_scaler, base_int, base_scaler, target_data, **kwargs):
        # Assert that AI_X and base_int are dask arrays
        assert isinstance(AI_X, da.Array), "AI_X should be a dask array"
        # and that they have the same length
        assert (
            AI_X.shape[0] == base_int.shape[0]
        ), "AI_X and base_int should have the same length"

        # self.AI_X = AI_X
        self.AI_scaler = AI_scaler
        self.base_int = base_int
        self.base_scaler = base_scaler
        self.target_scaler = kwargs.get("target_scaler", None)
        self.target_data = target_data
        self.set_name = kwargs.get("set_name", "unnamed")

        self.chunk_size = kwargs.get("chunk_size", 256)

        # # Attempt at optimizing dask array
        # self.AI_X = optimize(self.AI_X.rechunk((self.chunk_size, 5, 241, 241)))[0]
        # self.base_int = optimize(self.base_int.rechunk((self.chunk_size, 2)))[0]

        # Attempt at optimizing dask array
        # self.AI_X = self.AI_X.rechunk((self.chunk_size, 5, 241, 241))
        # self.base_int = self.base_int.rechunk((self.chunk_size, 2))

        # Scale the data using the scaler using dask.delayed.ravel
        interim = (
            # self.AI_X.rechunk((self.chunk_size, self.AI_X.shape[1], 241, 241))
            AI_X.rechunk((self.chunk_size, AI_X.shape[1], 241, 241))
            .to_delayed()
            .ravel()
        )
        interim = [
            da.from_delayed(
                self.AI_scaler.transform(block),
                shape=(
                    # min(self.chunk_size, self.AI_X.shape[0] - i * self.chunk_size),
                    min(self.chunk_size, AI_X.shape[0] - i * self.chunk_size),
                    AI_X.shape[1],
                    241,
                    241,
                ),
                # dtype=self.AI_X.dtype,
                dtype=np.float32,
            )
            for i, block in enumerate(interim)
        ]
        # self.AI_X = da.concatenate(interim)
        AI_X = da.concatenate(interim)

        # self.AI_X = da.map_blocks(
        #     self.AI_scaler.transform, self.AI_X, dtype=np.float32, chunks=self.AI_X.chunksize)
        self.scalars = da.map_blocks(
            self.base_scaler.transform,
            self.base_int,
            dtype=np.float32,
            chunks=self.base_int.chunksize,
        ).compute()

        if "track" in kwargs:
            self.scalars = np.hstack([self.scalars, kwargs["track"]]).astype(np.float32)

        if "leadtimes" in kwargs:
            self.scalars = np.hstack([self.scalars, kwargs["leadtimes"]]).astype(
                np.float32
            )

        self.num_scalars = self.scalars.shape[1]

        self.target_data = da.map_blocks(
            self.target_scaler.transform,
            self.target_data,
            dtype=np.float32,
            chunks=self.target_data.chunksize,
        ).compute()

        if kwargs.get("load_into_memory", False):
            # self.AI_X = self.AI_X.compute()
            self.AI_X = AI_X.compute()
        else:
            zarr_name = kwargs.get("zarr_name", "unnamed")
            cachedir = kwargs.get("cachedir", os.path.join(os.getcwd(), "cache"))
            zarr_path = os.path.join(cachedir, f"{zarr_name}.zarr")

            if not os.path.exists(zarr_path):
                os.makedirs(zarr_path)

            # save AI_X to zarr
            AI_X_path = os.path.join(zarr_path, "AI_X")
            if not os.path.exists(AI_X_path):
                # self.AI_X.rechunk(
                AI_X.rechunk(
                    # (self.chunk_size, self.AI_X.shape[1], 241, 241)
                    (self.chunk_size, AI_X.shape[1], 241, 241)
                ).to_zarr(AI_X_path)
                # self.AI_X.to_zarr(AI_X_path)
            elif kwargs.get("overwrite", True):
                # overwrite
                # self.AI_X.to_zarr(AI_X_path, overwrite=True)
                AI_X.to_zarr(AI_X_path, overwrite=True)

            # load AI_X from zarr
            # self.AI_X = da.from_zarr(AI_X_path)
            self.AI_X = zr.open(AI_X_path, dtype=np.float32)

        self.device = kwargs.get(
            "device",
            torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"),
        )

    def __len__(self):
        return len(self.base_int)

    def __getitem__(self, idx, **kwargs):

        # # Load the specific data points needed for this index
        AI_sample = self.AI_X[idx]
        scalar_sample = self.scalars[idx]
        target_sample = self.target_data[idx]
        if scalar_sample.ndim == 1:
            scalar_sample = scalar_sample[None, :]

        AI_sample = AI_sample.compute(schedule="threads")
        scalar_sample = scalar_sample.compute(schedule="threads")
        target_sample = target_sample.compute(schedule="threads")

        sample = (AI_sample, scalar_sample, target_sample)

        output = []
        for array in sample:
            if isinstance(array, np.ndarray):
                array = torch.from_numpy(array.astype(np.float32)).to("cpu")
                output.append(array)
            elif isinstance(array, da.Array):
                array = torch.from_numpy(array.compute().astype(np.float32)).to("cpu")
                output.append(array)
            elif isinstance(array, zr.core.Array):
                array = torch.from_numpy(array[:].astype(np.float32)).to("cpu")
                output.append(array)
            else:
                output.append(array)
        return (*output,)
        # return sample


# %%


class ZarrDataset(Dataset):
    def __init__(self, AI_X, base_int, target_data, **kwargs):
        # Assert that AI_X and base_int have the same length
        assert (
            AI_X.shape[0] == base_int.shape[0]
        ), "AI_X and base_int should have the same length"

        self.AI_X = AI_X[:]
        self.base_int = base_int
        self.target_data = target_data
        self.chunk_size = kwargs.get("chunk_size", 1000)
        self.scalars = base_int
        if "track" in kwargs:
            self.scalars = da.hstack([self.scalars, kwargs["track"]]).astype(np.float32)

        if "leadtimes" in kwargs:
            self.scalars = da.hstack([self.scalars, kwargs["leadtimes"]]).astype(
                np.float32
            )
        self.scalars = self.scalars.compute()
        self.num_scalars = self.scalars.shape[1]

        self.device = kwargs.get(
            "device",
            torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"),
        )

    def __len__(self):
        return len(self.base_int)

    def __getitem__(self, idx):
        # Load the specific data points needed for this index
        AI_sample = self.AI_X[idx]
        scalar_sample = self.scalars[idx]
        target_sample = self.target_data[idx]
        if scalar_sample.ndim == 1:
            scalar_sample = scalar_sample[None, :]

        sample = (
            torch.tensor(
                (AI_sample.compute() if isinstance(AI_sample, da.Array) else AI_sample),
                dtype=torch.float32,
            ),
            torch.tensor(
                (
                    scalar_sample.compute
                    if isinstance(scalar_sample, da.Array)
                    else scalar_sample
                ),
                dtype=torch.float32,
            ),
            torch.tensor(
                (
                    target_sample.compute()
                    if isinstance(target_sample, da.Array)
                    else target_sample
                ),
                dtype=torch.float32,
            ),
            # torch.tensor(scalar_sample, dtype=torch.float32),
            # torch.tensor(target_sample, dtype=torch.float32),
        )

        return sample


class ZarrDatasetv2(Dataset):
    def __init__(self, AI_X, base_int, target_data, **kwargs):
        # Assert that AI_X and base_int have the same length
        assert (
            AI_X.shape[0] == base_int.shape[0]
        ), "AI_X and base_int should have the same length"

        self.AI_X = AI_X[:]
        self.base_int = base_int[:]
        self.target_data = target_data[:]
        self.scalars = base_int
        if "track" in kwargs:
            self.scalars = np.hstack([self.scalars, kwargs["track"]]).astype(np.float16)

        if "leadtimes" in kwargs:
            self.scalars = np.hstack([self.scalars, kwargs["leadtimes"]]).astype(
                np.float16
            )
        # self.scalars = self.scalars.compute()
        self.num_scalars = self.scalars.shape[1]

        self.device = kwargs.get(
            "device",
            torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"),
        )

    def __len__(self):
        return len(self.base_int)

    def __getitem__(self, idx):
        # Load the specific data points needed for this index
        AI_sample = self.AI_X[idx]
        scalar_sample = self.scalars[idx]
        target_sample = self.target_data[idx]
        if scalar_sample.ndim == 1:
            scalar_sample = scalar_sample[None, :]

        sample = (
            torch.tensor(
                (AI_sample.compute() if isinstance(AI_sample, da.Array) else AI_sample),
                dtype=torch.float16,
            ),
            torch.tensor(
                (
                    scalar_sample.compute
                    if isinstance(scalar_sample, da.Array)
                    else scalar_sample
                ),
                dtype=torch.float16,
            ),
            torch.tensor(
                (
                    target_sample.compute()
                    if isinstance(target_sample, da.Array)
                    else target_sample
                ),
                dtype=torch.float16,
            ),
            # torch.tensor(scalar_sample, dtype=torch.float32),
            # torch.tensor(target_sample, dtype=torch.float32),
        )

        return sample


# %% Functions
def make_dataloader(dataset, **kwargs):
    dataset
    return DataLoader(dataset, **kwargs)


def latlon_to_sincos(positions):
    lat = positions[:, 0]
    lon = positions[:, 1]
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    return np.stack(
        [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)], axis=1
    )


def uv_to_magAngle(data, u_idx, v_idx):
    # Replace u and v with magnitude and angle
    u = data[:, u_idx]
    v = data[:, v_idx]
    mag = da.sqrt(u**2 + v**2)
    angle = da.arctan2(v, u)

    data[:, u_idx] = mag
    data[:, v_idx] = angle
    return data
