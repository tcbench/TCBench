# Climatology Baseline Models
# Date: 2024.05.24
# This file contains the implementation of baseline models for TCBench.
# Author: mgomezd1
# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da
import pandas as pd

# ML Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.pipeline import make_pipeline

# Backend Libraries
import joblib as jl

from utils import toolbox, constants, ML_functions as mlf
from utils import data_lib as dlib


# %% Utility Functions
def PolynomialRegression(degree=3, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def exponential_regressor(x, y):
    # convert to and ln(y)
    y = np.log(y)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(x[:, np.newaxis], y)

    # Get the parameters of the linear regression model
    slope = model.coef_[0]
    intercept = model.intercept_

    return lambda x: np.exp(intercept) * np.exp(slope * x)


def nearest_interpolator(values, df):
    # Get the first and second columns of the DataFrame
    first_col = df.iloc[:, 0].to_numpy()
    second_col = df.iloc[:, 1].to_numpy()

    # Initialize an empty list to store the results
    results = []

    # For each value in the input array, find the nearest value in the first column
    for value in values:
        diff = np.abs(first_col - value)
        idx = np.argmin(diff)
        nearest_value = second_col[idx]
        results.append(nearest_value)

    return np.array(results)


def plot_facecolors(fig, axes, **kwargs):
    figcolor = kwargs.get("figcolor", np.array([1, 21, 38]) / 255)
    axcolor = kwargs.get("axcolor", np.array([141, 166, 166]) / 255)
    textcolor = kwargs.get("textcolor", np.array([242, 240, 228]) / 255)

    # Change the facecolor of the figure
    fig.set_facecolor(figcolor)

    # Change the facecolor of the axes
    for ax in axes.flatten():
        ax.set_facecolor(axcolor)

    # Change the color of the title
    for ax in axes.flatten():
        title = ax.get_title()
        ax.set_title(title, color=textcolor)

    # Change the color of the x and y axis labels
    for ax in axes.flatten():
        ax.xaxis.label.set_color(textcolor)
        ax.yaxis.label.set_color(textcolor)

    # Change the color of the tick labels
    for ax in axes.flatten():
        for label in ax.get_xticklabels():
            label.set_color(textcolor)
        for label in ax.get_yticklabels():
            label.set_color(textcolor)


# %%
class DeltaIntensityClimatology(BaseEstimator):
    """
    Intensity climatology baseline model.
    This model predicts the intensity climatology based on the maximum wind speed
    in the training data. In order to calculate this, the model calculates the mean
    and standard deviation of the intensification rate between time steps in the
    training set of storms.
    """

    def __repr__(self):
        return f"TCBench Climatology Baseline - {self.name}"

    def __init__(self):
        self.name = "Delta Intensity Climatology"

    def fit(self, X, y=None):
        # Assert that the training set is a dictionary with numeric keys
        # that each point to a list of storm objects
        assert isinstance(X, dict), "Training data must be a dictionary."
        # assert all(
        #     [isinstance(key, int) for key in X.keys()]
        # ), "Keys of the training data must be integers."
        assert all(
            [isinstance(value, list) for value in X.values()]
        ), "Values of the training data must be lists."
        assert all(
            [
                all([isinstance(storm, toolbox.tc_track) for storm in value])
                for value in X.values()
            ]
        ), "Values of the training data must be lists of TCBench TC Tracks."

        # Concatenate the list of storms into a single list
        storm_list = []
        for key in X.keys():
            storm_list += X[key]

        records = []
        for storm in storm_list:
            # Make a boolean index for the values of storm.wind that are a numeric string
            numeric_bool = np.char.isnumeric(storm.wind.astype(str))

            # Make a boolean masks for the timestamps in 0, 6, 12, and 18 h
            time_bool = np.isin(pd.to_datetime(storm.timestamps).hour, [0, 6, 12, 18])

            # Copy the wind
            wind = storm.wind.copy()
            wind[~numeric_bool] = np.NaN
            wind = wind.astype(float)
            wind = wind[time_bool]

            # Calculate the intensification rate
            base_intensity = wind[:-1]
            intensification_rate = np.diff(wind)

            # Filter out the NaN values
            intensification_rate = intensification_rate[~np.isnan(base_intensity)]
            base_intensity = base_intensity[~np.isnan(base_intensity)]

            base_intensity = base_intensity[~np.isnan(intensification_rate)]
            intensification_rate = intensification_rate[~np.isnan(intensification_rate)]

            records.append(np.vstack([base_intensity, intensification_rate]).T)

        # Concatenate the records
        records = np.vstack(records)

        self.intensity_climatology = None
        unique_intensities = np.unique(records[:, 0])

        self.training_data = records

        # intensity_distribution = pd.DataFrame(
        #     np.vstack([unique_intensities, np.zeros(unique_intensities.shape[0])]).T,
        #     columns=["Intensity", "Count"],
        # )

        for intensity in unique_intensities:
            # Calculate the mean and standard deviation of the intensification rate
            mean = np.mean(records[records[:, 0] == intensity, 1])
            std = np.std(records[records[:, 0] == intensity, 1])

            # intensity_distribution.loc[intensity] += records[
            #     records[:, 0] == intensity
            # ].shape[0]

            # Save the mean and standard deviation using a pandas dataframe
            temp_df = pd.DataFrame(
                [[intensity, mean, std]],
                columns=[
                    "Intensity",
                    "Mean Intensification Rate",
                    "Sigma Intensification Rate",
                ],
            )
            if self.intensity_climatology is None:
                self.intensity_climatology = temp_df
            else:
                self.intensity_climatology = pd.concat(
                    [
                        self.intensity_climatology,
                        temp_df,
                    ]
                )

        self.mu_interpolator = lambda x: nearest_interpolator(
            x, self.intensity_climatology[["Intensity", "Mean Intensification Rate"]]
        )

        # find the intensities and sigma intensification rates fir which the
        # sigma is greater than 2.5, half of the "normal" bin size for intensity
        # reporting
        sigma_mask = self.intensity_climatology["Sigma Intensification Rate"] > 2.5

        # make a 3rd degree polynomial regression model for the sigma intensification rate
        self.sigma_regressor = PolynomialRegression(degree=4)
        self.sigma_regressor.fit(
            self.intensity_climatology["Intensity"][sigma_mask]
            .to_numpy()
            .reshape(-1, 1),
            self.intensity_climatology["Sigma Intensification Rate"][
                sigma_mask
            ].to_numpy(),
        )

        # self.sigma_regressor = exponential_regressor(
        #     self.intensity_climatology["Intensity"][sigma_mask].to_numpy(),
        #     self.intensity_climatology["Sigma Intensification Rate"][
        #         sigma_mask
        #     ].to_numpy(),
        # )

    def summary(self, **kwargs):
        assert self.intensity_climatology is not None, "Model has not been trained."
        assert (
            self.training_data is not None
        ), "Model training record not found - did you use the fit method?"

        fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
        int_bins = np.arange(
            self.training_data[:, 0].min(), self.training_data[:, 0].max(), 5
        )
        rate_bins = (
            np.arange(self.training_data[:, 1].min(), self.training_data[:, 1].max(), 5)
            - 2.5
        )
        axes[0, 0].hist(
            self.training_data[:, 0],
            bins=int_bins,  # bins=np.unique(self.training_data[:, 0])
            density=True,
        )
        axes[0, 0].set_title("Intensity Distribution")
        axes[0, 0].set_xlabel("Intensity (kt)")
        axes[0, 0].set_ylabel("Density")

        # ------------------------------------------------
        axes[0, 1].scatter(
            self.training_data[:, 0], self.training_data[:, 1], alpha=0.1
        )
        axes[0, 1].set_title("Intensification Rate vs. Intensity")
        axes[0, 1].set_xlabel("Intensity (kt)")
        axes[0, 1].set_ylabel("6h Intensification Rate (kt)")

        axes[1, 0].hist(
            self.training_data[:, 1],
            bins=rate_bins,  # bins=np.unique(self.training_data[:, 1])
            density=True,
        )
        axes[1, 0].set_title("Intensification Rate Distribution")
        axes[1, 0].set_xlabel("6h Intensification Rate (kt)")
        axes[1, 0].set_ylabel("Density")

        # ------------------------------------------------
        axes[1, 1].hist2d(
            self.training_data[:, 0],
            self.training_data[:, 1],
            bins=[
                int_bins,
                rate_bins,
            ],
            density=True,
            cmap="Greens",
        )
        axes[1, 1].set_title("Joint Distribution")
        axes[1, 1].set_xlabel("Intensity (kt)")
        axes[1, 1].set_ylabel("6h Intensification Rate (kt)")

        # Colorize the plot
        plot_facecolors(fig, axes)

        # Show figure
        plt.tight_layout()
        plt.show()
        plt.close()

        # ------------------------------------------------

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=200)

        # Plot the intensity vs mean intensification rate
        axes[0].scatter(
            self.intensity_climatology["Intensity"],
            self.intensity_climatology["Mean Intensification Rate"],
            label="Mean Intensification Rate",
        )
        axes[0].set_title("Intensity vs. Mean Intensification Rate")
        axes[0].set_xlabel("Intensity")
        axes[0].set_ylabel("Mean Intensification Rate")

        axlineopts = kwargs.get(
            "axlineopts", {"color": "black", "linewidth": 2, "linestyle": "--"}
        )

        # draw the horizontal axis
        axes[0].axhline(0, **axlineopts)
        axes[0].legend()

        # Plot the intensity vs sigma intensification rate
        axes[1].scatter(
            self.intensity_climatology["Intensity"],
            self.intensity_climatology["Sigma Intensification Rate"],
            label="Sigma Intensification Rate",
        )

        xmin = self.intensity_climatology["Intensity"].min()
        xmax = self.intensity_climatology["Intensity"].max() + 10
        x = np.linspace(xmin, xmax, 300).reshape(-1, 1)

        # Plot the sigma intensification rate regressor
        axes[1].plot(
            x,
            self.sigma_regressor.predict(x),
            label="Sigma Intensification Rate Regressor",
        )

        axes[1].set_title("Intensity vs. Sigma Intensification Rate")
        axes[1].set_xlabel("Intensity")
        axes[1].set_ylabel("Sigma Intensification Rate")
        axes[1].axhline(0, **axlineopts)
        axes[1].legend()

        # Colorize the plot
        plot_facecolors(fig, axes)

        # Show figure
        plt.tight_layout()
        plt.show()
        plt.close()

        # ------------------------------------------------

        # plot uncertainty regions
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=200)

        x = np.arange(10, 190, 5).reshape(-1, 1)
        data = self.predict(x)

        ax.plot(x, data[:, 0], label="Mean Intensification Rate")
        ax.fill_between(
            x.flatten(),
            data[:, 0] - data[:, 1],
            data[:, 0] + data[:, 1],
            alpha=0.5,
            label="Uncertainty Region",
            color="orange",
        )
        ax.set_title("Prediction Sweep (10-190 kts)")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Intensification Rate")

        # Colorize the plot
        plot_facecolors(fig, np.array([ax]))

        # Show figure
        plt.tight_layout()
        plt.show()
        plt.close()

    def predict(self, X):
        """
        Predict the intensity climatology.
        """
        assert self.intensity_climatology is not None, "Model has not been trained."

        # Get the unique intensities in the input data
        mu = self.mu_interpolator(X)
        sigma = self.sigma_regressor.predict(X)

        return np.vstack([mu, sigma]).T

    def storm_prediction_plotter(self, storm, **kwargs):

        stormline_color = kwargs.get("stormline_color", np.array([242, 92, 5]) / 255)
        tick_color = kwargs.get("tick_color", np.array([242, 240, 228]) / 255)

        wind = storm.wind.astype(str)

        times = storm.timestamps[np.char.isnumeric(wind)]
        wind = wind[np.char.isnumeric(wind)].astype(float)

        wind = wind[np.isin(pd.to_datetime(times).hour, [0, 6, 12, 18])]
        times = times[np.isin(pd.to_datetime(times).hour, [0, 6, 12, 18])]

        rates = self.predict(wind.reshape(-1, 1))

        predicted_winds = wind + rates[:, 0]
        predicted_times = times + np.timedelta64(6, "h")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)

        ax.scatter(
            predicted_times, predicted_winds, label="Predictions", color=stormline_color
        )
        ax.fill_between(
            predicted_times.flatten(),
            predicted_winds - rates[:, 1],
            predicted_winds + rates[:, 1],
            alpha=0.5,
            label="Uncertainty Region",
            color="orange",
        )
        ax.set_title(
            f"6h Climatological Prediction for Storm {storm.uid} - {storm.name}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.scatter(times, wind, label="Ground Truth Intensity", color="black")

        import matplotlib.dates as mdates

        # Set minor ticks to every 6 hours
        hours = mdates.HourLocator(interval=6)
        ax.xaxis.set_minor_locator(hours)

        # Set x-axis tick label color
        ax.tick_params(axis="both", colors=tick_color)
        # ax.xaxis.set_minor_locator(AutoMinorLocator())

        # Enable grid + color
        ax.grid(True)
        ax.grid(color=tick_color)
        # ax.grid(which="minor", color=tick_color, alpha=0.5)

        plt.xticks(rotation=45)

        # Colorize the plot
        plot_facecolors(fig, np.array([ax]))

        # Show figure
        plt.tight_layout()
        plt.show()
        plt.close()


class TC_DeltaIntensity_MLR(BaseEstimator):
    """
    Intensity Linear model baseline.
    This model predicts the change in intensity of a storm based on the intensity of
    the previous time step + the max wind speed predicted by the AI model
    """

    def __repr__(self):
        return f"TCBench Climatology Baseline - {self.name}"

    def __init__(self, **kwargs):
        self.name = "Delta Intensity MLR"
        self.model_type = LinearRegression
        self.u_v_indices = kwargs.get("u_v_indices", [0, 1])
        self.msl_index = kwargs.get("msl_index", 2)
        self.lr = kwargs.get("lr", 0.01)

    def prepare_data(self, base_intensity, AI_X):
        wind_magnitude = np.sqrt(
            AI_X[:, self.u_v_indices[0]] ** 2 + AI_X[:, self.u_v_indices[1]] ** 2
        )
        wind_magnitude = wind_magnitude.max(axis=[*range(1, wind_magnitude.ndim)])
        wind_magnitude = wind_magnitude.reshape(-1, 1)
        mslp = AI_X[:, self.msl_index]
        mslp = mslp.max(axis=[*range(1, mslp.ndim)])
        mslp = mslp.reshape(-1, 1)

        assert (
            base_intensity.shape[0] == AI_X.shape[0]
        ), "Unequal number of samples. Check AI model input size vs intensity size."

        assert (
            base_intensity.shape[1] == 2
        ), "Intensity array must have shape (n_samples, 2) corresponding to wind (idx_0) and pressure (idx_1)."

        return np.hstack([wind_magnitude, mslp, base_intensity])

    def fit(self, base_intensity, AI_X, y=None):
        X = self.prepare_data(base_intensity, AI_X)
        X = X.rechunk({1: X.shape[0] // 1000}).compute()
        delta = (y - base_intensity).rechunk({0: y.shape[0] // 1000}).compute()

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.models = []
        for i in range(y.shape[1]):
            model = self.model_type()
            model.fit(X, delta[:, i])
            self.models.append(model)

    def predict(self, base_intensity, AI_X):
        X = self.prepare_data(base_intensity, AI_X)

        y_preds = []
        for i in range(len(self.models)):
            y_preds.append(self.models[i].predict(X))

        return np.vstack(y_preds).T + base_intensity

    pass


class TC_DeltaIntensity_nonDilCNN(nn.Module):
    def __str__(self):
        return "Non_Dilated_CNN"

    def __init__(
        self,
        num_scalars,  # number of scalar inputs, e.g., base intensity, track, etc.
        **kwargs,
    ):
        super(TC_DeltaIntensity_nonDilCNN, self).__init__()

        # Assume inputs have 241x241 dimensions in channels (u, v, mslp, t_850, z_500)
        conv_depths = kwargs.get("depths", [32, 16, 16, 64, 96])

        self.pool_size = kwargs.get("pool_size", 2)
        self.pool_stride = kwargs.get("pool_stride", 2)

        # Layer that sees all inputs
        # Output size = 120x120
        self.conv1 = nn.Conv2d(
            5, conv_depths[0] + conv_depths[1], kernel_size=(2, 2), padding=0
        )

        # layer to convolve the first output with the first context layer.
        # There are 16+32=48 input channels and 64 output channels
        # The output dimensions are 120x120
        self.conv4 = nn.Conv2d(
            conv_depths[0] + conv_depths[1],
            conv_depths[2] + conv_depths[3],
            kernel_size=(3, 3),
            padding=1,
        )

        # layer to convolve the second output with the second context layer.
        # There are 64 + 16 = 80 input channels and 96 output channels
        # The output dimensions are 60x60
        self.conv5 = nn.Conv2d(
            conv_depths[2] + conv_depths[3],
            conv_depths[4],
            kernel_size=(3, 3),
            padding=1,
        )

        # dense layer to output the prediction. We predict the mean and standard
        # deviation of the intensity change, both for wind and pressure
        self.fc1 = nn.Linear(conv_depths[4] * 60 * 60, 128)

        # We will encode the scalars with a dense layer
        self.fc2 = nn.Linear(num_scalars, num_scalars * 8)

        # And make the prediction from the two dense layers
        self.fc3 = nn.Linear(
            128 + num_scalars * 8, 2 if kwargs.get("deterministic", False) else 4
        )

    def forward(self, x, scalars):

        caf = F.leaky_relu
        daf = F.leaky_relu
        pooling = F.max_pool2d

        # Apply first convolutional layer
        x = pooling(
            caf(self.conv1(x)), kernel_size=self.pool_size, stride=self.pool_stride
        )

        # Apply the second convolutional layer
        x = caf(self.conv4(x))
        x = pooling(x, kernel_size=self.pool_size, stride=self.pool_stride)

        # Apply the final convolutional layer
        x = caf(self.conv5(x))

        # Flatten the output
        x = x.view(-1, 60 * 60 * 96)

        # Apply the dense layers
        x = daf(self.fc1(x))
        scalars = torch.squeeze(daf(self.fc2(scalars)))

        # Concatenate the base intensity with the output of the dense layer
        x = torch.cat([x, scalars], dim=1)

        x = self.fc3(x)

        return x


class TC_DeltaIntensity_CNN(nn.Module):
    def __str__(self):
        return "Dilated_CNN"

    def __init__(self, **kwargs):
        super(TC_DeltaIntensity_CNN, self).__init__()

        # Assume inputs have 241x241 dimensions in channels (u, v, mslp, t_850, z_500)

        conv_depths = kwargs.get("depths", [32, 16, 16, 64, 96])

        self.caf = kwargs.get(
            "cnn_activation_function", F.leaky_relu
        )  # Convolutional activation function
        self.daf = kwargs.get(
            "dense_activation_function", F.leaky_relu
        )  # Dense activation function

        # Layer that sees all inputs
        # Output size = 120x120
        self.conv1 = nn.Conv2d(5, conv_depths[0], kernel_size=(2, 2), padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # First context layer
        # Output size = 120x120
        self.conv2 = nn.Conv2d(
            5, conv_depths[1], kernel_size=(3, 3), padding=1, stride=2, dilation=2
        )

        # Second context layer
        # Output size = 60x60
        # self.conv3 = nn.Conv2d(
        #     5, conv_depths[2], kernel_size=(3, 3), padding=1, stride=4, dilation=3
        # )
        self.conv3 = nn.Conv2d(
            5, conv_depths[2], kernel_size=(3, 3), padding=4, stride=4, dilation=6
        )

        # layer to convolve the first output with the first context layer.
        # There are 16+32=48 input channels and 64 output channels
        # The output dimensions are 120x120
        self.conv4 = nn.Conv2d(48, conv_depths[3], kernel_size=(3, 3), padding=1)

        # layer to convolve the second output with the second context layer.
        # There are 64 + 16 = 80 input channels and 96 output channels
        # The output dimensions are 60x60
        self.conv5 = nn.Conv2d(80, conv_depths[4], kernel_size=(3, 3), padding=1)

        # dense layer to output the prediction. We predict the mean and standard
        # deviation of the intensity change, both for wind and pressure
        self.fc1 = nn.Linear(conv_depths[4] * 60 * 60, 128)

        # We will encode the baseline intensity with a dense layer
        self.fc2 = nn.Linear(2, 16)

        # And make the prediction from the two dense layers
        self.fc3 = nn.Linear(128 + 16, 2 if kwargs.get("deterministic", False) else 4)

    def forward(self, x, base_int):
        # Get the context from the inputs
        x_context1 = self.caf(self.conv2(x))
        x_context2 = self.caf(self.conv3(x))

        # Apply first convolutional layer
        x = self.pool(self.caf(self.conv1(x)))

        # Apply the first context layer
        x = torch.cat([x, x_context1], dim=1)
        x = self.caf(self.conv4(x))
        x = self.pool(x)

        # Apply the second context layer
        x = torch.cat([x, x_context2], dim=1)
        x = self.caf(self.conv5(x))

        # Flatten the output
        x = x.view(-1, 60 * 60 * 96)

        # Apply the dense layers
        x = self.daf(self.fc1(x))
        base_int = torch.squeeze(self.daf(self.fc2(base_int)))

        # Concatenate the base intensity with the output of the dense layer
        x = torch.cat([x, base_int], dim=1)

        x = self.fc3(x)

        return x


class Regularized_NonDil_CNN(TC_DeltaIntensity_nonDilCNN):
    def __str__(self):
        return "Regularized_Dilated_CNN"

    def __init__(self, **kwargs):
        super(Regularized_NonDil_CNN, self).__init__(**kwargs)
        self.dropout2d = kwargs.get("dropout2d", 0.05)
        self.dropout = kwargs.get("dropout", 0.05)

    def forward(self, x, scalars):
        caf = F.leaky_relu
        daf = F.leaky_relu
        pooling = F.max_pool2d

        # Apply first convolutional layer
        x = pooling(
            caf(self.conv1(x)), kernel_size=self.pool_size, stride=self.pool_stride
        )
        x = F.dropout2d(x, p=self.dropout2d)

        # Apply the second convolutional layer
        x = caf(self.conv4(x))
        x = pooling(x, kernel_size=self.pool_size, stride=self.pool_stride)
        x = F.dropout2d(x, p=self.dropout2d)

        # Apply the final convolutional layer
        x = caf(self.conv5(x))
        x = F.dropout2d(x, p=self.dropout2d)

        # Flatten the output
        x = x.view(-1, 60 * 60 * 96)

        # Apply the dense layers
        x = daf(self.fc1(x))
        x = F.dropout(x, p=self.dropout)
        scalars = torch.squeeze(daf(self.fc2(scalars)))
        scalars = F.dropout(scalars, p=self.dropout)

        # Concatenate the base intensity with the output of the dense layer
        x = torch.cat([x, scalars], dim=1)

        x = self.fc3(x)

        return x


class Regularized_Dilated_CNN(TC_DeltaIntensity_CNN):
    def __str__(self):
        return "Regularized_Dilated_CNN"

    def __init__(self, **kwargs):
        super(Regularized_Dilated_CNN, self).__init__(**kwargs)
        self.dropout2d = kwargs.get("dropout2d", 0.25)
        self.dropout = kwargs.get("dropout", 0.25)

    def forward(self, x, base_int):
        # Get the context from the inputs
        x_context1 = F.relu(self.conv2(x))
        x_context2 = F.relu(self.conv3(x))

        # Apply first convolutional layer
        x = self.pool(F.relu(self.conv1(x)))

        # Apply 2d dropout
        x = F.dropout2d(x, p=self.dropout2d)

        # Apply the first context layer
        x = torch.cat([x, x_context1], dim=1)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Apply 2d dropout
        x = F.dropout2d(x, p=self.dropout2d)

        # Apply the second context layer
        x = torch.cat([x, x_context2], dim=1)
        x = F.relu(self.conv5(x))

        # Apply 2d dropout
        x = F.dropout2d(x, p=self.dropout2d)

        # Flatten the output
        x = x.view(-1, 60 * 60 * 96)

        # Apply the dense layers and dropout each path
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout)
        base_int = torch.squeeze(F.relu(self.fc2(base_int)))
        base_int = F.dropout(base_int, p=self.dropout)

        # Concatenate the base intensity with the output of the dense layer
        x = torch.cat([x, base_int], dim=1)

        x = self.fc3(x)

        return x


class SimpleCNN(nn.Module):
    def __str__(self):
        return "Simple_CNN"

    def __init__(self, num_scalars, fc_width=512, cnn_widths=[32, 64, 128], **kwargs):
        super(SimpleCNN, self).__init__()
        self.pool_size = kwargs.get("pool_size", 2)
        self.pool_stride = kwargs.get("pool_stride", 2)
        self.kernel_size = kwargs.get("kernel_size", [3, 3, 3])
        self.strides = kwargs.get("strides", [1, 1, 1])
        self.paddings = kwargs.get("paddings", [1, 1, 1])
        input_cols = kwargs.get("input_cols", 5)
        target_size = kwargs.get("target_size", 2)
        batch_norm = kwargs.get("batch_norm", True)

        self.bn = batch_norm
        self.target_size = target_size
        # Output size = (input_size - kernel_size + 2*padding) / stride + 1
        # after pooling output size = output_size / (pool_size * pool_stride)
        self.size = 241
        self.conv1 = nn.Conv2d(
            input_cols,  # For us, AI outputs contain up to 5 variables (u, v, mslp, t_850, z_500)
            cnn_widths[0],
            kernel_size=self.kernel_size[0],
            stride=self.strides[0],
            padding=self.paddings[0],
        )
        self.bn1 = nn.BatchNorm2d(cnn_widths[0]) if batch_norm else None
        self.size = (
            self.size - self.kernel_size[0] + 2 * self.paddings[0]
        ) / self.strides[0] + 1
        self.pool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_stride)
        self.size = (self.size - self.pool_size) // self.pool_stride + 1
        self.conv2 = nn.Conv2d(
            cnn_widths[0],
            cnn_widths[1],
            kernel_size=self.kernel_size[1],
            stride=self.strides[1],
            padding=self.paddings[1],
        )
        self.bn2 = nn.BatchNorm2d(cnn_widths[1]) if batch_norm else None
        self.size = (
            self.size - self.kernel_size[1] + 2 * self.paddings[1]
        ) / self.strides[1] + 1
        # pooling after each layer changes size
        self.size = (self.size - self.pool_size) // self.pool_stride + 1

        self.conv3 = nn.Conv2d(
            cnn_widths[1],
            cnn_widths[2],
            kernel_size=self.kernel_size[2],
            stride=self.strides[2],
            padding=self.paddings[2],
        )
        self.bn3 = nn.BatchNorm2d(cnn_widths[2]) if batch_norm else None
        self.size = (
            self.size - self.kernel_size[2] + 2 * self.paddings[2]
        ) / self.strides[2] + 1
        # pooling after each layer changes size
        self.size = (self.size - self.pool_size) // self.pool_stride + 1

        # calculate the flat size
        self.flat_size = int(cnn_widths[2] * self.size * self.size)
        self.fc1 = nn.Linear(self.flat_size, fc_width)

        self.num_outputs = self.target_size * (
            1 if kwargs.get("deterministic", False) else 2
        ) + (self.target_size if kwargs.get("aux_loss", False) else 0)

        # We will encode the baseline intensity with a dense layer
        self.fc2 = nn.Linear(num_scalars, num_scalars * 8)
        self.fc3 = nn.Linear(num_scalars * 8, num_scalars * 8)
        self.fc4 = nn.Linear(fc_width + num_scalars * 8, self.num_outputs)

    def forward(self, x, scalars):
        x = (
            self.pool(self.bn1(F.relu(self.conv1(x))))
            if self.bn
            else self.pool(F.relu(self.conv1(x)))
        )
        x = (
            self.pool(self.bn2(F.relu(self.conv2(x))))
            if self.bn
            else self.pool(F.relu(self.conv2(x)))
        )
        x = (
            self.pool(self.bn3(F.relu(self.conv3(x))))
            if self.bn
            else self.pool(F.relu(self.conv3(x)))
        )
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        scalars = F.relu(self.fc2(scalars))
        scalars = torch.squeeze(F.relu(self.fc3(scalars)))
        # Concatenate the base intensity with the output of the dense layer
        x = torch.cat([x, scalars], dim=1)
        x = self.fc4(x)
        return x


class RegularizedCNN(SimpleCNN):
    def __str__(self):
        return "Regularized_CNN"

    def __init__(self, num_scalars, fc_width=512, cnn_widths=[32, 64, 128], **kwargs):
        super().__init__(num_scalars, fc_width, cnn_widths, **kwargs)
        self.dropout2d = kwargs.get("dropout2d", 0.5)
        self.dropout = kwargs.get("dropout", 0.5)

    def forward(self, x, scalars):
        x = (
            F.dropout2d(
                self.pool(self.bn1(F.hardswish(self.conv1(x)))), p=self.dropout2d
            )
            if self.bn
            else F.dropout2d(self.pool(F.hardswish(self.conv1(x))), p=self.dropout2d)
        )
        x = (
            F.dropout2d(
                self.pool(self.bn2(F.hardswish(self.conv2(x)))), p=self.dropout2d
            )
            if self.bn
            else F.dropout2d(self.pool(F.hardswish(self.conv2(x))), p=self.dropout2d)
        )
        x = (
            F.dropout2d(
                self.pool(self.bn3(F.hardswish(self.conv3(x)))), p=self.dropout2d
            )
            if self.bn
            else F.dropout2d(self.pool(F.hardswish(self.conv3(x))), p=self.dropout2d)
        )
        x = x.view(-1, self.flat_size)
        x = F.dropout(F.hardswish(self.fc1(x)), p=self.dropout)

        # make sure x has the right shape
        if x.dim() == 3:
            x = x.unsqueeze(0)

        scalars = F.dropout(F.hardswish(self.fc2(scalars)), p=self.dropout)
        scalars = torch.squeeze(
            F.dropout(F.hardswish(self.fc3(scalars)), p=self.dropout)
        )
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
        # Concatenate the base intensity with the output of the dense layer
        x = torch.cat([x, scalars], dim=1)
        x = self.fc4(x)
        return x


# %%
# UNet written with ChatGPT:
class UNet(nn.Module):
    def __str__(self):
        return "UNet"

    def __init__(self, num_scalars, fc_width=512, cnn_widths=[32, 64, 128], **kwargs):
        super(UNet, self).__init__()
        self.pool_size = kwargs.get("pool_size", 2)
        self.pool_stride = kwargs.get("pool_stride", 2)
        self.kernel_size = kwargs.get("kernel_size", [3, 3, 3])
        self.strides = kwargs.get("strides", [1, 1, 1])
        self.paddings = kwargs.get("paddings", [1, 1, 1])
        input_cols = kwargs.get("input_cols", 5)
        target_size = kwargs.get("target_size", 2)
        batch_norm = kwargs.get("batch_norm", True)
        self.dropout2d = kwargs.get("dropout2d", 0.5)
        self.dropout = kwargs.get("dropout", 0.5)

        self.bn = batch_norm
        self.target_size = target_size

        # Encoder
        self.enc1 = self._conv_block(
            input_cols,
            cnn_widths[0],
            self.kernel_size[0],
            self.strides[0],
            self.paddings[0],
        )
        self.enc2 = self._conv_block(
            cnn_widths[0],
            cnn_widths[1],
            self.kernel_size[1],
            self.strides[1],
            self.paddings[1],
        )
        self.enc3 = self._conv_block(
            cnn_widths[1],
            cnn_widths[2],
            self.kernel_size[2],
            self.strides[2],
            self.paddings[2],
        )

        self.pool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_stride)

        # Bottleneck
        self.bottleneck = self._conv_block(
            cnn_widths[2],
            cnn_widths[2] * 2,
            self.kernel_size[-1],
            self.strides[-1],
            self.paddings[-1],
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(
            cnn_widths[2] * 2, cnn_widths[2], kernel_size=2, stride=2
        )
        self.dec3 = self._conv_block(
            cnn_widths[2],
            cnn_widths[2],
            self.kernel_size[2],
            self.strides[2],
            self.paddings[2],
        )

        self.upconv2 = nn.ConvTranspose2d(
            cnn_widths[2], cnn_widths[1], kernel_size=2, stride=2
        )
        self.dec2 = self._conv_block(
            cnn_widths[1],
            cnn_widths[1],
            self.kernel_size[1],
            self.strides[1],
            self.paddings[1],
        )

        self.upconv1 = nn.ConvTranspose2d(
            cnn_widths[1], cnn_widths[0], kernel_size=2, stride=2
        )
        self.dec1 = self._conv_block(
            cnn_widths[0],
            cnn_widths[0],
            self.kernel_size[0],
            self.strides[0],
            self.paddings[0],
        )

        # Fully connected layers for scalars
        self.fc1 = nn.Linear(cnn_widths[0] * 241 * 241, fc_width)
        self.fc2 = nn.Linear(num_scalars, num_scalars * 8)
        self.fc3 = nn.Linear(num_scalars * 8, num_scalars * 8)

        self.num_outputs = self.target_size * (
            1 if kwargs.get("deterministic", False) else 2
        ) + (self.target_size if kwargs.get("aux_loss", False) else 0)

        self.fc4 = nn.Linear(fc_width + num_scalars * 8, self.num_outputs)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels) if self.bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x, scalars):

        # Handle single item batches
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        # Encoder
        enc1 = F.dropout2d(self.enc1(x), p=self.dropout2d)
        x = self.pool(enc1)
        enc2 = F.dropout2d(self.enc2(x), p=self.dropout2d)
        x = self.pool(enc2)
        enc3 = F.dropout2d(self.enc3(x), p=self.dropout2d)
        x = self.pool(enc3)

        # Bottleneck
        x = F.dropout2d(self.bottleneck(x), p=self.dropout2d)

        # # Decoder
        # x = self.upconv3(x)
        # x = torch.cat([x, enc3], dim=1)
        # x = self.dec3(x)

        # x = self.upconv2(x)
        # x = torch.cat([x, enc2], dim=1)
        # x = self.dec2(x)

        # x = self.upconv1(x)
        # x = torch.cat([x, enc1], dim=1)
        # x = self.dec1(x)

        # Decoder
        x = self.upconv3(x)
        if x.size() != enc3.size():
            x = F.interpolate(
                x, size=enc3.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc3  # Residual connection
        x = self.dec3(x)

        x = self.upconv2(x)
        if x.size() != enc2.size():
            x = F.interpolate(
                x, size=enc2.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc2  # Residual connection
        x = self.dec2(x)

        x = self.upconv1(x)
        if x.size() != enc1.size():
            x = F.interpolate(
                x, size=enc1.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc1  # Residual connection
        x = self.dec1(x)

        # Flatten for FC layer
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        # Scalar processing
        scalars = F.relu(self.fc2(scalars))
        scalars = torch.squeeze(F.dropout(F.relu(self.fc3(scalars)), p=self.dropout))

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        # Concatenate and final FC layer
        x = torch.cat([x, scalars], dim=1)
        x = self.fc4(x)
        return x


class UNet_v2(UNet):
    def __str__(self):
        return "UNet_v2"

    def __init__(self, num_scalars, fc_width=512, cnn_widths=[32, 64, 128], **kwargs):
        super(UNet_v2, self).__init__(num_scalars, fc_width, cnn_widths, **kwargs)

        # Unet outchannels
        self.unet_channels = kwargs.get("unet_channels", 4)

        # add a convolutional layer after the unet that outputs a 4 channels
        self.conv_out = nn.Conv2d(cnn_widths[0], self.unet_channels, kernel_size=1)
        self.bn_out = (
            nn.BatchNorm2d(self.unet_channels)
            if kwargs.get("batch_norm", True)
            else None
        )

        self.fc1 = nn.Linear(241 * 241 * 4, fc_width)

        self.dropout2d = kwargs.get("dropout2d", 0.5)
        self.dropout = kwargs.get("dropout", 0.5)

    def forward(self, x, scalars):

        # Handle single item batches
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        # Encoder
        enc1 = F.dropout2d(self.enc1(x), p=self.dropout2d)
        x = self.pool(enc1)
        enc2 = F.dropout2d(self.enc2(x), p=self.dropout2d)
        x = self.pool(enc2)
        enc3 = F.dropout2d(self.enc3(x), p=self.dropout2d)
        x = self.pool(enc3)

        # Bottleneck
        x = F.dropout2d(self.bottleneck(x), p=self.dropout2d)

        # Decoder
        x = self.upconv3(x)
        if x.size() != enc3.size():
            x = F.interpolate(
                x, size=enc3.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc3  # Residual connection
        x = self.dec3(x)
        x = F.dropout2d(x, p=self.dropout2d)

        x = self.upconv2(x)
        if x.size() != enc2.size():
            x = F.interpolate(
                x, size=enc2.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc2  # Residual connection
        x = self.dec2(x)
        x = F.dropout2d(x, p=self.dropout2d)

        x = self.upconv1(x)
        if x.size() != enc1.size():
            x = F.interpolate(
                x, size=enc1.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc1  # Residual connection
        x = self.dec1(x)
        x = F.dropout2d(x, p=self.dropout2d)

        # Convolve to output a single channel
        x = self.conv_out(x)
        if self.bn_out:
            x = self.bn_out(x)
        x = F.relu(x)
        # make sure the output image is the right size
        x = F.interpolate(x, size=(241, 241), mode="bilinear", align_corners=False)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Scalar processing
        scalars = F.relu(self.fc2(scalars))
        scalars = torch.squeeze(F.dropout(F.relu(self.fc3(scalars)), p=self.dropout))

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        # Concatenate and final FC layer
        x = torch.cat([x, scalars], dim=1)
        x = self.fc4(x)
        return x


# %%
class Prob_UNet(UNet_v2):
    def __str__(self):
        return "Prob_UNet"

    def __init__(self, num_scalars, fc_width=512, cnn_widths=[32, 64, 128], **kwargs):
        super(UNet_v2, self).__init__(num_scalars, fc_width, cnn_widths, **kwargs)

        # Unet outchannels
        self.unet_channels = kwargs.get("unet_channels", 4)

        # add a convolutional layer after the unet that outputs a 4 channels
        self.conv_out = nn.Conv2d(cnn_widths[0], self.unet_channels, kernel_size=1)
        self.bn_out = (
            nn.BatchNorm2d(self.unet_channels)
            if kwargs.get("batch_norm", True)
            else None
        )

        self.fc1 = nn.Linear(241 * 241 * 4, fc_width)

        self.dropout2d = kwargs.get("dropout2d", 0.5)
        self.dropout = kwargs.get("dropout", 0.5)

        self.fc4 = nn.Linear(fc_width + num_scalars * 8, 2)  # branch_for_mean
        self.fc5 = nn.Linear(fc_width + num_scalars * 8, 2)  # branch_for_std

    def forward(self, x, scalars):
        # Handle single item batches
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        # Encoder
        enc1 = F.dropout2d(self.enc1(x), p=self.dropout2d)
        x = self.pool(enc1)
        enc2 = F.dropout2d(self.enc2(x), p=self.dropout2d)
        x = self.pool(enc2)
        enc3 = F.dropout2d(self.enc3(x), p=self.dropout2d)
        x = self.pool(enc3)

        # Bottleneck
        x = F.dropout2d(self.bottleneck(x), p=self.dropout2d)

        # Decoder
        x = self.upconv3(x)
        if x.size() != enc3.size():
            x = F.interpolate(
                x, size=enc3.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc3  # Residual connection
        x = self.dec3(x)
        x = F.dropout2d(x, p=self.dropout2d)

        x = self.upconv2(x)
        if x.size() != enc2.size():
            x = F.interpolate(
                x, size=enc2.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc2  # Residual connection
        x = self.dec2(x)
        x = F.dropout2d(x, p=self.dropout2d)

        x = self.upconv1(x)
        if x.size() != enc1.size():
            x = F.interpolate(
                x, size=enc1.size()[2:], mode="bilinear", align_corners=False
            )
        x = x + enc1  # Residual connection
        x = self.dec1(x)
        x = F.dropout2d(x, p=self.dropout2d)

        # Convolve to output a single channel
        x = self.conv_out(x)
        if self.bn_out:
            x = self.bn_out(x)
        x = F.relu(x)
        # make sure the output image is the right size
        x = F.interpolate(x, size=(241, 241), mode="bilinear", align_corners=False)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Scalar processing
        scalars = F.relu(self.fc2(scalars))
        scalars = torch.squeeze(F.dropout(F.relu(self.fc3(scalars)), p=self.dropout))

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        # Concatenate and final FC layer
        x = torch.cat([x, scalars], dim=1)
        x_mean = self.fc4(x)
        x_std = F.relu(self.fc5(x))

        # Concatenate mean and std outputs
        x = torch.cat([x_mean, x_std], dim=1)

        return x


# %%


class TorchMLR(nn.Module):
    def __str__(self):
        return "TorchMLR"

    def __init__(self, **kwargs):
        super(TorchMLR, self).__init__()

        input_cols = kwargs.get("input_cols", 5)
        num_scalars = kwargs.get("num_scalars", 2)

        self.linear = nn.Linear(
            input_cols * 2 + num_scalars, 2 if kwargs.get("deterministic", False) else 4
        )  # Linear layer

    def forward(self, x, scalars):
        # Find the maximum and min values across the x channels
        maxs = x.max(dim=3).values.max(dim=2).values
        mins = x.min(dim=3).values.min(dim=2).values

        scalars = torch.squeeze(scalars)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
            print(scalars.shape)

        # Concatenate the max and min values with the scalar inputs
        x = torch.cat([maxs, mins, scalars], dim=1)
        x = self.linear(x)
        return x


class TorchMLRv2(nn.Module):
    def __str__(self):
        return "TorchMLR"

    def __init__(self, **kwargs):
        super(TorchMLRv2, self).__init__()

        num_scalars = kwargs.get("num_scalars", 2)

        self.linear = nn.Linear(
            num_scalars, 2 if kwargs.get("deterministic", False) else 4
        )  # Linear layer

    def forward(self, x):
        # Find the maximum and min values across the x channels
        x = torch.squeeze(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Concatenate the max and min values with the scalar inputs
        x = self.linear(x)
        return x


class ANN(nn.Module):
    def __str__(self):
        return "SimpleANN"

    def __repr__(self):
        return "SimpleANN"

    def __init__(self, **kwargs):
        super(ANN, self).__init__()

        num_scalars = kwargs.get("num_scalars", 2)
        unit_multiplier = kwargs.get("unit_multiplier", 4)

        activation_function = kwargs.get("activation_function", nn.Hardswish)

        self.layers = nn.Sequential(
            nn.Linear(num_scalars, num_scalars * unit_multiplier),
            activation_function(),
            nn.Linear(num_scalars * unit_multiplier, num_scalars * unit_multiplier),
            activation_function(),
            nn.Linear(num_scalars * unit_multiplier, num_scalars * unit_multiplier),
            activation_function(),
            nn.Linear(num_scalars * unit_multiplier, num_scalars * unit_multiplier),
            activation_function(),
            nn.Linear(num_scalars * unit_multiplier, num_scalars * unit_multiplier),
            activation_function(),
            nn.Linear(num_scalars * unit_multiplier, num_scalars * unit_multiplier),
            activation_function(),
            nn.Linear(
                num_scalars * unit_multiplier,
                2 if kwargs.get("deterministic", False) else 4,
            ),
        )

    def forward(self, x):
        # Find the maximum and min values across the x channels
        x = torch.squeeze(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Forward pass
        x = self.layers(x)
        return x


class AveClimatology:
    def __str__(self):
        return "TCBench Average Climatology Baseline"

    def __init__(self, **kwargs):
        self.name = "Ave Climatology"
        self.mu = {}
        self.sigma = {}

    def fit(self, target, leadtimes):
        unique_leadtimes = np.unique(leadtimes)
        for leadtime in unique_leadtimes:
            mask = leadtimes == leadtime
            self.mu[leadtime] = target[mask].mean(axis=0)
            self.sigma[leadtime] = target[mask].std(axis=0)

    def predict(self, leadtimes):
        mu = np.array([self.mu[leadtime] for leadtime in leadtimes])
        sigma = np.array([self.sigma[leadtime] for leadtime in leadtimes])
        return np.vstack([mu, sigma]).T


# %%

# %%
