import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

temperature_file = "data/temperature.csv"
irradiance_file = "data/irradiance.csv"


def get_csv_data(temperature_file, irradiance_file):
    irradiance = pd.read_csv(irradiance_file)
    irradiance["index_date"] = pd.to_datetime(irradiance["index_date"])
    irradiance.set_index(irradiance["index_date"], inplace=True)

    temperature = pd.read_csv(temperature_file)
    temperature["valid_time"] = pd.to_datetime(temperature["valid_time"])
    temperature.set_index(temperature["valid_time"], inplace=True)

    return irradiance, temperature


irradiance, temperature = get_csv_data(temperature_file, irradiance_file)


def assign_season(month):
    if 3 <= month <= 5:
        return "spring"
    elif 6 <= month <= 8:
        return "summer"
    elif 9 <= month <= 11:
        return "autumn"
    else:
        return "winter"


irradiance["season"] = irradiance.index.month.map(assign_season)
temperature["season"] = temperature.index.month.map(assign_season)

irradiance["time"] = irradiance.index.strftime("%H:%M")
temperature["time"] = temperature.index.strftime("%H:%M")

seasons = ["winter", "spring", "summer", "autumn"]
irradiance["season"] = pd.Categorical(
    irradiance["season"], categories=seasons, ordered=True
)
temperature["season"] = pd.Categorical(
    temperature["season"], categories=seasons, ordered=True
)


def compute_seasonal_time_averages(df):
    return df.groupby(["season", "time"]).mean()


avg_irradiance = compute_seasonal_time_averages(irradiance)
avg_temperature = compute_seasonal_time_averages(temperature)

# Order df by season


# Define the plotting function
def plot_seasonal_profiles(avg_irradiance, avg_temperature, seasons):
    fig, axs = plt.subplots(len(seasons), 1, figsize=(12, 8), sharex=True)

    for i, season in enumerate(seasons):
        # Plot irradiance
        axs[i].plot(
            pd.to_datetime(avg_irradiance.loc[season].index),
            avg_irradiance.loc[season]["GHI"],
            label="GHI",
            color="orange",
            linewidth=2,
        )

        # Create a twin axis for temperature
        ax2 = axs[i].twinx()
        ax2.plot(
            pd.to_datetime(avg_temperature.loc[season].index),
            avg_temperature.loc[season]["t2m"],
            label="Temperature",
            color="blue",
            linewidth=2,
            linestyle="--",
        )

        # Formatting the x-axis to show hours and minutes
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Set titles and labels
        axs[i].set_title(f"{season.capitalize()} Average Profile")
        axs[i].set_ylabel("Irradiance (W/m²)", color="orange")
        ax2.set_ylabel("Temperature (°C)", color="blue")

        # Set legend
        axs[i].legend(loc="upper left")
        ax2.legend(loc="upper right")

    # Improve spacing and set a common x label
    plt.tight_layout()
    axs[-1].set_xlabel("Time of Day")
    plt.show()


avg_irradiance.to_csv("irradiance_seasons.csv")
avg_temperature.to_csv("temperature_seasons.csv")
# Call the plotting function
plot_seasonal_profiles(avg_irradiance, avg_temperature, seasons)
