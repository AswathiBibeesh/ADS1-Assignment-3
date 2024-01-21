# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:26:54 2024

@author: aswat
"""
# Import necessary modules
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import scipy.optimize as opt

def read_file(file_name):
    """
    Read a CSV file and transpose the data.

    Parameters:
    - file_name (str): Path to the CSV file.

    Returns:
    - df (pd.DataFrame): Original DataFrame.
    - df_transpose (pd.DataFrame): Transposed DataFrame.
    """
    df = pd.read_csv(file_name, skiprows=4)
    df_transpose = pd.DataFrame.transpose(df)
    return df, df_transpose

def norm(array):
    """
    Normalize an array.

    Parameters:
    - array (np.ndarray): Input array.

    Returns:
    - scaled (np.ndarray): Scaled array.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array - min_val) / (max_val - min_val)
    return scaled

def norm_df(df, first=0, last=None):
    """
    Normalize numerical columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - first (int): Index of the first column to normalize.
    - last (int): Index of the last column to normalize.

    Returns:
    - df (pd.DataFrame): Normalized DataFrame.
    """
    for col in df.columns[first:last]:
        df[col] = norm(df[col])
    return df

def exp_growth(t, scale, growth):
    """
    Exponential growth function.

    Parameters:
    - t (np.ndarray): Time values.
    - scale (float): Scaling factor.
    - growth (float): Growth rate.

    Returns:
    - f (np.ndarray): Exponential growth values.
    """
    f = scale * np.exp(growth * (t - 1970.0))
    return f

def fit_exponential_growth(df_years, df_values):
    """
    Fit exponential growth to the data.

    Parameters:
    - df_years (pd.Series): Years column.
    - df_values (pd.Series): Values to fit.

    Returns:
    - popt (list): Fitted parameters.
    """
    popt, _ = opt.curve_fit(exp_growth, df_years, df_values)
    return popt

def kmeans_clustering(df_fit, n_clusters=4):
    """
    Perform k-means clustering on the given DataFrame.

    Parameters:
    - df_fit (pd.DataFrame): DataFrame with columns to fit.
    - n_clusters (int): Number of clusters.

    Returns:
    - labels (np.ndarray): Cluster labels.
    - centers (np.ndarray): Cluster centers.
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(df_fit)

    # Extract labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    return labels, centers

def generate_predictions(df_gdp_countries, norm_func, kmeans_func, n_clusters=4):
    """
    Generate predictions for the next 10 years using the cluster model.

    Parameters:
    - df_gdp_countries (pd.DataFrame): GDP per capita DataFrame.
    - norm_func (function): Function to normalize data.
    - kmeans_func (function): Function to perform k-means clustering.
    - n_clusters (int): Number of clusters.

    Returns:
    - prediction_df (pd.DataFrame): DataFrame with predictions.
    """
    future_years = np.arange(df_gdp_countries["Years"].max() + 1, df_gdp_countries["Years"].max() + 11)
    future_data = np.column_stack((future_years, np.zeros_like(future_years)))

    # Predict the cluster assignments for future data
    future_labels, _ = kmeans_func(norm_func(future_data), n_clusters=n_clusters)

    # Display the predictions
    prediction_df = pd.DataFrame({"Years": future_years, "Predicted_Cluster": future_labels})
    return prediction_df

# Reads the files into dataframes
df_pop_total, df_pop = read_file("C:\\Users\\aswat\Downloads\\population_growth\\API_SP.POP.GROW_DS2_en_csv_v2_6298705.csv")
df_gdp_total, df_gdp_countries = read_file("C:\\Users\\aswat\\Downloads\\GDP_per_capita\\API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_6298776.csv")

# Header setting for population dataframe
header = df_pop.iloc[0].values.tolist()
df_pop.columns = header

# Cleaning the population dataframe
df_pop = df_pop.iloc[0:]
df_pop = df_pop.iloc[11:55]
df_pop.index = df_pop.index.astype(int)
df_pop = df_pop[df_pop.index > 1961]
df_pop["Years"] = df_pop.index
first_column = df_pop.pop("Years")
df_pop.insert(0, "Years", first_column)
df_pop = df_pop.reset_index(drop=True)

# Plot the graph before fitting
plt.plot(df_pop["Years"], df_pop["World"])
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Population growth graph before fitting")
plt.legend()
plt.show()

# Convert values from float to numeric
df_pop["Years"] = pd.to_numeric(df_pop["Years"])
df_pop["World"] = pd.to_numeric(df_pop["World"])

# Fit exponential growth
popt_first_fit = fit_exponential_growth(df_pop["Years"], df_pop["World"])

# Calculate and plot the result for first fit
print("Fit parameter (First Fit):", popt_first_fit)
df_pop["pop_exp_first_fit"] = exp_growth(df_pop["Years"], *popt_first_fit)
plt.figure()
plt.plot(df_pop["Years"], df_pop["World"], label="Years")
plt.plot(df_pop["Years"], df_pop["pop_exp_first_fit"], label="fit")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("First Fit")
plt.show()

# Improved the start value and plot a better fitting line
popt_improved_start = [4e8, 0.02]
df_pop["pop_exp_improved_start"] = exp_growth(df_pop["Years"], *popt_improved_start)
plt.figure()
plt.plot(df_pop["Years"], df_pop["World"], label="Years")
plt.plot(df_pop["Years"], df_pop["pop_exp_improved_start"], label="fit")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Improved Start Value")
plt.show()

# Final fit exponential growth
popt_final_fit = fit_exponential_growth(df_pop["Years"], df_pop["World"])

# Plot the best fitting line
df_pop["pop_exp_final_fit"] = exp_growth(df_pop["Years"], *popt_final_fit)
plt.figure()
plt.plot(df_pop["Years"], df_pop["World"], label="Years")
plt.plot(df_pop["Years"], df_pop["pop_exp_final_fit"], label="fit")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Final Fit: Population Growth with Exponential Fit")
plt.show()

# Clustering for GDP per capita
# Header setting for GDP dataframe
header = df_gdp_countries.iloc[0].values.tolist()
df_gdp_countries.columns = header

# Cleaning the GDP dataframe
df_gdp_countries = df_gdp_countries.iloc[0:]
df_gdp_countries = df_gdp_countries.iloc[11:55]
df_gdp_countries.index = df_gdp_countries.index.astype(int)
df_gdp_countries = df_gdp_countries[df_gdp_countries.index > 1961]
df_gdp_countries["Years"] = df_gdp_countries.index
first_column = df_gdp_countries.pop("Years")
df_gdp_countries.insert(0, "Years", first_column)
df_gdp_countries = df_gdp_countries.reset_index(drop=True)

# Plot GDP per capita before clustering
plt.plot(df_gdp_countries["Years"], df_gdp_countries["World"])
plt.xlabel("Years")
plt.ylabel("GDP Per Capita")
plt.title("GDP per capita graph Before clustering")
plt.legend()
plt.show()

# Extract columns for fitting and normalize
df_fit = df_gdp_countries[["Years", "World"]].copy()
df_fit = norm_df(df_fit)

# Display normalized dataframe summary
print(df_fit.describe())

# Perform k-means clustering
labels, centers = kmeans_clustering(df_fit, n_clusters=4)

# Plot for 4 clusters
plt.figure(figsize=(5.0, 5.0))
plt.scatter(df_fit["Years"], df_fit["World"], c=labels, cmap="Accent")

# Show cluster centres
for ic in range(4):
    x = df_gdp_countries.index
    y = df_gdp_countries["World"]
    xc, yc = centers[ic, :]
    plt.plot(xc, yc, "dk", markersize=5)

plt.xlabel("Years")
plt.ylabel("GDP Per Capita")
plt.title("GDP per capita after clustering with 4 clusters")
plt.legend()
plt.show()

# Generate predictions for the next 10 years using the cluster model
predictions = generate_predictions(df_gdp_countries, norm, kmeans_clustering, n_clusters=4)

# Display the predictions
print("Predictions for the next 10 years:")
print(predictions)

