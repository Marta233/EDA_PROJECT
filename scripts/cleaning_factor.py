import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
from scipy.stats import ttest_ind
import plotly.graph_objects as go

 

class Cleaning_factor:
    def __init__(self, df):
        self.df = df

    def plot_line_graph_with_timestamp(self, x, y_cols):
        # Convert x column to datetime
        self.df[x] = pd.to_datetime(self.df[x], errors='coerce')
        
        # Set up subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        # Plot each y_col in a separate subplot
        for i, y_col in enumerate(y_cols):
            self.df.plot(x=x, y=y_col, kind='line', ax=axes[i])
            axes[i].set_title(y_col)

        # Set overall title and adjust layout
        plt.suptitle("Line Graph with Timestamp")
        plt.subplots_adjust(top=0.9, wspace=0.4, hspace=0.6)
        plt.show()
    def evaluate_cleaning_impact(self, timestamp_col, cleaning_col, mod_cols):
        # Convert timestamp column to datetime if not already
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])

        # Group by Cleaning status and compute mean of ModA and ModB
        mean_values = self.df.groupby(cleaning_col)[mod_cols].mean().reset_index()
        print(f"Mean values by {cleaning_col} status:\n", mean_values)

        # Plotting the time series of ModA and ModB with cleaning events using Plotly
        fig = px.line(self.df, x=timestamp_col, y=mod_cols, color=cleaning_col,
                      title='Impact of Cleaning on Sensor Readings Over Time')
        fig.show()

        # Statistical comparison
        ttest_results = {}
        for mod_col in mod_cols:
            cleaned = self.df[self.df[cleaning_col] == 1][mod_col]
            not_cleaned = self.df[self.df[cleaning_col] == 0][mod_col]

            t_stat, p_value = ttest_ind(cleaned, not_cleaned, equal_var=False)
            ttest_results[mod_col] = {'t_stat': t_stat, 'p_value': p_value}
            print(f'Test results for {mod_col} - t-statistic: {t_stat}, p-value: {p_value}')

        return mean_values, ttest_results

   