import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from scipy.stats import ttest_ind
import plotly.graph_objects as go
class Cleaning_factor():
    def __init__(self,df):
        self.df = df
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