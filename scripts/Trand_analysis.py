import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import ttest_ind  # For statistical comparison
import matplotlib.pyplot as plt
class Trand_analysis:
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
