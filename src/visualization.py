import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

class Visualization:
    def __init__(self, df):
        self.df = df

    def plot_line_graph_with_timestamp(self, x, y_cols):
        self.df[x] = pd.to_datetime(self.df[x], errors='coerce')
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        for i, y_col in enumerate(y_cols):
            self.df.plot(x=x, y=y_col, kind='line', ax=axes[i])
            axes[i].set_title(y_col)

        plt.suptitle("Line Graph with Timestamp")
        plt.subplots_adjust(top=0.9, wspace=0.4, hspace=0.6)
        plt.show()

