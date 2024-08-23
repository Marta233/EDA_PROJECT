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

    def evaluate_cleaning_impact(self, timestamp_col, cleaning_col, mod_cols):
        # Convert timestamp column to datetime if not already
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])

        # Group by Cleaning status and compute mean of ModA and ModB
        mean_values = self.df.groupby(cleaning_col)[mod_cols].mean().reset_index()
        print(f"Mean values by {cleaning_col} status:\n", mean_values)

        # Plotting the time series of ModA and ModB with cleaning events
        fig, ax = plt.subplots(figsize=(12, 6))
        for mod_col in mod_cols:
            self.df.plot(x=timestamp_col, y=mod_col, color=(self.df[cleaning_col] == 1).astype(int), ax=ax)
        ax.set_title('Impact of Cleaning on Sensor Readings Over Time')
        plt.show()

        # Statistical comparison
        ttest_results = {}
        for mod_col in mod_cols:
            cleaned = self.df[self.df[cleaning_col] == 1][mod_col]
            not_cleaned = self.df[self.df[cleaning_col] == 0][mod_col]

            t_stat, p_value = ttest_ind(cleaned, not_cleaned, equal_var=False)
            ttest_results[mod_col] = {'t_stat': t_stat, 'p_value': p_value}
            print(f'Test results for {mod_col} - t-statistic: {t_stat}, p-value: {p_value}')

        return mean_values, ttest_results

    def plot_correlation_heatmap(self, features):
        corr_matrix = self.df[features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(corr_matrix, cmap='YlOrRd')

        # Add labels and title
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(features)
        ax.set_yticklabels(features)
        ax.set_title('Correlation Heatmap')

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        plt.colorbar(ax.imshow(corr_matrix, cmap='YlOrRd'))
        plt.show()

    def plot_scatter_matrix(self, features):
        fig, axes = plt.subplots(len(features), len(features), figsize=(12, 12))

        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                axes[i, j].scatter(self.df[feature1], self.df[feature2])
                axes[i, j].set_xlabel(feature1)
                axes[i, j].set_ylabel(feature2)

        plt.suptitle('Scatter Matrix')
        plt.tight_layout()
        plt.show()

    def histogram_for_list_fields(self, list_fields):
        for field in list_fields:
            fig, ax = plt.subplots(figsize=(8, 6))
            self.df[field].hist(bins=20, ax=ax)
            ax.set_title(f'Histogram for {field}')
            plt.show()

    def analyze_z_score(self, field):
        mean = self.df[field].mean()
        std_dev = self.df[field].std()
        z_scores = (self.df[field] - mean) / std_dev
        return z_scores

    def bubble_chart(self, x, y, size_col, color_col, title='Bubble Chart'):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(self.df[x], self.df[y], s=self.df[size_col] * 100, c=self.df[color_col], alpha=0.7)

        # Add labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)

        # Add a colorbar for the color dimension
        cbar = plt.colorbar(ax.scatter([], [], s=100, c=[0], alpha=0.7))
        cbar.set_label(color_col)

        plt.show()

    def temperature_analysis(self):
        # Check if required columns exist
        required_cols = ['RH', 'Tamb', 'GHI', 'DNI', 'DHI']
        if not all(col in self.df.columns for col in required_cols):
            print("Required columns are missing.")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        y_cols = ['Tamb', 'GHI', 'DNI', 'DHI']
        for i, y_col in enumerate(y_cols):
            # Scatter plot
            axes[i].scatter(self.df['RH'], self.df[y_col])
            axes[i].set_xlabel('RH')
            axes[i].set_ylabel(y_col)
            axes[i].set_title(f'{y_col} vs RH')

            # Linear regression
            X = self.df[['RH']].dropna()  # Drop NaN values
            y = self.df[y_col].dropna()
            model = LinearRegression().fit(X, y)
            pred = model.predict(X)

            # Add the linear regression line to the plot
            axes[i].plot(X['RH'], pred, color='r', label='Regression Line')
            axes[i].legend()

        plt.suptitle('Influence of Relative Humidity on Temperature and Solar Radiation')
        plt.tight_layout()
        plt.show()