import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import ttest_ind  # For statistical comparison
import matplotlib.pyplot as plt
import seaborn as sns

class Visual:
    def __init__(self, df):
        """
        Initialize the Visual class with a DataFrame.
        
        :param df: Pandas DataFrame containing the data.
        """
        self.df = df

    def plot_line_graph(self, x, y):
        """
        Plot a line graph for the given x and y columns.
        
        :param x: Column name for the x-axis.
        :param y: Column name for the y-axis.
        """
        # Convert x to datetime if necessary
        self.df[x] = pd.to_datetime(self.df[x], format='mixed', dayfirst=True, errors='coerce')
        fig = px.line(self.df, x=x, y=y)
        fig.show()

    def evaluate_cleaning_impact(self, timestamp_col, cleaning_col, mod_cols):
        """
        Evaluate the impact of cleaning on sensor readings and perform statistical comparison.
        
        :param timestamp_col: Column name for the timestamp.
        :param cleaning_col: Column name for cleaning status (0 or 1).
        :param mod_cols: List of columns for which to evaluate the impact.
        :return: Mean values by cleaning status and statistical test results.
        """
        # Convert timestamp column to datetime if not already
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])

        # Group by cleaning status and compute mean of ModA and ModB
        mean_values = self.df.groupby(cleaning_col)[mod_cols].mean().reset_index()
        print(f"Mean values by {cleaning_col} status:\n", mean_values)

        # Plot time series of ModA and ModB with cleaning events
        fig = px.line(self.df, x=timestamp_col, y=mod_cols, color=cleaning_col,
                     title='Impact of Cleaning on Sensor Readings Over Time')
        fig.show()

        # Perform statistical comparison (t-test)
        ttest_results = {}
        for mod_col in mod_cols:
            cleaned = self.df[self.df[cleaning_col] == 1][mod_col]
            not_cleaned = self.df[self.df[cleaning_col] == 0][mod_col]

            t_stat, p_value = ttest_ind(cleaned, not_cleaned, equal_var=False)
            ttest_results[mod_col] = {'t_stat': t_stat, 'p_value': p_value}
            print(f'Test results for {mod_col} - t-statistic: {t_stat}, p-value: {p_value}')

        return mean_values, ttest_results

    def plot_correlation_heatmap(self, features):
        """
        Plot a heatmap of the correlation matrix for the given features.
        
        :param features: List of column names to include in the correlation matrix.
        """
        corr_matrix = self.df[features].corr()
    
        fig = go.Figure(data=go.Heatmap(
            x=features,
            y=features,
            z=corr_matrix,
            colorscale='YlOrRd',
            text=np.round(corr_matrix, 2).values,
            texttemplate='%{text}'
        ))
    
        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=600
        )
    
        fig.show()

    def plot_scatter_matrix(self, features):
        """
        Plot a scatter matrix for the given features.
        
        :param features: List of column names to include in the scatter matrix.
        """
        fig = px.scatter_matrix(
            self.df[features],
            dimensions=features,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
    
        fig.update_layout(
            title='Scatter Matrix',
            width=800,
            height=600
        )
    
        fig.show()

    def histogram_for_list_fields(self, list_fields):
        """
        Plot histograms for the given list of fields.
        
        :param list_fields: List of column names for which to plot histograms.
        """
        for field in list_fields:
            fig = px.histogram(self.df, x=field, nbins=20, title=f'Histogram for {field}')
            fig.show()

    def plot_histograms(self, variables):
        """
        Plot histograms for the given list of variables using Matplotlib and Seaborn.
        
        :param variables: List of column names for which to plot histograms.
        """
        plt.figure(figsize=(14, 10))
        for i, var in enumerate(variables, 1):
            plt.subplot(3, 3, i)
            sns.histplot(self.df[var], kde=True, color='skyblue', bins=30)
            plt.title(f'{var} Histogram')
            plt.xlabel(var)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def outlier_count_z_score_all_fields(self, z_score_threshold=3):
        """
        Count outliers in all numeric fields using Z-score method.
        
        :param z_score_threshold: Z-score threshold for identifying outliers.
        :return: Dictionary with column names as keys and outlier counts as values.
        """
        outlier_counts = {}
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                z_scores = (self.df[column] - self.df[column].mean()) / self.df[column].std()
                outlier_count = (abs(z_scores) > z_score_threshold).sum()
                outlier_counts[column] = outlier_count

        return outlier_counts

    def outlier_count_iqr_all_fields(self):
        """
        Count outliers in all numeric fields using IQR method.
        
        :return: Dictionary with column names as keys and outlier counts as values.
        """
        outlier_counts = {}
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                q1 = self.df[column].quantile(0.25)
                q3 = self.df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
                outlier_counts[column] = outlier_count

        return outlier_counts

    def create_bubble_chart(self, x_col, y_col, size_col, color_col):
        """
        Create and show a bubble chart.
        
        :param x_col: Column name for the X-axis.
        :param y_col: Column name for the Y-axis.
        :param size_col: Column name for the bubble size.
        :param color_col: Column name for the bubble color.
        """
        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            size=size_col,  # Bubble size represents the size column
            color=color_col, # Bubble color represents the color column
            hover_name=self.df.index, # Hover over the points
            title="Bubble Chart",
            labels={x_col: x_col, y_col: y_col, size_col: size_col, color_col: color_col}
        )

        # Show the plot
        fig.show()

    def analyze_temperature_rh_solar(self, temp_col, rh_col, solar_cols):
        """
        Analyze the relationship between temperature, relative humidity, and solar radiation.
        
        :param temp_col: Column name for temperature.
        :param rh_col: Column name for relative humidity.
        :param solar_cols: List of column names for solar radiation.
        """
        # 1. Create a scatter plot of temperature vs. relative humidity
        fig = px.scatter(self.df, x=temp_col, y=rh_col, 
                        title="Temperature vs. Relative Humidity")
        fig.show()

        # 2. Compute the correlation coefficient between temperature and relative humidity
        temp_rh_corr = self.df[temp_col].corr(self.df[rh_col])
        print(f"Correlation between temperature and relative humidity: {temp_rh_corr:.2f}")

        # 3. Create a 3D scatter plot for each solar radiation column
        for solar_col in solar_cols:
            fig = go.Figure(data=[go.Scatter3d(
                x=self.df[temp_col],
                y=self.df[rh_col],
                z=self.df[solar_col],
                mode='markers'
            )])
            fig.update_layout(
                title=f'3D Scatter Plot of Temperature, Relative Humidity, and {solar_col}',
                scene=dict(
                    xaxis_title='Temperature',
                    yaxis_title='Relative Humidity',
                    zaxis_title=solar_col
                )
            )
            fig.show()

        # 4. Perform linear regression to model the relationship between temperature, 
        #    relative humidity, and each solar radiation column
        for solar_col in solar_cols:
            X = self.df[[temp_col, rh_col]]
            y = self.df[solar_col]
            model = LinearRegression()
            model.fit(X, y)
            print(f"Regression coefficients for {solar_col}: "
                  f"intercept={model.intercept_:.2f}, "
                  f"temperature={model.coef_[0]:.2f}, "
                  f"relative_humidity={model.coef_[1]:.2f}")
