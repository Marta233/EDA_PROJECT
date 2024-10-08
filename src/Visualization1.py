import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import ttest_ind  # For statistical comparison

class Visualization1:
    def __init__(self, df):
        self.df = df

    def plot_line_graph(self, x, y):
        self.df[x] = pd.to_datetime(self.df[x], format='mixed', dayfirst=True, errors='coerce')
        fig = px.line(self.df, x=x, y=y)
        fig.show()

    def evaluate_cleaning_impact(self, timestamp_col, cleaning_col, mod_cols):
        # Convert timestamp column to datetime if not already
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])

        # Group by Cleaning status and compute mean of ModA and ModB
        mean_values = self.df.groupby(cleaning_col)[mod_cols].mean().reset_index()
        print(f"Mean values by {cleaning_col} status:\n", mean_values)

        # Plotting the time series of ModA and ModB with cleaning events
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

    def plot_correlation_heatmap(self, features):
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
        for field in list_fields:
            fig = px.histogram(self.df, x=field, nbins=20, title=f'Histogram for {field}')
            fig.show()

    def analyze_z_score(self, field):
        mean = self.df[field].mean()
        std_dev = self.df[field].std()
        z_scores = (self.df[field] - mean) / std_dev
        return z_scores

    def bubble_chart(self, x, y, size_col, color_col, title='Bubble Chart'):
        """
        Create a bubble chart with Plotly.

        Parameters:
        - x (str): The column name for the x-axis.
        - y (str): The column name for the y-axis.
        - size_col (str): The column name for bubble size.
        - color_col (str): The column name for bubble color.
        - title (str): The title of the chart.

        Returns:
        - fig (plotly.graph_objects.Figure): The created bubble chart figure.
        """
        fig = px.scatter(
            self.df,
            x=x,
            y=y,
            size=size_col,
            color=color_col,
            hover_name=self.df.index,
            labels={
                x: x,
                y: y,
                size_col: size_col,
                color_col: color_col
            },
            title=title
        )

        # Update layout for better visual appeal
        fig.update_layout(
            legend_title=color_col,
            coloraxis_colorbar_title=color_col,
            xaxis_title=x,
            yaxis_title=y
        )

        # Show the plot
        fig.show()

    def temperature_analysis(self):
        """
        Analyzes the influence of Relative Humidity (RH) on temperature readings and solar radiation.

        This function generates scatter plots of RH vs. Tamb (temperature), RH vs. GHI, RH vs. DNI, and RH vs. DHI,
        and overlays linear regression lines to analyze the relationship.

        Returns:
        - None: Displays the scatter plots with linear regression lines.
        """

        # Check if required columns exist
        required_cols = ['RH', 'Tamb', 'GHI', 'DNI', 'DHI']
        if not all(col in self.df.columns for col in required_cols):
            print("Required columns are missing.")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'RH vs Tamb',
                'RH vs GHI',
                'RH vs DNI',
                'RH vs DHI'
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )

        y_cols = ['Tamb', 'GHI', 'DNI', 'DHI']
        for i, y_col in enumerate(y_cols):
            row = i // 2 + 1
            col = i % 2 + 1

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=self.df['RH'],
                    y=self.df[y_col],
                    mode='markers',
                    name=f'{y_col} vs RH'
                ),
                row=row, col=col
            )

            # Linear regression
            X = self.df[['RH']].dropna()  # Drop NaN values
            y = self.df[y_col].dropna()
            model = LinearRegression().fit(X, y)
            pred = model.predict(X)

            # Add the linear regression line to the plot
            fig.add_trace(
                go.Scatter(
                    x=X['RH'],
                    y=pred,
                    mode='lines',
                    name=f'Regression: {y_col} vs RH'
                ),
                row=row, col=col
            )

        # Update layout
        fig.update_layout(
            title_text='Influence of Relative Humidity on Temperature and Solar Radiation',
            height=800,  # Adjust height of the figure
            width=1000,  # Adjust width of the figure
            showlegend=False
        )

        # Show the figure
        fig.show()
