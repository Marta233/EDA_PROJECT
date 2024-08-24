import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
class Visual:
    def __init__(self, df):
        self.df = df

    def plot_line_graph(self, x, y):
        # Convert x to datetime if necessary
        self.df[x] = pd.to_datetime(self.df[x], format='mixed', dayfirst=True, errors='coerce')
        fig = px.line(self.df, x=x, y=y)
        st.plotly_chart(fig)

    def evaluate_cleaning_impact(self, timestamp_col, cleaning_col, mod_cols):
        # Convert timestamp column to datetime if not already
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])

        # Group by cleaning status and compute mean of ModA and ModB
        mean_values = self.df.groupby(cleaning_col)[mod_cols].mean().reset_index()
        st.write(f"Mean values by {cleaning_col} status:", mean_values)

        # Plotting the time series of ModA and ModB with cleaning events
        fig = px.line(self.df, x=timestamp_col, y=mod_cols, color=cleaning_col,
                     title='Impact of Cleaning on Sensor Readings Over Time')
        st.plotly_chart(fig)

        # Statistical comparison
        ttest_results = {}
        for mod_col in mod_cols:
            cleaned = self.df[self.df[cleaning_col] == 1][mod_col]
            not_cleaned = self.df[self.df[cleaning_col] == 0][mod_col]

            t_stat, p_value = ttest_ind(cleaned, not_cleaned, equal_var=False)
            ttest_results[mod_col] = {'t_stat': t_stat, 'p_value': p_value}
            st.write(f'Test results for {mod_col} - t-statistic: {t_stat}, p-value: {p_value}')

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

        st.plotly_chart(fig)

    def plot_histograms(self, variables):
        st.pyplot()
        plt.figure(figsize=(14, 10))
        for i, var in enumerate(variables, 1):
            plt.subplot(3, 3, i)
            sns.histplot(self.df[var], kde=True, color='skyblue', bins=30)
            plt.title(f'{var} Histogram')
            plt.xlabel(var)
            plt.ylabel('Frequency')
        plt.tight_layout()
        st.pyplot()

    def outlier_count_z_score_all_fields(self, z_score_threshold=3):
        outlier_counts = {}
        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:
                z_scores = (self.df[column] - self.df[column].mean()) / self.df[column].std()
                outlier_count = (abs(z_scores) > z_score_threshold).sum()
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

        st.plotly_chart(fig)

    def analyze_temperature_rh_solar(self, temp_col, rh_col, solar_cols):
    # Ensure the selected columns are numeric
        if not pd.api.types.is_numeric_dtype(self.df[temp_col]) or not pd.api.types.is_numeric_dtype(self.df[rh_col]):
            st.error("Selected columns for temperature and relative humidity must be numeric.")
            return

        fig = px.scatter(self.df, x=temp_col, y=rh_col, title="Temperature vs. Relative Humidity")
        st.plotly_chart(fig)

        # Calculate and display correlation between temperature and relative humidity
        temp_rh_corr = self.df[temp_col].corr(self.df[rh_col])
        st.write(f"Correlation between temperature and relative humidity: {temp_rh_corr:.2f}")

        for solar_col in solar_cols:
            if not pd.api.types.is_numeric_dtype(self.df[solar_col]):
                st.error(f"Column {solar_col} is not numeric. Skipping 3D scatter plot and regression analysis.")
                continue

            # 3D Scatter Plot
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
            st.plotly_chart(fig)

            # Linear Regression
            X = self.df[[temp_col, rh_col]].dropna()
            y = self.df[solar_col].dropna()
            if X.empty or y.empty or X.shape[0] != y.shape[0]:
                st.error(f"Insufficient data for regression analysis on {solar_col}.")
                continue

            model = LinearRegression()
            model.fit(X, y)
            st.write(f"Regression coefficients for {solar_col}: "
                    f"intercept={model.intercept_:.2f}, "
                    f"temperature={model.coef_[0]:.2f}, "
                    f"relative_humidity={model.coef_[1]:.2f}")

# Streamlit application code
def main():
    st.title('Data Analysis Dashboard')

    # Upload file
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            visual = Visual(df)
            
            # Line Graph
            st.header("Line Graph")
            x_axis = st.selectbox("Select X-axis column for Line Graph", df.columns)
            y_axis = st.selectbox("Select Y-axis column for Line Graph", df.columns)
            if st.button("Plot Line Graph"):
                try:
                    visual.plot_line_graph(x=x_axis, y=y_axis)
                except Exception as e:
                    st.error(f"Error plotting line graph: {e}")

            # Correlation Heatmap
            st.header("Correlation Heatmap")
            features = st.multiselect("Select Features for Correlation Heatmap", df.columns)
            if st.button("Plot Correlation Heatmap"):
                try:
                    visual.plot_correlation_heatmap(features)
                except Exception as e:
                    st.error(f"Error plotting correlation heatmap: {e}")

            # Outliers
            st.header("Outlier Detection")
            z_score_threshold = st.slider("Z-Score Threshold", 0, 10, 3)
            if st.button("Detect Outliers (Z-Score)"):
                try:
                    outliers = visual.outlier_count_z_score_all_fields(z_score_threshold)
                    st.write("Outlier Counts (Z-Score):", outliers)
                except Exception as e:
                    st.error(f"Error detecting outliers (Z-Score): {e}")

            # Bubble Chart
            st.header("Bubble Chart")
            x_col = st.selectbox("Select X-axis column for Bubble Chart", df.columns)
            y_col = st.selectbox("Select Y-axis column for Bubble Chart", df.columns)
            size_col = st.selectbox("Select Size column for Bubble Chart", df.columns)
            color_col = st.selectbox("Select Color column for Bubble Chart", df.columns)
            if st.button("Create Bubble Chart"):
                try:
                    visual.create_bubble_chart(x_col, y_col, size_col, color_col)
                except Exception as e:
                    st.error(f"Error creating bubble chart: {e}")

            # Temperature vs. RH vs. Solar
            st.header("Temperature vs. RH vs. Solar")
            temp_col = st.selectbox("Select Temperature column", df.columns)
            rh_col = st.selectbox("Select Relative Humidity column", df.columns)
            solar_cols = st.multiselect("Select Solar Radiation columns", df.columns)
            if st.button("Analyze Temperature, RH, and Solar"):
                try:
                    visual.analyze_temperature_rh_solar(temp_col, rh_col, solar_cols)
                except Exception as e:
                    st.error(f"Error analyzing temperature, RH, and solar: {e}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

if __name__ == "__main__":
    main()
