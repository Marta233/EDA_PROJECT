Here’s a README file for your solar data analysis code:

---

# Solar Data Analysis Toolkit


## Overview
MoonLight Energy Solutions aims to develop a strategic approach to significantly enhance its operational efficiency and sustainability through targeted solar investments. As an Analytics Engineer at MoonLight Energy Solutions, your task is to perform a quick analysis of an environmental measurement provided by the engineering team and translate your observation as a strategy report. Your analysis should focus on identifying key trends and learn valuable insights that will support your data-driven case - your recommendation based on the statistical analysis and EDA.  In particular, your analysis and recommendation must present a strategy focusing on identifying high-potential regions for solar installation that align with the company's long-term sustainability goals. Your report should provide an insight to help realize the overarching objectives of MoonLight Energy Solutions. the datset is a solar radiation data of Benin, Sierra Leone, and Togo.

The structure of the data is as follows
Timestamp (yyyy-mm-dd hh:mm): Date and time of each observation.
GHI (W/m²): Global Horizontal Irradiance, the total solar radiation received per square meter on a horizontal surface.
DNI (W/m²): Direct Normal Irradiance, the amount of solar radiation received per square meter on a surface perpendicular to the rays of the sun.
DHI (W/m²): Diffuse Horizontal Irradiance, solar radiation received per square meter on a horizontal surface that does not arrive on a direct path from the sun.
ModA (W/m²): Measurements from a module or sensor (A), similar to irradiance.
ModB (W/m²): Measurements from a module or sensor (B), similar to irradiance.
Tamb (°C): Ambient Temperature in degrees Celsius.
RH (%): Relative Humidity as a percentage of moisture in the air.
WS (m/s): Wind Speed in meters per second.
WSgust (m/s): Maximum Wind Gust Speed in meters per second.
WSstdev (m/s): Standard Deviation of Wind Speed, indicating variability.
WD (°N (to east)): Wind Direction in degrees from north.
WDstdev: Standard Deviation of Wind Direction, showing directional variability.
BP (hPa): Barometric Pressure in hectopascals.
Cleaning (1 or 0): Signifying whether cleaning (possibly of the modules or sensors) occurred.
Precipitation (mm/min): Precipitation rate measured in millimeters per minute.
TModA (°C): Temperature of Module A in degrees Celsius.
TModB (°C): Temperature of Module B in degrees Celsius.
Comments: This column is designed for any additional notes.

This toolkit provides a set of Python classes and functions for cleansing, visualizing, and analyzing solar data. It includes methods to handle negative values, remove unnecessary columns, and perform various statistical analyses and visualizations on solar sensor data.

## Installation

Before using this toolkit, ensure you have the following Python packages installed:

```bash
pip install pandas plotly scikit-learn numpy scipy matplotlib seaborn
```

## Usage

### Data Cleansing

The `Data_cleansing` class is designed to handle basic data preprocessing tasks, such as imputing negative values with zeros and removing specified columns.

**Example:**

```python
import pandas as pd

# Assuming 'data' is a pandas DataFrame containing your solar data
cleanser = Data_cleansing(data)

# Impute negative values with zero in specified columns
cleanser.impute_negative_with_zero(['GHI', 'DNI', 'DHI'])

# Remove unwanted columns
cleanser.remove_columns(['Comments', 'TModA'])
```

### Data Visualization

The `Visual` class offers various methods for visualizing and analyzing solar data.

**Initialization:**

```python
visual = Visual(data)
```

**Plot a Line Graph:**

```python
visual.plot_line_graph(x='timestamp', y='GHI')
```

**Evaluate Cleaning Impact on Sensor Readings:**

```python
mean_values, ttest_results = visual.evaluate_cleaning_impact(timestamp_col='timestamp', cleaning_col='Cleaning', mod_cols=['ModA', 'ModB'])
```

**Plot a Correlation Heatmap:**

```python
visual.plot_correlation_heatmap(features=['GHI', 'DNI', 'DHI', 'ModA', 'ModB'])
```

**Plot a Scatter Matrix:**

```python
visual.plot_scatter_matrix(features=['GHI', 'DNI', 'DHI'])
```

**Plot Histograms:**

```python
visual.plot_histograms(variables=['GHI', 'DNI', 'DHI'])
```

**Count Outliers Using Z-Score:**

```python
outliers = visual.outlier_count_z_score_all_fields(z_score_threshold=3)
```

**Create a Bubble Chart:**

```python
visual.create_bubble_chart(x_col='GHI', y_col='DNI', size_col='ModA', color_col='ModB')
```

**Analyze Temperature, Relative Humidity, and Solar Radiation:**

```python
visual.analyze_temperature_rh_solar(temp_col='Tamb', rh_col='RH', solar_cols=['GHI', 'DNI'])
```

## Running the Code

The code can be run in any Python environment, including Jupyter Notebooks. Simply instantiate the classes with your data and call the relevant methods as shown in the examples above.

---

This README provides a comprehensive guide to using the solar data analysis toolkit. You can customize it further based on your specific use cases or project requirements.
