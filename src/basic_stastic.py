import pandas as pd
class BasicStas:
    def __init__(self,df):
        self.df = df
    def basic_stats(self):
        return self.df.describe()
    def check_missing_count(self):
        return self.df.isnull().sum()
    def check_outlier_count(self):
        Q1 = self.df.quantile(0.25, numeric_only=True)
        Q3 = self.df.quantile(0.75, numeric_only=True)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        aligned_df, aligned_lower_bound = self.df.align(lower_bound, axis=1, copy=False)
        aligned_df, aligned_upper_bound = aligned_df.align(upper_bound, axis=1, copy=False)

        outliers = (aligned_df < aligned_lower_bound) | (aligned_df > aligned_upper_bound)
        outlier_counts = outliers.sum()

        return outlier_counts
    def check_outlier_remove(self):
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return self.df
    def check_outlier_bas_columns(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        return outliers

    def check_duplicate_count(self):
        return self.df.duplicated().sum()
    def check_duplicate_remove(self):
        self.df = self.df.drop_duplicates()
        return self.df
    def check_data_type(self):
        return self.df.dtypes
    def check_column_count(self):
        return len(self.df.columns)
    def check_row_count(self):
        return len(self.df)
    def check_column_names(self):
        return list(self.df.columns)