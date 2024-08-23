import pandas as pd

class Data_cleansing():
    def __init__(self, data):
        self.data = data
    def impute_negative_with_zero(self,col_lists):
        for col in col_lists:
            self.data[col] = self.data[col].apply(lambda x: 0 if x < 0 else x)
    def remove_columns(self, col_lists):
        self.data = self.data.drop(col_lists, axis=1)