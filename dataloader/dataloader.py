"""Data Loader class"""
import pandas as pd

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads subset of data for demonstration purposes"""
        return pd.read_csv(data_config.path, index_col=0)

    @staticmethod
    def descale(descaler, values):
      values_2d = values
      return descaler.inverse_transform(values_2d)
