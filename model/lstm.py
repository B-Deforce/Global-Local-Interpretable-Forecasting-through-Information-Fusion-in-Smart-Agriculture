
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet
import pytorch_lightning as pl
import pandas as pd
import torch.nn as nn
import torch.optim as optim



class LSTM_dataprep():
    """Class to build TFT model"""
    def __init__(self, model_config, full_df_pure, full_df=None):
        """
        Args:
            model_config: config file for model init
            full_df_pure: df without rogue sensors
            full_df: df with rogue sensors
        """
        self.config = model_config
        self.dataset_pure = full_df_pure
        self.dataset_full = full_df

    def build_dataset(
        self,
        train_val_combo = False,
        ):
        """
        Args: 
            train_val_combo: whether to build dataset with train and val as one or split off
            This is handy when one wants to e.g. refit the final model on both train and val to then test on final 
            test set.
        """
        # adapted from https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
        # define Pytorch dataset
        max_encoder_length = self.config.max_encoder_length # max lookback window
        max_prediction_length = self.config.max_pred_length

        if self.config.drop_rogue:
            df = self.dataset_pure
        else:
            df = self.dataset_full

        # set appropriate dtypes for TFT
        df["measurement_month"] = pd.to_datetime(df["measurement_date"]).dt.month.astype(str).astype("category")
        df["measurement_year"] = pd.to_datetime(df["measurement_date"]).dt.year.astype(str).astype("category")
        df["depth_cm"] = df["depth_cm"].astype(str).astype("category")
        df["extreme_sensor_value"] = (df["soil_moisture"] <= -199.).astype(int).astype(str).astype("category")

        # train set: everything from 2007 (this way the network sees at least one full growing cycle)
        # + all data up until the last two f-horizons for validation and the last horizon for test of '08 and '09
        self.train_df = df[lambda x: (x.time_idx < x.time_idx_max - 2*5) | (x.measurement_year == "2007")].copy()
        self.val_df = df[lambda x: (x.time_idx < x.time_idx_max - 5) & (x.measurement_year != "2007")].copy()
        self.test_df = df[lambda x: (x.measurement_year != "2007")].copy()
        self.train_val_df = df[lambda x: (x.time_idx < x.time_idx_max - 5) | (x.measurement_year == "2007")].copy()

        if train_val_combo:
            self.fit_df = self.train_val_df.copy()
        else:
            self.fit_df = self.train_df.copy()

        # scale continuous cols to avoid vanishing/exploding gradients
        self.minmax_scaler = MinMaxScaler(feature_range=(-1,1))
        col_cont = [
                'soil_moisture',
                'precip_daily',
                'irrigation_amount',
                "soil_temp",
                "ETo",
                "soil_moisture_diff"
                ]

        # MinMaxScaling
        self.fit_df.loc[:,[f"{i}_scaled" for i in col_cont]] = self.minmax_scaler.fit_transform(self.fit_df[col_cont])
        self.test_df.loc[:,[f"{i}_scaled" for i in col_cont]] = self.minmax_scaler.transform(self.test_df[col_cont])
        self.val_df.loc[:,[f"{i}_scaled" for i in col_cont]] = self.minmax_scaler.transform(self.val_df[col_cont])
        
        self.training = TimeSeriesDataSet(
            # ensures last 7 days of each sensor are not used in training (saved for val)
            data=self.fit_df,
            time_idx="time_idx",
            target="soil_moisture_diff_scaled",
            # weight="weight", # add weight to more recent values frm dom knowledge
            group_ids=["sensor_id", "measurement_year"], # grouped by sensor and year
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=[
                "soil_moisture_diff_scaled",
                "irrigation_amount_scaled", 
                "soil_temp_scaled",
                "ETo_scaled",
                "precip_daily_scaled"
                ],
            allow_missing_timesteps=True, # will autofill with ffill
            target_normalizer=None,
            scalers={
                "irrigation_amount_scaled": None, 
                "soil_temp_scaled": None,
                "ETo_scaled": None,
                "precip_daily_scaled": None,
            }
        )
        batch_size = self.config.batch_size
        
        # val: predict last seven days in training data
        train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)

        # refit on both training and validation data before reporting test performance
        val = TimeSeriesDataSet.from_dataset(self.training, self.val_df, predict=True, stop_randomization=True)
        val_dataloader = val.to_dataloader(train=False, batch_size=batch_size*10, num_workers=4)

        test = TimeSeriesDataSet.from_dataset(self.training, self.test_df, predict=True, stop_randomization=True)
        test_dataloader = test.to_dataloader(train=False, batch_size=batch_size*10, num_workers=4)

        return self.training, train_dataloader, val_dataloader, test_dataloader

class SoilMoist_LSTM(nn.Module):

  def __init__(self, model_config):
    super().__init__()
    self.config = model_config

    self.lstm = nn.LSTM(
        input_size=self.config.n_features,
        hidden_size=self.config.n_hidden,
        batch_first=True,
        num_layers=self.config.n_layers,
        dropout=self.config.dropout # 0.2
    )

    self.regressor = nn.Linear(self.config.n_hidden, 5)

  def forward(self, x):
    self.lstm.flatten_parameters() # GPU optimization

    _, (hidden, _) = self.lstm(x)
    out = hidden[-1] # returns output of last LSTM layer in order to pass it to fully connected layer

    return self.regressor(out)

# create lightning module that uses LSTM

class SoilMoistPredictor(pl.LightningModule):

  def __init__(self, model_config):
    super().__init__()
    self.model = SoilMoist_LSTM(model_config=model_config)
    self.criterion = nn.MSELoss()

  def forward(self, x, labels=None):
    output = self.model(x)
    loss = 0
    if labels is not None: 
      loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    x, y = batch
    sequences = x["encoder_cont"]
    labels = y[0]

    loss, outputs = self(sequences, labels)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    sequences = x["encoder_cont"]
    labels = y[0]

    loss, outputs = self(sequences, labels)
    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    sequences = x["encoder_cont"]
    labels = y[0]

    loss, outputs = self(sequences, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return optim.AdamW(self.parameters(), lr=0.001)