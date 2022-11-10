
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import pytorch_lightning as pl
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from pytorch_lightning.loggers import WandbLogger
import wandb
import pandas as pd

class TFT():
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
            # are static over group_ids
            static_categoricals=[
                "orchard_name", 
                "soil_text_0to30cm", 
                "depth_cm",
                "measurement_year",
                "pruning_treatment",
                "irrigation_treatment",
                ],
            # dynamic within group_ids and known in future
            time_varying_known_categoricals=["measurement_month"],
            time_varying_known_reals=[
                "precip_daily_scaled", 
                "ETo_scaled",
                ], 
            time_varying_unknown_reals=[
                "soil_moisture_diff_scaled",
                "irrigation_amount_scaled", 
                "soil_temp_scaled",
                ],
            allow_missing_timesteps=True, # will autofill with ffill
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=False,
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

    def build_model(
        self, 
        training, 
        wandb_logger, 
        callbacks: list=None
        ):
    # configure network and trainer
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='gpu',
            devices=1,
            limit_train_batches=1.0,
            callbacks=callbacks,
            logger=wandb_logger,
        )
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.config.learning_rate,
            lstm_layers=self.config.lstm_layers,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout, 
            hidden_continuous_size=self.config.hidden_continuous_size,
            output_size=5, # equals number of quantiles
            optimizer=self.config.optimizer,
            loss=QuantileLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
            logging_metrics=[MAE(), RMSE()],
            reduce_on_plateau_patience=20,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        return trainer, tft

    def sweepstaker(self):
        wandb.init()
        config = wandb.config
        wandb_logger = WandbLogger()

        # setup data
        training, train_dataloader, val_dataloader, test_dataloader = self.build_dataset(config, train_val_combo=False)
        
        # setup model - note how we refer to sweep parameters with wandb.config
        trainer, model = self.build_model(config, training, wandb_logger)
        
        # train
        trainer.fit(
            model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader,
            )