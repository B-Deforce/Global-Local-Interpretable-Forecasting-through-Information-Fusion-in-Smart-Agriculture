"""VAR model class"""
import numpy as np
import statsmodels.api as sm

from utils.config import Config
from dataloader.dataloader import DataLoader

#oos_pred_l = []
#f_true = []
#lag_order_l = []
#var_MAE_year = {
#    "2008": [],
#    "2009": []
#    }
#var_MSE_year = {
#    "2008": [],
#    "2009": []
#    }
#var_MAE_raw = []
#var_MSE_raw = []
#var_MDA_raw = []
#ts_length = []
#errors = []
#adf_results = {
#    'irrigation_amount': [],
#    'soil_temp': [],
#    'precip_daily': [],
#    'ETo':[]
#               }

class MultiVAR():
    def __init__(self, model_config, df):
        self.fh = model_config.fh
        self.dataset = df

    def load_data(self):
        self.dataset = self.dataset[self.dataset["measurement_year"] != '2007']
        self.sensor_keys = self.dataset[['measurement_year',
                                         'sensor_id']].drop_duplicates().values

    def train(self):
        """Fits a VAR model for each time-series
        Returns: 
            local_fit List[Tuple]: returns a list of tuples containing
            - fitted model
            - lag order
            - time-series on which the model was fitted
            - results from ADF test
        """
        local_fit = []
        for i, j in enumerate(self.sensor_keys):
            year, id = j[0], j[1]
            single_ts = self.dataset[((self.dataset["measurement_year"] == year) & (self.dataset["sensor_id"] == id))].copy()
            # placeholder
            single_ts.loc[:, "sm_diff"] = 0
            # due to differencing first row is invalid for analysis
            single_ts.iloc[1:,-1] = single_ts["soil_moisture"][1:].values - single_ts["soil_moisture"][:-1].values
            if len(single_ts) < 20:
              print(f"sensor_id {id} ({year}) dropped due to insufficient length")
              continue
            self.t_var_reals = ['sm_diff',
                           'irrigation_amount',
                           'soil_temp',
                           'precip_daily',
                           'ETo']
            # adf test of stationarity: if >.05 --> non-stationary
            for k in self.t_var_reals[1:]:
                  adf_res = sm.tsa.stattools.adfuller(single_ts[k], regression="ct")
                  if adf_res[1] > 0.05:
                    single_ts.iloc[1:].loc[:,k] = single_ts[k][1:].values - single_ts[k][:-1].values
            data = single_ts[self.t_var_reals].values[1:-self.fh,:] # drop first row due to differencing and fh as test set
            try:
              var_model = sm.tsa.VAR(data)
              var_fit = var_model.fit(ic="aic", maxlags=4, trend="c")
              lag_order = var_fit.k_ar
              if lag_order == 0:
                print(f"sensor_id {id} ({year}) dropped due to insufficient length")
                continue
            except (np.linalg.LinAlgError, ValueError) as error:
              lag_order = 1
            # refit on full data after determining lag order
            # we want to exclude the last 5 days which we wish to forecast as well as the needed lag order
            # we also drop the first row due to differencing
            data = single_ts[self.t_var_reals].values[1:-self.fh-lag_order,:]
            var_model = sm.tsa.VAR(data)
            var_fit = var_model.fit(lag_order, trend="c")
            local_fit.append((var_fit, lag_order, single_ts, adf_res[1]))
        return local_fit

    def predict(self, local_fit):
        """ Predicts using a fitted sm model on unseen test-data
        Args: 
            all_ts (List): list tuples containing the results from model training
        """
        result = []
        for i in local_fit:
            fitted_model = i[0]
            lag_order = i[1]
            single_ts = i[2]
            # forecast with unseen lagged data
            # substract lag_order and f-horizon from last value - this is the test set
            f = fitted_model.forecast(single_ts[self.t_var_reals].values[-self.fh-lag_order:-self.fh,:], self.fh)
            # out of sample prediction: add differenced values recursively to "f-horizon - 1" (the last seen value)
            oos_pred = [np.sum(f[:i+1,0]) for i in range(f.shape[1])] + single_ts["soil_moisture"].values[-self.fh-1]
            true = single_ts["soil_moisture"].values[-self.fh:]
            result.append((oos_pred, true))
        return result

        # calc MAE
        assert len(oos_pred) == len(single_ts["soil_moisture"].values[-self.fh:])
        mae = np.abs(oos_pred - single_ts["soil_moisture"].values[-self.fh:]).mean()
        mse = ((oos_pred - single_ts["soil_moisture"].values[-self.fh:])**2).mean()
        mda = (np.sign(oos_pred[1:] - oos_pred[:-1]) == np.sign(single_ts["soil_moisture"].values[-self.fh+1:] - single_ts["soil_moisture"][-self.fh:-1])).astype(int)
        var_MAE_year[year].append(mae)
        var_MSE_year[year].append(mse)
        var_MAE_raw.append(mae)
        var_MSE_raw.append(mse)
        var_MDA_raw.append(mda)
        f_true.append((id, year, single_ts["soil_moisture"].values[-self.fh:]))

        # plot results
        if counter / 10 == 1.:
            counter = 0
            print(f"VAR_MAE: {mae}")
            print(f"original length of ts: {len(single_ts)}")
            print(f"length of ts minus 1st row for fh={self.fh} and lag={lag_order}: {data.shape[0]}")
            print(f"len of fitted values, first {lag_order} rows are dropped to include lag: {var_fit.fittedvalues.shape[0]}")
            plt.plot(y_pred, label="y_hat")
            plt.plot(single_ts["soil_moisture"].values[1+lag_order:], label="y_true")
            plt.plot(range(len(single_ts["soil_moisture"].values[1+lag_order:]) - (self.fh-1), # +1 for zero-indexing
                            len(single_ts["soil_moisture"].values[1+lag_order:])+1),
                        oos_pred,
                        label="OOS_y_hat")
        #var_fit.plot_forecast(10)
            plt.legend()
            plt.show()
        counter += 1
