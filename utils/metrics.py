
import numpy as np

class Metrics:

  @staticmethod
  def q_risk(predicted, actuals, q):
    q_risk = []
    for i in range(actuals.shape[0]):
      for j in range(actuals.shape[1]):
        y = actuals[i][j]
        y_pred = predicted[i][j]
        q_risk.append(max(q * (y-y_pred), (1-q) * (y_pred-y)))
    return (2 * np.nansum(q_risk) / np.nansum(np.abs(predicted)))  

  @staticmethod
  def MDA(y_pred, y_true):
    return (np.sign(y_pred[:,1:] - y_pred[:,:-1]) == np.sign(y_true[:,1:] - y_true[:,:-1]))  
