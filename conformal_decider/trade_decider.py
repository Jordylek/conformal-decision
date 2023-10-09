import numpy as np
from .base_decider import BaseConformalDecider
from scipy import stats


class TradeDecider(BaseConformalDecider):
    def __init__(self, horizon, eta, epsilon, target_returns, predicted_returns, dt, volatilities, loss_type="drawdown"):
        super().__init__(horizon=horizon, eta=eta, epsilon=epsilon)
        self.target_returns = target_returns
        self.predicted_returns = predicted_returns
        self.dt = dt
        self.volatilities = volatilities
        self.loss_type = loss_type
        assert self.loss_type in ("drawdown", "variance")

    @staticmethod
    def confidence_interval(prediction, alpha, vol, dt):
        z = stats.norm.ppf(1 - np.clip(alpha, 0, 1) / 2)
        return np.sort(prediction + vol * np.sqrt(dt) * z * np.array([-1, 1]))

    @staticmethod
    def get_action(interval):
        if interval[0] > 0:
            return 1
        elif interval[1] < 0:
            return -1
        else:
            return 0

    def loss_function(self, decision, target):
        """
        :param decision:
        :param target:
        :return:
        """
        action = decision["action"]
        if self.loss_type == "drawdown":
            return np.clip(-target * action, 0, None)
        elif self.loss_type == "variance":
            return np.abs(target * action)

    def decision_function(self, features, lam):
        """
        Use the confidence set build around the predicted return to decide whether to buy or sell.
        Buy if the confidence inverval is above 0
        Sell if the confidence interval is below 0
        """
        prediction = features["prediction"]
        vol = features["vol"]
        confidence_interval = self.confidence_interval(prediction=prediction, alpha=lam, vol=vol, dt=self.dt)
        return {"confidence_interval": confidence_interval,
                "action": self.get_action(interval=confidence_interval)}

    def update_target(self, t):
        return self.target_returns[t]

    def update_features(self, t, features, target):
        return {"prediction": self.predicted_returns[t],
                "vol": self.volatilities[t]}

    def update_others(self, t, features, target, decision, others, lam):
        if t == 0:
            others["coverage"] = np.full_like(self.target_returns, False)
            others["returns"] = np.zeros_like(self.target_returns)

        interval = decision["confidence_interval"]
        others["coverage"][t] = interval[0] <= target <= interval[1]
        others["returns"][t] = target * decision["action"]

        return others


class ACITradeDecider(TradeDecider):

    def __init__(self, horizon, eta, epsilon, target_returns, predicted_returns, dt, volatilities):
        super().__init__(horizon=horizon, eta=eta, epsilon=epsilon, target_returns=target_returns,
                         predicted_returns=predicted_returns, dt=dt, volatilities=volatilities)

    def loss_function(self, decision, target):
        interval = decision["confidence_interval"]
        cover = interval[0] <= target <= interval[1]
        return 1 - cover
