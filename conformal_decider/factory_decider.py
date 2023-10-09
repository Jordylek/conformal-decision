import numpy as np
from .base_decider import BaseConformalDecider
from scipy import stats


def get_item_rate(param):
    return np.sqrt(np.clip(param, 1e-3, None))


class FactoryDecider(BaseConformalDecider):
    def __init__(self, horizon, eta, epsilon, poisson_coef, binomial_coef):
        super().__init__(horizon=horizon, eta=eta, epsilon=epsilon)
        self.poisson_coef = poisson_coef
        self.binomial_coef = binomial_coef

    @staticmethod
    def number_defects(number_items, defect_prob):
        return np.random.binomial(n=number_items, p=defect_prob, size=1)

    @staticmethod
    def number_items(rate):
        return np.random.poisson(lam=rate, size=1)

    def loss_function(self, decision, target):
        if decision[1] == 0:
            return 0
        return decision[0] / decision[1]

    def decision_function(self, features, lam):
        """
        Returns the probability of defecting. It is a beta distribution with parameters lam*features and features.
        :param features: b[t]
        :param lam:
        :return:
        """
        lam = np.clip(lam, 1e-5, None)
        defect_prob = lam * features[1]
        items_rate = get_item_rate(lam) * features[0]
        defect_prob = np.clip(defect_prob, 0, 1)
        num_items = self.number_items(rate=items_rate)
        num_defect = self.number_defects(number_items=num_items, defect_prob=defect_prob)
        return num_defect, num_items

    def update_target(self, t):
        return None

    def update_features(self, t, features, target):
        return self.poisson_coef[t], self.binomial_coef[t]

    def update_others(self, t, features, target, decision, others, lam):
        if t == 0:
            others = {'num_items': np.zeros(self.horizon),
                      'num_defects': np.zeros(self.horizon)}
        others['num_items'][t] = decision[1]
        others['num_defects'][t] = decision[0]
        return others


class FactoryBandit:

    def __init__(self, horizon, arms, epsilon, poisson_coef, binomial_coef):
        self.horizon = horizon
        self.epsilon = epsilon
        self.poisson_coef = poisson_coef
        self.binomial_coef = binomial_coef
        self.arms = arms
        self.num_arms = len(arms)
        self.max_items = self.poisson_coef * 2

    @staticmethod
    def number_defects(number_items, defect_prob):
        return np.random.binomial(n=number_items, p=defect_prob, size=1)

    @staticmethod
    def number_items(rate):
        return np.random.poisson(lam=rate, size=1)

    def loss_function(self, decision, target):
        if decision[1] == 0:
            return 0
        return decision[0] / decision[1]

    def decision_function(self, arm, features=None):
        """
        Returns the probability of defecting. It is a beta distribution with parameters lam*features and features.
        :param features: b[t]
        :param lam:
        :return:
        """
        defect_prob = arm * features[1]
        items_rate = get_item_rate(arm) * features[0]
        defect_prob = np.clip(defect_prob, 0, 1)
        num_items = self.number_items(rate=items_rate)
        num_defect = self.number_defects(number_items=num_items, defect_prob=defect_prob)
        return num_defect, num_items

    def select_arm_idx(self, t, counts, losses_per_arm, utilities_per_arm, emp_risk, selection_type):
        # Implement  UCB
        if t < len(self.arms):
            return t
        else:
            if selection_type == "max_util":
                ucb = utilities_per_arm / counts + np.sqrt(2 * np.log(t) / counts)
                arm = np.argmax(ucb)
            elif selection_type == "min_loss":
                lcb = losses_per_arm / counts - np.sqrt(2 * np.log(t) / counts)
                arm = np.argmin(lcb)
            elif selection_type == "target_loss":
                loss = np.abs(losses_per_arm / counts - self.epsilon)
                lcb = loss - np.sqrt(2 * np.log(t) / counts)
                arm = np.argmin(lcb)
            elif selection_type == "adaptive_learn":
                ucb = utilities_per_arm / counts + np.sqrt(2 * np.log(t) / counts)
                valid_arm = (emp_risk * counts + losses_per_arm) / (counts + 1) <= self.epsilon
                if valid_arm.sum() > 0:
                    ucb[~valid_arm] = np.inf
                arm = np.argmax(ucb)
            else:
                raise ValueError("Unknown selection type {}".format(selection_type))
            return arm

    def update_others(self, t, decision, others):
        if t == 0:
            others = {'num_items': np.zeros(self.horizon),
                      'num_defects': np.zeros(self.horizon)}
        others['num_items'][t] = decision[1]
        others['num_defects'][t] = decision[0]
        return others

    def update_features(self, t, features, target):
        return self.poisson_coef[t], self.binomial_coef[t]

    def run_bandits(self, selection_type="max_util"):
        assert selection_type in ("max_util", "min_loss", "target_loss", "adaptive_learn")
        selections = np.zeros(self.horizon)
        counts = np.zeros_like(self.arms)
        losses = np.zeros(self.horizon)
        losses_per_arm = np.zeros_like(self.arms)
        utilities = np.zeros(self.horizon)
        utilities_per_arm = np.zeros_like(self.arms)
        decisions = []
        features, target = None, None
        others = {}
        for t in range(self.horizon):
            # Update features and target
            arm_idx = self.select_arm_idx(t, counts, losses_per_arm, utilities_per_arm,
                                          emp_risk=losses.mean(), selection_type=selection_type)
            arm = self.arms[arm_idx]
            counts[arm_idx] += 1
            selections[t] = arm
            features = self.update_features(t, features=features, target=target)
            decision = self.decision_function(arm=arm, features=features)
            utility = (decision[1] - decision[0]) / self.max_items[t]
            utilities_per_arm[arm_idx] += utility
            utilities[t] = utility
            loss = self.loss_function(decision=decision, target=target)
            losses_per_arm[arm_idx] += loss
            losses[t] = loss
            decisions.append(decision)
            others = self.update_others(t, others=others, decision=decision)

        return {"selections": selections,
                "counts": counts,
                "losses_per_arm": losses_per_arm,
                "losses": losses,
                "decisions": decisions,
                "utilities_per_arm": utilities_per_arm,
                "utilities": utilities,
                **others}
