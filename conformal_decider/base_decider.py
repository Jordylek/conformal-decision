from abc import ABC, abstractmethod

import numpy as np


class BaseConformalDecider(ABC):

    def __init__(self, epsilon, eta, horizon):
        self.epsilon = epsilon
        self.eta = eta
        self.horizon = horizon

    def _step(self, previous_lambda, loss) -> float:
        return previous_lambda + self.eta * (self.epsilon - loss)

    @abstractmethod
    def loss_function(self, decision, target) -> float:
        pass

    @abstractmethod
    def decision_function(self, features, lam):
        pass

    def update_features(self, t, features, target):
        return None

    def update_target(self, t):
        return None

    def update_others(self, t, features, target, decision, others, lam):
        return {}

    def run_decider(self, initial_lam):
        lambdas = np.zeros(self.horizon + 1)
        losses = np.zeros(self.horizon)
        decisions = []
        lambdas[0] = initial_lam
        features, target = None, None
        others = {}
        for t in range(self.horizon):
            # Update features and target
            features = self.update_features(t, features=features, target=target)
            target = self.update_target(t)

            # Compute decision
            lam = lambdas[t]
            decision = self.decision_function(features=features, lam=lam)
            others = self.update_others(t, features=features, target=target,
                                        others=others, decision=decision,
                                        lam=lam)

            decisions.append(decision)

            # Compute loss
            loss = self.loss_function(decision=decision, target=target)
            losses[t] = loss

            # Update lambda
            lambdas[t + 1] = self._step(previous_lambda=lambdas[t], loss=loss)

        return {"lambdas": lambdas,
                "decisions": decisions,
                "losses": losses,
                **others}

