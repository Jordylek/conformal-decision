import itertools
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from conformal_decider.trade_decider import TradeDecider, ACITradeDecider
import argparse
import pickle as pkl

# n_runs = 100
# n_years = 5
# steps_per_year = 252 * 7
# T = n_years * steps_per_year
# dt = 1 / steps_per_year

# mu = 0.08

# volatilities = np.full(T, 0.2)
# pred_volatilities = np.full(T, 0.2)


# epsilon = 0.25 / steps_per_year
# cc_eta = 2
# cc_initial_lam = 0.1

# var_epsilon = 0.25 / steps_per_year

# aci_eta = 0.005
# aci_initial_lam = 0.1
# aci_epsilon = 0.1

# rhos = [-0.1, -0.05,  0.0, 0.05, 0.1, 0.2]

# np.random.seed(42)
# brownians = np.random.randn(2, n_runs, T)


def run_deciders(run_id, args, brownians):
    returns = args.mu * args.dt + args.vol * \
        np.sqrt(args.dt) * brownians[0, run_id]

    corr_brownian = args.rho * brownians[0, run_id] + \
        np.sqrt(1 - args.rho ** 2) * brownians[1, run_id]
    predictions = args.mu * args.dt + args.vol * \
        np.sqrt(args.dt) * corr_brownian

    volatilities = np.full(args.T, args.vol)
    cc_decider = TradeDecider(horizon=args.T, eta=args.cc_eta, epsilon=args.epsilon, target_returns=returns,
                              predicted_returns=predictions, dt=args.dt, volatilities=volatilities)

    aci = ACITradeDecider(horizon=args.T, eta=args.aci_eta, epsilon=args.aci_epsilon, target_returns=returns,
                          predicted_returns=predictions, dt=args.dt, volatilities=volatilities)

    cc_output = cc_decider.run_decider(initial_lam=args.cc_initial_lam)
    aci_output = aci.run_decider(initial_lam=args.aci_initial_lam)

    greedy_returns = np.sign(predictions) * returns

    outputs = {"cc": cc_output,
               "aci": aci_output,
               "others": {"true_returns": returns,
                          "predictions": predictions},
               "rho": args.rho,
               "run_id": run_id,
               }

    ans = {"avg_return_per_year": {"cc": cc_output["returns"].sum() / args.n_years,
                                "aci": aci_output["returns"].sum() / args.n_years,
                                "buy_and_hold": returns.sum() / args.n_years,
                                "greedy": (np.sign(predictions) * returns).sum() / args.n_years,
                                },

           "avg_loss_per_year": {"cc": -np.clip(cc_output["returns"], None, 0).sum() / args.n_years,
                              "aci": -np.clip(aci_output["returns"], None, 0).sum() / args.n_years,
                              "buy_and_hold": -np.clip(returns, None, 0).sum() / args.n_years,
                              "greedy": -np.clip(greedy_returns, None, 0).sum() / args.n_years,
                              },

           "return_std": {"cc": cc_output["returns"].std() * np.sqrt(args.steps_per_year),
                          "aci": aci_output["returns"].std() * np.sqrt(args.steps_per_year),
                          "buy_and_hold": returns.std() * np.sqrt(args.steps_per_year),
                          "greedy": (np.sign(predictions) * returns).std() * np.sqrt(args.steps_per_year),
                          },

           "avg_lambda": {"cc": cc_output["lambdas"].mean(),
                          "aci": aci_output["lambdas"].mean(),
                          "buy_and_hold": np.nan,
                          "greedy": np.nan,
                          },
           }

    return ans, outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--n_years", type=int, default=5)
    parser.add_argument("--steps_per_day", type=int, default=7)
    parser.add_argument("--mu", type=float, default=0.08)
    parser.add_argument("--trend", type=float, default=0.04)
    parser.add_argument("--vol", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--cc_eta", type=float, default=2)
    parser.add_argument("--cc_initial_lam", type=float, default=0.1)
    parser.add_argument("--aci_eta", type=float, default=0.005)
    parser.add_argument("--aci_initial_lam", type=float, default=0.1)
    parser.add_argument("--aci_epsilon", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.steps_per_year = args.steps_per_day * 252
    args.T = args.n_years * args.steps_per_year
    args.dt = 1 / args.steps_per_year
    args.epsilon /= args.steps_per_year  # Convert to epsilon per step

    np.random.seed(args.seed)
    brownians = np.random.randn(2, args.n_runs, args.T)

    # out = run_deciders(0, args, brownians)

    all_runs = Parallel(n_jobs=8)(delayed(run_deciders)(
        i, args, brownians) for i in range(args.n_runs))
    print()
    metrics = defaultdict(lambda: defaultdict(lambda: np.zeros((args.n_runs))))
    for run in all_runs:
        run_id = run[1]["run_id"]
        for metric, values in run[0].items():
            for key, value in values.items():
                metrics[key][metric][run_id] = value

    metrics = {key: dict(values) for key, values in metrics.items()}

    for key, values in metrics.items():
        for metric, value in values.items():
            print(key, metric, value.mean(axis=0))
    print("DONE")
