from collections import defaultdict
import numpy as np
from joblib import Parallel, delayed
from conformal_decider.factory_decider import FactoryDecider, FactoryBandit
import argparse

# n_runs = 1000
# T = 2000
# init_lam = 0.1
# eta = 0.1
# epsilon = 0.05

# poisson_coef = 10
# binomial_coef = 0.75

ARMS = np.arange(0.05, 0.6, 0.05)


SELECTION_TYPES = ("max_util", "min_loss", "target_loss")
output_bandit = {}



def run_deciders(run_id, args, decider, decider_fix,
                 bandit_decider, selection_types=SELECTION_TYPES):
    np.random.seed(args.seed + run_id)

    outputs = {}
    outputs["ours"] = decider.run_decider(initial_lam=args.init_lam)
    outputs["fixed_eps"] = decider_fix.run_decider(initial_lam=args.fix_lambda_fix_eps)
    outputs["fixed_max_items"] = decider_fix.run_decider(initial_lam=args.fix_lambda_max_items)

    for selection_type in selection_types:
        out = bandit_decider.run_bandits(selection_type=selection_type)
        out["lambdas"] = out["selections"]
        outputs["bandit-" + selection_type] = out
    ans = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for key, values in outputs.items():
        ans[key]["empirical_risk"] = values["losses"].mean()
        ans[key]["num_items"] = values["num_items"].sum()
        ans[key]["num_good_items"] = (values["num_items"] - values["num_defects"]).sum()
        ans[key]["ratio_defects"] = values["num_defects"].sum() / values["num_items"].sum()
        ans[key]["avg_lambda"] = values["lambdas"].mean()

    for selection_type in SELECTION_TYPES:
        output_bandit[selection_type] = bandit_decider.run_bandits(selection_type=selection_type)
    ans = {key: dict(values) for key, values in ans.items()}
    return ans, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--init_lam", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    
    parser.add_argument("--poisson_coef", type=float, default=10)
    parser.add_argument("--binomial_coef", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    poisson_coef = np.full(args.T, args.poisson_coef)
    binomial_coef = np.full(args.T, args.binomial_coef)

    decider = FactoryDecider(horizon=args.T, eta=args.eta, epsilon=args.epsilon,
                            poisson_coef=poisson_coef, binomial_coef=binomial_coef)
    output = decider.run_decider(initial_lam=args.init_lam)

    decider_fix = FactoryDecider(horizon=args.T, eta=0, epsilon=args.epsilon,
                                poisson_coef=poisson_coef, binomial_coef=binomial_coef)

    args.fix_lambda_fix_eps = args.epsilon / binomial_coef.mean()
    args.fix_lambda_max_items = 1 / (3 * binomial_coef.mean())
    bandit_decider = FactoryBandit(horizon=args.T, epsilon=args.epsilon, arms=ARMS,
                                poisson_coef=poisson_coef, binomial_coef=binomial_coef)

    # run_deciders(45, args=args, decider=decider, decider_fix=decider_fix, bandit_decider=bandit_decider)
    all_runs = Parallel(n_jobs=8)(delayed(run_deciders)(i, args=args, decider=decider,
     decider_fix=decider_fix, bandit_decider=bandit_decider) for i in range(args.n_runs))
    metrics = defaultdict(lambda: defaultdict(lambda: np.zeros(args.n_runs)))
    for run_id, run in enumerate(all_runs):
        for key, values in run[0].items():
            for metric, value in values.items():
                metrics[key][metric][run_id] = value

    metrics = {key: dict(values) for key, values in metrics.items()}

    for key, values in metrics.items():
        for metric, value in values.items():
            print(key, metric, value.mean())

