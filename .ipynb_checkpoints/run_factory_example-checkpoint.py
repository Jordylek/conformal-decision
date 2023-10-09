from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed

from factory.factory_decider import FactoryDecider2, FactoryBandit

n_runs = 1000
T = 2000
init_lam = 0.1
eta = 0.1
epsilon = 0.05

poisson_coef = 10
binomial_coef = 0.75

poisson_coef = np.full(T, poisson_coef)
binomial_coef = np.full(T, binomial_coef)

decider = FactoryDecider2(horizon=T, eta=eta, epsilon=epsilon,
                          poisson_coef=poisson_coef, binomial_coef=binomial_coef)
output = decider.run_decider(initial_lam=init_lam)

decider_fix = FactoryDecider2(horizon=T, eta=0, epsilon=epsilon,
                              poisson_coef=poisson_coef, binomial_coef=binomial_coef)

fix_lambda_fix_eps = epsilon / binomial_coef.mean()
# fix_lambda_max_items = (1 - binomial_coef.mean()) / (2 * binomial_coef.mean())
fix_lambda_max_items = 1 / (3 * binomial_coef.mean())
# output_fix = decider_fix.run_decider(initial_lam=fix_lambda_fix_eps)
arms = np.arange(0.05, 0.6, 0.05)
bandit_decider = FactoryBandit(horizon=T, epsilon=epsilon, arms=arms,
                               poisson_coef=poisson_coef, binomial_coef=binomial_coef)
selection_types = ("max_util", "min_loss", "target_loss", "adaptive_learn")
output_bandit = {}
for selection_type in selection_types:
    output_bandit[selection_type] = bandit_decider.run_bandits(selection_type=selection_type)


def d1():
    return defaultdict(d2)


def d2():
    return np.zeros(n_runs)


# def d3():
#     return


metrics = defaultdict(d1)


def run_deciders(run_id, decider=decider, decider_fix=decider_fix,
                 bandit_decider=bandit_decider, selection_types=selection_types):
    # global metrics
    # global decider
    # global decider_fix
    # global bandit_decider
    outputs = {}
    outputs["ours"] = decider.run_decider(initial_lam=init_lam)
    outputs["fixed_eps"] = decider_fix.run_decider(initial_lam=fix_lambda_fix_eps)
    outputs["fixed_max_items"] = decider_fix.run_decider(initial_lam=fix_lambda_max_items)

    for selection_type in selection_types:
        out = bandit_decider.run_bandits(selection_type=selection_type)
        out["lambdas"] = out["selections"]
        outputs["bandit-" + selection_type] = out
    ans = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    # print(outputs["bandit-" + selection_type]["selections"])
    for key, values in outputs.items():
        ans[key]["empirical_risk"] = values["losses"].mean()
        ans[key]["num_items"] = values["num_items"].sum()
        ans[key]["num_good_items"] = (values["num_items"] - values["num_defects"]).sum()
        ans[key]["ratio_defects"] = values["num_defects"].sum() / values["num_items"].sum()
        ans[key]["avg_lambda"] = values["lambdas"].mean()

    ans = {key: dict(values) for key, values in ans.items()}
    return ans, outputs


one_run = run_deciders(502)

# print(out)

if __name__ == '__main__':
    # pool = dill.extend(Pool)()
    # pool = Pool()
    # pool.map(run_deciders, range(n_runs))
    # pool.close()

    all_runs = Parallel(n_jobs=8)(delayed(run_deciders)(i) for i in range(n_runs))
    metrics = defaultdict(lambda: defaultdict(lambda: np.zeros(n_runs)))
    for run_id, run in enumerate(all_runs):
        for key, values in run[0].items():
            for metric, value in values.items():
                metrics[key][metric][run_id] = value

    metrics = {key: dict(values) for key, values in metrics.items()}

    for key, values in metrics.items():
        for metric, value in values.items():
            print(key, metric, value.mean())

    import pickle as pkl

    with open("factory_metrics_constant_Cp.pkl", "wb") as f:
        pkl.dump(metrics, f)

    # print({"ours": output["losses"].mean(), "fix": output_fix["losses"].mean()})
    # print({"ours": (output["num_items"] - output["num_defects"]).sum(),
    #         "fix": (output_fix["num_items"] - output_fix["num_defects"]).sum()})
    # print({k: output_bandit[k]["losses"].mean() for k in selection_types})
    # print({k: (output_bandit[k]["num_items"] - output_bandit[k]["num_defects"]).sum() for k in selection_types})
    #
    # print()
