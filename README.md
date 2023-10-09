# Conformal Decision Making: 
This is the official repository of [Conformal Decision Making](http://arxiv.org/abs/2208.02814) by [Jordan Lekeufack*](https://jordylek.github.io/), [Anastasios N. Angelopoulos*](https://people.eecs.berkeley.edu/~angelopoulos/), [Andrea Bajcsy*](https://www.cs.cmu.edu/~abajcsy/), [Michael I. Jordan*](http://people.eecs.berkeley.edu/~jordan/), [Jitendra Malik*](http://people.eecs.berkeley.edu/~malik/)


<p align="center">
    <a style="text-decoration:none !important;" href="http://arxiv.org/abs/2208.02814" alt="arXiv"> <img src="https://img.shields.io/badge/paper-arXiv-red" /> </a>
    <a style="text-decoration:none !important;" href="https://conformal-decision.github.io/" alt="website"> <img src="https://img.shields.io/badge/website-Berkeley-yellow" /> </a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a>

</p>

## Technical background
*Conformal Decision Theory*(CDT) is a framework for producing safe autonomous decision despice imperfect machine learning predictions. Given a family of decision functions $\mathcal{D}_t = \{D^\lambda_t: \, \lambda \in \mathbb{R}\}$. At each timestep $t$, the agent receive an input $x_t$, and must make a decision function $D^\lambda_t$ that will output an action $u_t:= D_\lambda(x_t)$, then incurs a loss $\mathcal{L}(u_t, y_t)$. CDT provides a way to select $\lambda_t$, such that the empirical risk is controlled:
$$
\frac{1}{T} \sum_{t=1}^T \mathcal{L}(D_t^{\lambda_t}(x_t), y_t) \leq \varepsilon + O(1/T)
$$

## Setup
With Anaconda, create a new environment and the packages in `requirements.txt`

```
    conda create -n conformaldt python=3.9
    pip install -r requirements.txt
```


## Short Examples
`run_factory_example.py` runs the Factory example presented in the paper. `run_trading_example` runs the trading example.

## Stanford Drone Example
To run the SDD example, You first need to download the code from the [website](https://cvgl.stanford.edu/projects/uav_data/)

```
    wget http://vatic2.stanford.edu/stanford_campus_dataset.zip
    unzip stanford_campus_dataset.zip
```

You then need to create a cache for the predictions of humans next position
```
    python sdd/darts_cache.py
```

Then you can create the trajectory for the robot and generate the video

```
    python sdd/plan_trajectory.py
    python sdd/make_results.py
```

### Citation 

```
@article{lekeufack2024decision,
  author    = {Lekeufack, Jordan, and Angelopoulos, Anastasios N, and Bajcsy, Andrea, and Jordan, Michael I., and Malik, Jitendra},
  title     = {Conformal Decision Theory: Safe Autonomous Decisions Without Distributions},
  journal   = {arXiv},
  year      = {2024},
}
```