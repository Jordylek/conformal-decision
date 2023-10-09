# Conformal Decision Making: 
This is the official repository of [Conformal Decision Making](https://conformal-decision.github.io/static/pdf/submission.pdf) by [Jordan Lekeufack*](https://jordylek.github.io/), [Anastasios N. Angelopoulos*](https://people.eecs.berkeley.edu/~angelopoulos/), [Andrea Bajcsy*](https://www.cs.cmu.edu/~abajcsy/), [Michael I. Jordan*](http://people.eecs.berkeley.edu/~jordan/), [Jitendra Malik*](http://people.eecs.berkeley.edu/~malik/)


<p align="center">
    <a style="text-decoration:none !important;" href="https://conformal-decision.github.io/static/pdf/submission.pdf" alt="arXiv"> <img src="https://img.shields.io/badge/paper-arXiv-red" /> </a>
    <a style="text-decoration:none !important;" href="https://conformal-decision.github.io/" alt="website"> <img src="https://img.shields.io/badge/website-CDT-yellow" /> </a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a>

</p>

## Technical background
*Conformal Decision Theory*(CDT) is a framework for producing safe autonomous decision despice imperfect machine learning predictions. Given a family of decision functions $\{D^\lambda_t\, \lambda \in \mathbb{R}\}$. At each timestep $t$, the agent receive an input $x_t$, and must make a decision function $D_t^{\lambda_t}$ that will output an action $u_t:= D_\lambda(x_t)$, then incurs a loss $\mathcal{L}(u_t, y_t)$. CDT provides a way to select $\lambda_t$, such that the empirical risk is controlled:
$$\frac{1}{T} \sum_{t=1}^T \mathcal{L}(D_t^{\lambda_t}(x_t), y_t) \leq \varepsilon + O(1/T)$$

## Setup
With Anaconda, create a new environment and the packages in `requirements.txt`

```
    conda create -n conformaldt python=3.9
    pip install -r requirements.txt
```


## Short Examples
`run_factory_example.py` runs the Factory example presented in the paper. `run_trading_example` runs the trading example.

## Stanford Drone Example
To run the SDD example, You first need to download the dataset from the [website](https://cvgl.stanford.edu/projects/uav_data/) and unzip it in your directory of choice.

```
    wget http://vatic2.stanford.edu/stanford_campus_dataset.zip
    unzip stanford_campus_dataset.zip
```

You also need to download [`ynet_additional_files`](https://drive.google.com/file/d/1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa/view?usp=sharing). 

Then you need to edit the default arguments to [`load_SDD`](https://github.com/Jordylek/conformal-decision/blob/d3f3e97157d7f1ce0957cbba910a699be3f16f8b/sdd/utils/preprocessing.py#L14) to point to these filepaths.

You then need to create a cache for the predictions of the humans' next positions
```
    bash sdd/bash-cache-darts.py
```

Then you can create the trajectory for the robot and generate the video

```
    bash bash-traj.sh
```

The videos will be stored in `sdd/videos` and the results in `sdd/metrics`.

### Citation 

```
@article{lekeufack2024decision,
  author    = {Lekeufack, Jordan, and Angelopoulos, Anastasios N, and Bajcsy, Andrea, and Jordan, Michael I., and Malik, Jitendra},
  title     = {Conformal Decision Theory: Safe Autonomous Decisions Without Distributions},
  journal   = {arXiv},
  year      = {2024},
}
```
