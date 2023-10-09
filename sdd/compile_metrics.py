import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import subprocess
import glob
import pdb
from multiprocessing import Pool
from tqdm import tqdm
from utils.preprocessing import load_SDD
from utils.plotting_utils import generate_video
from utils.metrics import make_metrics
from plan_trajectory import setup_params
import yaml

if __name__ == "__main__":
    # Setup
    scene = sys.argv[1]
    forecaster = sys.argv[2]
    sigfigs=4

    metrics_folder = './metrics/' + scene + '/'
    metrics_files = os.listdir(metrics_folder)
    metrics_dfs = [ pd.read_csv(os.path.join(metrics_folder, f)) for f in metrics_files if '.csv' in f ]
    metrics_df = pd.concat(metrics_dfs, ignore_index=True)
    metrics_df = metrics_df[metrics_df['forecaster'] == forecaster].drop(columns=['Unnamed: 0', 'forecaster'], axis=1)
    # Sort the metrics_df by first method, then lr
    metrics_df.loc[(metrics_df['method'] == 'conformal controller') & (metrics_df['lr'] == 0), 'method'] = 'Aggressive'
    metrics_df = metrics_df.sort_values(by=['method', 'lr'], ascending=[True, True])
    metrics_df.set_index(['method', 'lr'], inplace=True)

    # Find the columns with the character % and rename them to have a \% instead
    # (otherwise the LaTeX table will not compile)
    for col in metrics_df.columns:
        if '%' in col:
            metrics_df.rename(columns={col: col.replace('%', r'\%')}, inplace=True)

    # Function to round to two significant figures
    def round_to_sf(x, sf):
        if x == 0:
            return 0
        else:
            return round(x, sf - 1 - int(np.floor(np.log10(abs(x)))))

    # Function to apply the coloring and rounding
    def rounding(col):
        def format_value(x):
            if np.isinf(x):
                return '$\infty$'
            elif np.isnan(x):
                return 'NaN'
            else:
                x = round_to_sf(x, sigfigs)
                if np.isreal(x) and np.isclose(x, round(x)):
                    return str(int(round(x)))
                else:
                    return format(x, f".{sigfigs}g")
        return [format_value(x) if not isinstance(x, (bool, str)) else x for x in col]

    # Apply the function to each column
    styled_df = metrics_df.apply(rounding)

    # Create a Styler object from the styled DataFrame
    styler = styled_df.style

    # Convert the Styler to a LaTeX table and save it to a file
    latex_table = styler.to_latex()

    # Remove unnecessary zeros after the decimal point
    latex_table = re.sub(r'(\.\d*?)0+\s', r'\1 ', latex_table)

# If the above regex leaves a dot at the end of a number, remove the dot as well
    latex_table = re.sub(r'\.\s', ' ', latex_table)
    latex_table = re.sub(r'True', r'\\cmark', latex_table)
    latex_table = re.sub(r'False', r'\\xmark', latex_table)
    latex_table = re.sub(r'aci', r'\\makecell{ACI \\\\($\\alpha=0.01$)}', latex_table)
    latex_table = re.sub(r'conformal controller', r'\\makecell{Conformal \\\\Controller \\\\($\\epsilon=2$m)}', latex_table)
    latex_table = re.sub(r'conservative', r'Conservative', latex_table)

    # Save the LaTeX table to a file
    with open(os.path.join(metrics_folder, scene + '_' + forecaster + '.tex'), 'w') as f:
        f.write(latex_table)
