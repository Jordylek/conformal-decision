import os, sys
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
    method = sys.argv[3]
    lr = float(sys.argv[4])
    json_name = './params/' + scene + '.json'
    step = 1
    params = yaml.safe_load(open(json_name))
    mode = 'train' if scene in os.listdir('./data/ynet_additional_files/data/SDD/train') else 'test'
    # By default, do not overwrite, or if True, do
    overwrite = False if len(sys.argv) < 6 else sys.argv[5] == 'True'
    cmap = plt.get_cmap('terrain')
    frames, _, _ = load_SDD(scene, mode=mode, load_frames=[0])
    H, W, _  = frames[0].shape
    params = setup_params(params, H, W, lr, method)

    df = pd.read_csv('./data/' + forecaster + '/' + scene + '.csv')
    str_append = forecaster + '_' + method.replace(' ', '_') + "_" + str(lr).replace('.', '_')

    plan_folder = './plans/' + scene + '/'
    plan_filename = os.path.join(plan_folder, 'plan_df_' + str_append + '.csv')

    metrics_folder = './metrics/' + scene + '/'
    metrics_filename = os.path.join(metrics_folder, 'metrics_df_' + str_append + '.csv')

    videos_folder = './videos/' + scene + '/'
    video_filename = str_append

    try:
        plan_df = pd.read_csv(plan_filename)
    except:
        raise ValueError("No plan found for " + str(scene) + str(lr) + str(method))

    frames_to_read = plan_df[plan_df.r_goal_reached == False].frame.unique()
    frames_to_read = frames_to_read[::step]
    if (frames_to_read.max() < plan_df.frame.max()) & ((plan_df.r_goal_reached == True).sum() > 0):
        frames_to_read = np.append(frames_to_read, int(plan_df[plan_df.r_goal_reached == True].frame.unique().min()))
    # Generate video
    cmap_lut = cmap(np.random.permutation(np.linspace(0, 1, plan_df.metaId.max()+1)))
    frames, _, _ = load_SDD(scene, mode=mode, load_frames=frames_to_read)
    generate_video(frames, plan_df, cmap_lut, videos_folder, video_filename, aheads=params['aheads'], params=params, truncate_video=True, frame_indexes=frames_to_read)

    # Generate metrics
    make_metrics(plan_df, metrics_folder, metrics_filename, forecaster, method, float(lr), params)
