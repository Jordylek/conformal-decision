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
from utils.preprocessing import load_SDD
from utils.spline_planner import SplinePlanner
from utils import spline_planner_utils
from tqdm import tqdm
import yaml

def collapse_samples(df):
    """
    Collapses the samples in the dataframe df to calculate the mean and standard deviation
    of the 'pred_x', 'pred_y', 'goal_x', and 'goal_y' columns, grouped by all columns other
    than these and 'sample'.

    Parameters:
    df (pd.DataFrame): The input dataframe

    Returns:
    pd.DataFrame: The output dataframe with the collapsed samples
    """
    # Handle potential renaming and dropping of columns
    if 'x_y' in df.columns and 'y_y' in df.columns:
        df.drop(['x_y', 'y_y'], axis=1, inplace=True)
    if 'x_x' in df.columns and 'y_x' in df.columns:
        df.rename({'x_x': 'x', 'y_x' : 'y'}, axis=1, inplace=True)

    # Drop unwanted columns
    df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True, errors='ignore')

    # Identify columns other than 'pred_x', 'pred_y', 'goal_x', 'goal_y' and 'sample' to group by
    groupby_cols = [col for col in df.columns if col not in ('pred_x', 'pred_y', 'goal_x', 'goal_y', 'sample')]

    # Group by the identified columns and calculate mean and std dev of 'pred_x', 'pred_y', 'goal_x', and 'goal_y'
    print("df before: ", len(df))
    df = df.groupby(groupby_cols, as_index=False).agg(
        pred_x=('pred_x', 'mean'),
        pred_x_std=('pred_x', 'std'),
        pred_y=('pred_y', 'mean'),
        pred_y_std=('pred_y', 'std'),
        goal_x=('goal_x', 'mean'),
        goal_y=('goal_y', 'mean'),
    )
    print("df after: ", len(df))

    return df

def plan_trajectory(df, params):
    planner = SplinePlanner(
        params['num_waypts'],
        params['horizon'],
        params['max_linear_vel'],
        params['max_angular_vel'],
        params['gmin'],
        params['gmax'],
        params['gnums'],
        params['goal_rad'],
        params['human_rad'],
        params['lr'],
        params['alpha'],
        params['binary_env_img_path']
    )
    r_start = params['r_start']
    j = 0
    r_goal = params['r_goal'][j]
    # If there's only one goal, and you're in it, no need to plan
    r_goal_reached = (np.linalg.norm(r_start[:2] - params['r_goal'][-1][:2]) <= params['goal_rad']) & (len(params['r_goal']) == 1)

    plan_df = []
    worst_residuals = []
    print("[plan_trajectory] Planning...")
    for frame_idx in tqdm(sorted(df[df.frame >= params['starting_frame']].frame.unique())):
        # Check if goal has been reached
        if np.linalg.norm(r_start[:2] - r_goal[:2]) <= params['goal_rad'] and not r_goal_reached:
            j += 1
        if j == len(params['r_goal']): # If we've exceeded the number of goals
            r_goal_reached = True
        if not r_goal_reached:
            r_goal = params['r_goal'][j]
            # Prepare data
            curr_df = df[df.frame == frame_idx]
            human_spline_preds = np.stack([
                np.stack([
                curr_df[curr_df.metaId == mid].pred_x.to_numpy(),
                curr_df[curr_df.metaId == mid].pred_y.to_numpy()
                ], axis=0)
            for mid in curr_df.metaId.unique()], axis=0)

            # Plan
            agent_df = curr_df[curr_df.metaId >= 0]
            all_residuals_timestep = np.sqrt((agent_df.pred_x - agent_df.future_x)**2 + (agent_df.pred_y - agent_df.future_y)**2)
            worst_residuals += [ all_residuals_timestep.max() ]
            robot_spline_plan = planner.plan( r_start, r_goal, human_spline_preds, params['method'], worst_residuals=np.array(worst_residuals))

            plan_df += [
                pd.DataFrame({
                    'frame': frame_idx,
                    'x': (params['horizon']-1)*[r_start[1]], # x
                    'y': (params['horizon']-1)*[r_start[0]], # y
                    'aheads': np.arange(params['horizon'])[:-1] + 1,
                    'pred_x': robot_spline_plan[1][1:],
                    'pred_y': robot_spline_plan[0][1:],
                    'r_goal_reached': False,
                    'r_goal_count': j,
                    'lambda' : planner.lamda,
                    'alphat' : planner.alphat,
                    'metaId': -1
                })
            ]
            # Take next step
            r_start = [ robot_spline_plan[0][1], robot_spline_plan[1][1], robot_spline_plan[4][1], robot_spline_plan[2][1] ]
        else:
            plan_df += [
                pd.DataFrame({
                    'frame': frame_idx,
                    'x': (params['horizon']-1)*[r_start[1]], # x
                    'y': (params['horizon']-1)*[r_start[0]], # y
                    'aheads': np.arange(params['horizon'])[:-1] + 1,
                    'pred_x': (params['horizon']-1)*[r_start[1]],
                    'pred_y': (params['horizon']-1)*[r_start[0]],
                    'r_goal_reached': True,
                    'r_goal_count': j,
                    'lambda' : None,
                    'alphat' : None,
                    'metaId': -1
                })
            ]

    plan_df = pd.concat([df] + plan_df, axis=0, ignore_index=True)
    return plan_df

def convert_state(state, H, W):
    return np.array([ float(state[0]) * W, float(state[1]) * H, float(state[2]), float(state[3]) ])

def convert_units(params, H, W):
    params['H_m'] = int(params['H_m'])
    params['W_m'] = int(params['W_m'])
    px_per_m = 0.5*H/params['H_m'] + 0.5*W/params['W_m'] # Average for robustness
    aheads=list(range(int(params['horizon'])))

    # Unit conversions
    params['gmin'] = [params['gmin'][0]*W, params['gmin'][1]*H]
    params['gmax'] = [params['gmax'][0]*W, params['gmax'][1]*H]
    params['aheads'] = aheads
    #params['sd_obs'] = np.ones(params['gnums']) * 10000
    params['binary_env_img_path'] = 'nexus_4_frame_bin.png'
    params['max_linear_vel'] *= px_per_m / float(params['fps'])
    params['goal_rad'] *= px_per_m
    params['human_rad'] *= px_per_m
    params['r_start'] = convert_state(params['r_start'], H, W)
    params['r_goal'] = [ convert_state(g, H, W) for g in params['r_goal'] ]
    params['px_per_m'] = px_per_m
    return params

def setup_params(params, H, W, lr, method):
    params['r_goal'] = [np.array(list(g.values())).squeeze() for g in params['r_goal']]
    params = convert_units(params, H, W)
    params['lr'] = lr
    params['method'] = method
    return params

if __name__ == "__main__":
    # Setup
    scene = sys.argv[1]
    forecaster = sys.argv[2]
    method = sys.argv[3]
    lr = float(sys.argv[4])
    json_name = './params/' + scene + '.json'
    params = yaml.safe_load(open(json_name))

    mode = 'train' if scene in os.listdir('./data/train') else 'test'
    # By default, do not overwrite, or if True, do
    overwrite = False if len(sys.argv) < 6 else sys.argv[5] == 'True'
    cmap = plt.get_cmap('terrain')
    frames, _, reference_image = load_SDD(scene, load_frames=[0], mode=mode)
    H, W, _ = frames[0].shape
    params = setup_params(params, H, W, lr, method=method)

    str_append = forecaster + '_' + method.replace(' ', '_') + "_" + str(lr).replace('.', '_')
    plan_folder = './plans/' + scene + '/'
    plan_filename = os.path.join(plan_folder, 'plan_df_' + str_append + '.csv')

    if overwrite or not os.path.exists(plan_filename):
        if overwrite:
            print(f"[plan_trajectory] Planning {str_append} from scratch, overwriting previous save if any.")
        elif not os.path.exists(plan_filename):
            print(f"[plan_trajectory] cannot find {str_append}, planning from scratch.")
        # Setup parameters
        df = pd.read_csv('./data/' + forecaster + '/' + scene + '.csv')
        if forecaster == 'ynet':
            df = collapse_samples(df)

        # Create future trajectories
        fake_df = df.copy()
        fake_df['frame'] = df['frame'] - df['ahead']
        fake_df.rename(columns={'x' : 'future_x', 'y' : 'future_y'}, inplace=True)
        if forecaster == 'darts':
            fake_df.drop(['Unnamed: 0', 'trackId', 'pred_x', 'pred_y'], axis=1, inplace=True)
        elif forecaster == 'ynet':
            fake_df.drop(['trackId', 'sceneId', 'pred_x', 'pred_y', 'pred_x_std', 'pred_y_std', 'goal_x', 'goal_y'], axis=1, inplace=True)
        fake_df = fake_df.loc[fake_df['frame'] >= 0]
        future_df = df.merge(fake_df, how='left', on=['frame', 'metaId', 'ahead'])

        # Plan trajectory
        plan_df = plan_trajectory(future_df, params)
        # Save the plans, along with the forecaster and learning rate
        os.makedirs(plan_folder, exist_ok=True)
        plan_df.to_csv(plan_filename)
    else:
        print(f"Already cached {str_append}")
