import os, sys
import numpy as np
import pandas as pd
import pdb

def get_distances(df, px_per_m=None):
    unique_frames = df[df.r_goal_reached == False].frame.unique()
    distances = []
    for frame in sorted(unique_frames):
        curr_df = df[df.frame == frame]
        distances += [ # TODO: Fix this issue elsewhere too.
            np.sqrt((curr_df[curr_df.metaId == -1].y.iloc[0] - curr_df[curr_df.metaId != -1].x.to_numpy())**2 +
            (curr_df[curr_df.metaId == -1].x.iloc[0] - curr_df[curr_df.metaId != -1].y.to_numpy())**2).min()
        ]
    distances = np.array(distances) / px_per_m if px_per_m is not None else np.array(distances)
    return distances

def make_metrics(df, metrics_folder, metrics_filename, forecaster, method, lr, params, fps=30):
    unique_frames = df[df.r_goal_reached == False].frame.unique()
    # Calculate distances of closest human to robot in each frame
    distances = get_distances(df)
    collision_distance = ((params['robot_size'] + params['human_size'])*params['px_per_m'])
    did_collide = distances.min() < collision_distance
    did_succeed = df.r_goal_count.max() > 0
    metrics = {
        'forecaster' : forecaster,
        'method' : method,
        'lr' : lr,
        'success' : str(did_succeed),
        'goal time (s)' : (unique_frames.max() - unique_frames.min())/fps/df.r_goal_count.max() if did_succeed else np.infty,
        'safe' : str(~did_collide),
        'min dist (m)' : distances.min()/params['px_per_m'],
        'avg dist (m)' : distances.mean()/params['px_per_m'],
        '5% dist (m)'  : np.quantile(distances, 0.05)/params['px_per_m'],
        '10% dist (m)' : np.quantile(distances, 0.1)/params['px_per_m'],
        '25% dist (m)' : np.quantile(distances, 0.25)/params['px_per_m'],
        '50% dist (m)' : np.quantile(distances, 0.5)/params['px_per_m']
    }
    metrics = pd.DataFrame(metrics, index=[0])
    # Save out
    os.makedirs(metrics_folder, exist_ok=True)
    metrics.to_csv(metrics_filename)
