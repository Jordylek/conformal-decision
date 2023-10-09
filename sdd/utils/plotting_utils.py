import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import multivariate_normal
import seaborn as sns
import subprocess
import glob
import cv2
from multiprocessing import Pool
import pdb

def plot_gaussian_over_image(img, mean, std, rgb_color):
    """Plot a 2D Gaussian PDF over an image.

    Args:
    img_path (str): The path to the image file.
    mean (tuple): The mean of the Gaussian PDF.
    std (float): The standard deviation of the Gaussian PDF.
    rgb_color (tuple): The RGB color to use for the Gaussian PDF. Each component should be in the range [0, 1].
    """
    # Define the covariance matrix for the Gaussian PDF.
    cov = np.diag([std, std]) ** 2

    # Create a grid of points over which to evaluate the Gaussian PDF.
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Evaluate the Gaussian PDF at each point in the grid.
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)

    # Create an RGBA image for the Gaussian PDF.
    rgba = np.ones((img.shape[0], img.shape[1], 4))
    rgba[..., :3] = rgb_color[:3]
    rgba[..., 3] = Z / np.max(Z)
    rgba[..., 3][rgba[..., 3] < 0.01] = 0

    # Overlay the Gaussian PDF, using the specified RGBA image.
    plt.imshow(rgba, extent=[0, img.shape[1], 0, img.shape[0]])

def plot_trajectories(
    frame,
    df,
    cmap_lut,
    savename=None,
    aheads = [0,9,19,29],
    goals=False,
    show=True, # Only active when savename = None
):
    fig = plt.figure()
    fig.set_size_inches(1. * frame.shape[1] / frame.shape[0], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(frame)
    for meta in df.metaId.unique():
        curr_x = df[df.metaId == meta].x.iloc[0]
        curr_y = df[df.metaId == meta].y.iloc[0]
        if goals:
            goal_x = df[df.metaId == meta].goal_x.iloc[0]
            goal_y = df[df.metaId == meta].goal_y.iloc[0]
        pred_trajs_x = []
        pred_trajs_y = []
        gt_trajs_x = []
        gt_trajs_y = []

        for ahead in aheads:
            local_df = df[(df.metaId == meta) &  (df.ahead == ahead)]
            if len(local_df) == 0:
                continue
            pred_trajs_x += [local_df.pred_x.to_numpy()]
            pred_trajs_y += [local_df.pred_y.to_numpy()]
            try:
                gt_trajs_x += [local_df.future_x.to_numpy()[0]]
                gt_trajs_y += [local_df.future_y.to_numpy()[0]]
            except:
                pass
        pred_trajs_x = np.clip(np.stack(pred_trajs_x, axis=0), 0, frame.shape[1])
        pred_trajs_y = np.clip(np.stack(pred_trajs_y, axis=0), 0, frame.shape[0])
        gt_trajs_x = np.clip(np.array(gt_trajs_x), 0, frame.shape[1])
        gt_trajs_y = np.clip(np.array(gt_trajs_y), 0, frame.shape[0])
        #ax.plot(trajs_x, trajs_y, color=cmap_lut[meta], alpha=0.1, linewidth=0.5) # Full distributions!
        ax.plot(gt_trajs_x, gt_trajs_y, color='r', alpha=0.5, linewidth=0.2) # GT!
        ax.plot(pred_trajs_x.mean(axis=1), pred_trajs_y.mean(axis=1), color=cmap_lut[meta], alpha=0.5, linewidth=0.2) # Mean!
        #ax.plot(trajs_x[:,0], trajs_y[:,1], color=cmap_lut[meta], alpha=0.7, linewidth=2) # One sample!
        #for j in range(trajs_x.shape[0]):
        #    plot_gaussian_over_image(frame, [trajs_x[j].mean(), trajs_y[j].mean()], trajs_x[j].std(), cmap_lut[meta])
        if goals:
            ax.scatter(goal_x*4, goal_y*4, color=cmap_lut[meta], alpha=0.7, marker='X', s=1, linewidths=0.1)
        ax.scatter(curr_x, curr_y, facecolor="none", edgecolor=cmap_lut[meta], s=3, linewidths=0.1, alpha=0.7)

    if (savename is None) and show:
        plt.gcf().set_dpi(frame.shape[0])
        plt.show()
    else:
        plt.savefig(savename, dpi=frame.shape[0], bbox_inches='tight', pad_inches=0)
    return fig

def plot_trajectories_darts(
    frame,
    df,
    cmap_lut,
    savename=None,
    aheads = [0,9,19,29],
    goal=True,
    show=True, # Only active when savename = None
    params=None,
):
    downsample = 1 if (params is None) or ('downsample' not in params.keys()) else params['downsample']
    fig = plt.figure()
    fig.set_size_inches(1. * frame.shape[1] / frame.shape[0], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(frame)
    ax.autoscale(False)
    human_radius_pt = params['human_rad']* 72 / frame.shape[0] # In pt
    human_area_pt = np.pi * (human_radius_pt ** 2)
    # For all the humans, plot their paths differently
    for meta in df[df.metaId >= 0].metaId.unique():
        curr_x = df[df.metaId == meta].x.iloc[0]
        curr_y = df[df.metaId == meta].y.iloc[0]
        pred_trajs_x = np.clip(df[df.metaId == meta].pred_x, 0, frame.shape[0])
        pred_trajs_y = np.clip(df[df.metaId == meta].pred_y, 0, frame.shape[1])
        gt_trajs_x = np.clip(df[df.metaId == meta].future_x, 0, frame.shape[0])
        gt_trajs_y = np.clip(df[df.metaId == meta].future_y, 0, frame.shape[1])
        ax.plot(gt_trajs_x, gt_trajs_y, color='r', alpha=0.5, linewidth=0.2) # GT!
        ax.plot(pred_trajs_x, pred_trajs_y, color=cmap_lut[meta], alpha=0.5, linewidth=0.2) # Mean!
        ax.scatter(curr_x, curr_y, facecolor="none", edgecolor=cmap_lut[meta], s=human_area_pt, linewidths=0.1, alpha=0.7) # Draw circles with circle object? obs_patch = plt.Circle(circle_xy, circle_r, edgecolor='k', linestyle='--', fill=False)
    # Plot a robot
    curr_x = df[df.metaId == -1].x.iloc[0]
    curr_y = df[df.metaId == -1].y.iloc[0]
    pred_trajs_x = np.clip(df[df.metaId == -1].pred_x, 0, frame.shape[0])
    pred_trajs_y = np.clip(df[df.metaId == -1].pred_y, 0, frame.shape[1])
    r_goal_reached = df[df.metaId == -1].r_goal_reached.iloc[0] 
    if not r_goal_reached:
        ax.scatter(pred_trajs_y, pred_trajs_x, s=0.1, alpha=0.1, color='r')
    add_robot(ax, curr_x, curr_y, 2, 2, r_goal_reached)
    if goal:
        r_goal_count = int(df[df.metaId == -1]['r_goal_count'].iloc[0]) if not r_goal_reached else -1
        r_goal = params['r_goal'][r_goal_count]
        ax.scatter(r_goal[0], r_goal[1], s=3, marker='*', edgecolors='none', color='gold')
    if savename is None:
        if show:
            plt.gcf().set_dpi(frame.shape[0] / downsample)
            plt.show()
    else:
        plt.savefig(savename, dpi=frame.shape[0] / downsample, bbox_inches='tight', pad_inches=0)
    return fig

def add_robot(ax, r, c, r_h, r_w, r_goal_reached):
    # Read the image file
    robot_image = plt.imread('./assets/robot_happy.png') if r_goal_reached else plt.imread('./assets/robot.png')

    # Compute the zoom factor to control the size of the image
    zoom = min(r_h / robot_image.shape[0], r_w / robot_image.shape[1])

    # Create an OffsetImage object for the robot emoji
    imagebox = OffsetImage(robot_image, zoom=zoom)

    # Create an AnnotationBbox object with the emoji and the position where you want to place it
    ab = AnnotationBbox(imagebox, (c, r), frameon=False, boxcoords="data", pad=0)

    # Add the AnnotationBbox object to the axes
    ax.add_artist(ab)

def save_image_for_video(datum):
    frame, df, cmap_lut, savename, aheads, params = datum[0], datum[1], datum[2], datum[3], datum[4], datum[5]
    fig = plot_trajectories_darts(
        frame,
        df,
        cmap_lut,
        savename=savename,
        aheads=aheads,
        params=params
    )
    plt.close(fig)

def generate_video(frames, df, cmap_lut, folder, video_name, aheads, params=None, truncate_video=True, frame_indexes=None, step=1):
    os.makedirs(folder, exist_ok=True)
    if truncate_video:
        good_frames = df[df.r_goal_reached == False].frame.unique()
        if good_frames.max() < df.frame.max():
            good_frames = np.append(good_frames, int(df[df.r_goal_reached == True].frame.unique().min()))
        df = df[df.frame.isin(good_frames)]
    if frame_indexes is None:
        frame_indexes = np.arange(len(frames))
    starting_frame_idx = int(frame_indexes.min())
    args_list = [[frames[np.where(frame_indexes==frame_indexes[j])[0][0]], df[df.frame == frame_indexes[j]], cmap_lut, folder + f"{video_name}_%09d.png" % j, aheads, params] for j in range(frame_indexes.shape[0])]
    with Pool(8) as pool:
        pool.map(save_image_for_video, args_list)
    subprocess.call([
        'ffmpeg', "-y", "-framerate", "29.97",
        "-start_number", f"{0}",
        "-i", folder + f"/{video_name}_%09d.png",
        "-q:v", "0",
        "-pix_fmt", "yuv420p",
        folder + '/' + video_name + '.mp4'
    ])
    for file_name in glob.glob(folder + f"/{video_name}_*.png"):
        os.remove(file_name)
