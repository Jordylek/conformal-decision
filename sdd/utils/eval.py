import torch
import torch.nn as nn
from utils.image_utils import get_patch, sampling, image2world, create_dist_mat
from utils.kmeans import kmeans
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
import os

def torch_multivariate_gaussian_heatmap(coordinates, H, W, dist, sigma_factor, ratio, device, rot=False):
    """
    Create Gaussian Kernel for CWS
    """
    ax = torch.linspace(0, H, H, device=device) - coordinates[1]
    ay = torch.linspace(0, W, W, device=device) - coordinates[0]
    xx, yy = torch.meshgrid([ax, ay])
    meshgrid = torch.stack([yy, xx], dim=-1)
    radians = torch.atan2(dist[0], dist[1])

    c, s = torch.cos(radians), torch.sin(radians)
    R = torch.Tensor([[c, s], [-s, c]]).to(device)
    if rot:
        R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
    dist_norm = dist.square().sum(-1).sqrt() + 5  # some small padding to avoid division by zero

    conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]).to(device)
    conv = torch.square(conv)
    T = torch.matmul(R, conv)
    T = torch.matmul(T, R.T)

    kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
    kernel = torch.exp(-0.5 * kernel)
    return kernel / kernel.sum()

def normalize_image(img):
    norm_img = img - img.min()
    return norm_img / norm_img.max()

def get_future_samples(model, device, input_template, observed, last_observed, scene_image_seg, waypoints, temperature, num_goals, num_traj, obs_len, rel_thresh, CWS_params):
    _, _, H, W = scene_image_seg.shape
    observed_map = get_patch(input_template, observed, H, W)
    observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

    semantic_image = scene_image_seg.expand(observed_map.shape[0], -1, -1, -1)

    # Forward pass
    # Calculate features
    feature_input = torch.cat([semantic_image, observed_map], dim=1)
    features = model.pred_features(feature_input)

    # Predict goal and waypoint probability distributions
    pred_waypoint_map = model.pred_goal(features)
    pred_waypoint_map = pred_waypoint_map[:, waypoints]

    pred_waypoint_map_sigmoid = pred_waypoint_map / temperature
    pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)

    ################################################ TTST ##################################################
    goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
    goal_samples = goal_samples.permute(2, 0, 1, 3)

    num_clusters = num_goals - 1
    goal_samples_softargmax = model.softargmax(pred_waypoint_map[:, -1:])  # first sample is softargmax sample

    # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
    goal_samples_list = []
    for person in range(goal_samples.shape[1]):
        goal_sample = goal_samples[:, person, 0]

        # Actual k-means clustering, Outputs:
        # cluster_ids_x -  Information to which cluster_idx each point belongs to
        # cluster_centers - list of centroids, which are our new goal samples
        cluster_ids_x, cluster_centers = kmeans(X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
        goal_samples_list.append(cluster_centers)

    goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
    goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
    goal_samples_output = goal_samples

    #goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=num_goals)
    #goal_samples = goal_samples.permute(2, 0, 1, 3)

    # Predict waypoints:
    # in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
    if len(waypoints) == 1:
        waypoint_samples = goal_samples
    else:
        ################################################ CWS ###################################################
        # CWS Begin
        sigma_factor = CWS_params['sigma_factor']
        ratio = CWS_params['ratio']
        rot = CWS_params['rot']

        goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
        waypoint_samples_list = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
        for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
            waypoint_list = []  # for each K sample have a separate list
            waypoint_list.append(waypoint_samples)

            for waypoint_num in reversed(range(len(waypoints)-1)):
                distance = last_observed.to(device) - waypoint_samples
                gaussian_heatmaps = []
                traj_idx = g_num // num_goals  # idx of trajectory for the same goal
                for dist, coordinate in zip(distance, waypoint_samples):  # for each person
                    length_ratio = 1 / (waypoint_num + 2)
                    gauss_mean = coordinate + (dist * length_ratio)  # Get the intermediate point's location using CV model
                    sigma_factor_ = sigma_factor - traj_idx
                    gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
                gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]

                waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
                waypoint_map = waypoint_map_before * gaussian_heatmaps
                # normalize waypoint map
                waypoint_map = (waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)

                # For first traj samples use softargmax
                if g_num // num_goals == 0:
                    # Softargmax
                    waypoint_samples = model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
                    waypoint_samples = waypoint_samples.squeeze(0)
                else:
                    waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
                    waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                    waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
                waypoint_list.append(waypoint_samples)
            waypoint_list = waypoint_list[::-1]
            waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
            waypoint_samples_list.append(waypoint_list)
        waypoint_samples = torch.stack(waypoint_samples_list)

    # Interpolate trajectories given goal and waypoints
    future_samples = []
    for waypoint in waypoint_samples:
        waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)
        waypoint_map = torch.stack(waypoint_map).reshape([-1, len(waypoints), H, W])

        waypoint_maps_downsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]
        waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled

        traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypoint_maps_downsampled)]

        pred_traj_map = model.pred_traj(traj_input)
        pred_traj = model.softargmax(pred_traj_map)
        future_samples.append(pred_traj)
    future_samples = torch.stack(future_samples)
    return future_samples, waypoint_samples, goal_samples_output


def evaluate(model, val_loader, val_images, num_goals, num_traj, obs_len, batch_size, device, waypoints, resize, temperature, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None, dataset_name=None, homo_mat=None, mode='val'):
    """

    :param model: torch model
    :param val_loader: torch dataloader
    :param val_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
    :param num_goals: int, number of goals
    :param num_traj: int, number of trajectories per goal
    :param obs_len: int, observed timesteps
    :param batch_size: int, batch_size
    :param device: torch device
    :param input_template: torch.Tensor, heatmap template
    :param waypoints: number of waypoints
    :param resize: resize factor
    :param temperature: float, temperature to control peakiness of heatmap
    :param use_TTST: bool
    :param use_CWS: bool
    :param rel_thresh: float
    :param CWS_params: dict
    :param dataset_name: ['sdd','ind','eth']
    :param params: dict with hyperparameters
    :param homo_mat: dict with homography matrix
    :param mode: ['val', 'test']
    :return: val_ADE, val_FDE for one epoch
    """
    model.eval()
    val_ADE = []
    val_FDE = []
    counter = 0
    plot_future_samples = True
    # Create template
    input_template = torch.Tensor(create_dist_mat(size=int(4200 * resize))).to(device)
    
    with torch.no_grad():
        # outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
        for trajectory, meta, scene in val_loader:
            # Get scene image and apply semantic segmentation
            scene_image = val_images[scene].to(device).unsqueeze(0)
            scene_image = model.segmentation(scene_image)

            for i in range(0, len(trajectory), batch_size):
                # Create Heatmaps for past and ground-truth future trajectories
                _, _, H, W = scene_image.shape
                observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
                last_observed = trajectory[i:i+batch_size, obs_len-1, :]
                gt_future = trajectory[i:i+batch_size, obs_len:].to(device)

                future_samples, waypoint_samples, _ = get_future_samples(model, device, input_template, observed, last_observed, scene_image, waypoints, temperature, num_goals, num_traj, obs_len, rel_thresh, CWS_params)
                
                if plot_future_samples:
                    fig = plt.figure(figsize=(10,10))
                    plt.imshow(normalize_image(val_images[scene].permute(1,2,0)))
                    plt.gca().axis('off')
                    for id_idx in range(future_samples.shape[1]):
                        for sample in range(future_samples.shape[0]):
                            plt.plot(future_samples[sample,id_idx,:,0].cpu(), future_samples[sample,id_idx,:,1].cpu(), color='r', alpha=0.2)
                        plt.plot(trajectory[id_idx, obs_len:, 0], trajectory[id_idx, obs_len:, 1], color='g')
                        plt.plot(trajectory[id_idx, :obs_len, 0], trajectory[id_idx, :obs_len, 1], color='b')
                    plt.show()
                

                gt_goal = gt_future[:, -1:]

                val_FDE.append(((((gt_goal - waypoint_samples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
                val_ADE.append(((((gt_future - future_samples) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])

        val_ADE = torch.cat(val_ADE).mean()
        val_FDE = torch.cat(val_FDE).mean()

    return val_ADE.item(), val_FDE.item()

def evaluate_with_vis(model, val_loader, val_images, num_goals, num_traj, obs_len, batch_size, device, waypoints, resize, temperature, use_TTST=False, use_CWS=False, rel_thresh=0.002, CWS_params=None, dataset_name=None, homo_mat=None, mode='val', m2former_seg=False, seg_pkls_val=None, img_dir = '/home/aa/Code/conformal-decision/sdd/neerja-test', frame_path=None):
    """

    :param model: torch model
    :param val_loader: torch dataloader
    :param val_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
    :param num_goals: int, number of goals
    :param num_traj: int, number of trajectories per goal
    :param obs_len: int, observed timesteps
    :param batch_size: int, batch_size
    :param device: torch device
    :param input_template: torch.Tensor, heatmap template
    :param waypoints: number of waypoints
    :param resize: resize factor
    :param temperature: float, temperature to control peakiness of heatmap
    :param use_TTST: bool
    :param use_CWS: bool
    :param rel_thresh: float
    :param CWS_params: dict
    :param dataset_name: ['sdd','ind','eth']
    :param params: dict with hyperparameters
    :param homo_mat: dict with homography matrix
    :param mode: ['val', 'test']
    :return: val_ADE, val_FDE for one epoch
    """

    import matplotlib.pyplot as plt 
    import numpy as np
    from skimage import color
    import os
    # import cv2

    print("evaluating with visualization")
    os.makedirs(img_dir, exist_ok=True)

    model.eval()
    val_ADE = []
    val_FDE = []
    counter = 0

    vis = True
    input_template = torch.Tensor(create_dist_mat(size=int(4200 * resize))).to(device)

    with torch.no_grad():
        # outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
        for trajectory, meta, scene, meta_id in tqdm(val_loader):
            meta_ids = meta_id[0]
            meta = meta[0]

            os.makedirs(os.path.join(img_dir, scene), exist_ok=True)

            if m2former_seg:
                scene_image = torch.Tensor(seg_pkls_val[scene]).to(device).unsqueeze(0)
            else:
                scene_image = val_images[scene].to(device).unsqueeze(0)
                img_np = np.squeeze(np.asarray(scene_image.cpu().detach())).transpose(1,2,0)
                img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
                plt.imshow(img_np)
                plt.show()
                scene_image = model.segmentation(scene_image)

                seg_np = np.squeeze(np.asarray(scene_image.cpu().detach())).transpose(1,2,0)
                seg_np_max = np.argmax(seg_np, axis=2)
                color_img = color.label2rgb(seg_np_max, img_np)
                plt.imshow(color_img)
                plt.show()

            # # Get scene image and apply semantic segmentation
            # scene_image = val_images[scene].to(device).unsqueeze(0)
            # img_np = np.squeeze(np.asarray(scene_image.cpu().detach())).transpose(1,2,0)
            # img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
            # # print("image shape")
            # # print(img_np.shape)
            # # print(trajectory, meta, scene)


            # plt.imshow(img_np)
            # # cv2.imshow('image', img_np)
            # # cv2.waitKey(0)
            
            # scene_image = model.segmentation(scene_image)
            

            if dataset_name == 'eth':
                print(counter)
                counter += batch_size
                # Break after certain number of batches to approximate evaluation, else one epoch takes really long
                if counter > 30 and mode == 'val':
                    break

            for i in range(0, len(trajectory), batch_size):
                print("i, trajectory, batch")
                print(i)
                # print(trajectory[i:i+batch_size, :, :])
                meta_ids_batch = meta_ids[i:i+batch_size]

                # Create Heatmaps for past and ground-truth future trajectories
                _, _, H, W = scene_image.shape
                observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
                observed_map = get_patch(input_template, observed, H, W)
                observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

                gt_future = trajectory[i:i+batch_size, obs_len:].to(device)
                semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)

                # visualize observed trajectory over image 
                # print(trajectory.shape)
                traj_batch = trajectory[i:i+batch_size, :, :]
                # print(traj_batch.shape)
                observed_batch = trajectory[i:i+batch_size, :obs_len, :].cpu().numpy()
                # print(observed_batch.shape)
                future_gt_batch = trajectory[i:i+batch_size, obs_len:, :].cpu().numpy()
                # print(future_gt_batch.shape)

                # print(np.amax(observed_batch))
                # print(np.amin(observed_batch))
                # print("img shape")
                # print(img_np.shape)
                # print(resize)

                if vis: 
                    # print("\n visualizing \n")
                    # print(resize)
                    meta_id_0 = meta_ids_batch[0]
                    # print("meta id: ")
                    # print(meta_id_0)

                    
                    # print(traj_batch[0, :, :])
                    # print(meta)
                    meta_id_rows = meta.loc[meta['metaId'] == meta_id_0]
                    # print(meta_id_rows)
                    frames = meta_id_rows['frame'].to_numpy()
                    # print(frames)
                    mid_frame = frames[obs_len]
                    # print(mid_frame)

                    
                    image_file = f'{mid_frame:04}.jpg'
                    frame_img = cv2.imread(os.path.join(frame_path, scene, scene, 'rgb', image_file), 1)
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

                    plt.figure()
                    plt.imshow(frame_img)
                    plt.scatter(observed_batch[0,:,0]/resize, observed_batch[0,:,1]/resize, s=25, marker=".", c='b', alpha=0.5)
                    plt.scatter(future_gt_batch[0,:,0]/resize, future_gt_batch[0,:,1]/resize, s=25, marker=".", c='g', alpha=0.5)
                    plt.show()
                # print(observed)

                # Forward pass
                # Calculate features
                feature_input = torch.cat([semantic_image, observed_map], dim=1)
                features = model.pred_features(feature_input)

                # Predict goal and waypoint probability distributions
                pred_waypoint_map = model.pred_goal(features)
                pred_waypoint_map = pred_waypoint_map[:, waypoints]

                # pred_waypoint_np = np.squeeze(np.asarray(pred_waypoint_map.cpu().detach())).transpose(1,2,0)
                # for c in range(pred_waypoint_np.shape[2]):
                #     waypoint_t = pred_waypoint_np[:, :, c]
                #     plt.imshow(waypoint_t)
                #     plt.show()

                pred_waypoint_map_sigmoid = pred_waypoint_map / temperature
                pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)

                ################################################ TTST ##################################################
                if use_TTST:
                    # TTST Begin
                    # sample a large amount of goals to be clustered
                    goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
                    goal_samples = goal_samples.permute(2, 0, 1, 3)

                    num_clusters = num_goals - 1
                    goal_samples_softargmax = model.softargmax(pred_waypoint_map[:, -1:])  # first sample is softargmax sample

                    # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
                    goal_samples_list = []
                    for person in range(goal_samples.shape[1]):
                        goal_sample = goal_samples[:, person, 0]

                        # Actual k-means clustering, Outputs:
                        # cluster_ids_x -  Information to which cluster_idx each point belongs to
                        # cluster_centers - list of centroids, which are our new goal samples
                        cluster_ids_x, cluster_centers = kmeans(X=goal_sample, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=False, tol=0.001, iter_limit=1000)
                        goal_samples_list.append(cluster_centers)

                    goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
                    goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
                    # TTST End

                # Not using TTST
                else:
                    goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], num_samples=num_goals)
                    goal_samples = goal_samples.permute(2, 0, 1, 3)

                print(goal_samples.shape)

                if vis: 


                    goals_batch_0 = np.squeeze(goal_samples.cpu().numpy()[:, 0, :, :])
                    print(goals_batch_0.shape)

                    plt.scatter(goals_batch_0[:,0]/resize, goals_batch_0[:,1]/resize, s=25, marker="v", c='r', alpha=0.5)
                    plt.show()


                # Predict waypoints:
                # in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
                if len(waypoints) == 1:
                    waypoint_samples = goal_samples

                ################################################ CWS ###################################################
                # CWS Begin
                if use_CWS and len(waypoints) > 1:
                    sigma_factor = CWS_params['sigma_factor']
                    ratio = CWS_params['ratio']
                    rot = CWS_params['rot']

                    goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
                    last_observed = trajectory[i:i+batch_size, obs_len-1].to(device)  # [N, 2]
                    waypoint_samples_list = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
                    for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
                        waypoint_list = []  # for each K sample have a separate list
                        waypoint_list.append(waypoint_samples)

                        for waypoint_num in reversed(range(len(waypoints)-1)):
                            distance = last_observed - waypoint_samples
                            gaussian_heatmaps = []
                            traj_idx = g_num // num_goals  # idx of trajectory for the same goal
                            for dist, coordinate in zip(distance, waypoint_samples):  # for each person
                                length_ratio = 1 / (waypoint_num + 2)
                                gauss_mean = coordinate + (dist * length_ratio)  # Get the intermediate point's location using CV model
                                sigma_factor_ = sigma_factor - traj_idx
                                gaussian_heatmaps.append(torch_multivariate_gaussian_heatmap(gauss_mean, H, W, dist, sigma_factor_, ratio, device, rot))
                            gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]

                            waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
                            waypoint_map = waypoint_map_before * gaussian_heatmaps
                            # normalize waypoint map
                            waypoint_map = (waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)).view_as(waypoint_map)

                            # For first traj samples use softargmax
                            if g_num // num_goals == 0:
                                # Softargmax
                                waypoint_samples = model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
                                waypoint_samples = waypoint_samples.squeeze(0)
                            else:
                                waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
                                waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                                waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
                            waypoint_list.append(waypoint_samples)

                        waypoint_list = waypoint_list[::-1]
                        waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
                        waypoint_samples_list.append(waypoint_list)
                    waypoint_samples = torch.stack(waypoint_samples_list)

                    # CWS End

                # If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
                elif not use_CWS and len(waypoints) > 1:
                    waypoint_samples = sampling(pred_waypoint_map_sigmoid[:, :-1], num_samples=num_goals * num_traj)
                    waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                    goal_samples = goal_samples.repeat(num_traj, 1, 1, 1)  # repeat K_a times
                    waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)

                # Interpolate trajectories given goal and waypoints
                future_samples = []
                for waypoint in waypoint_samples:
                    waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W)
                    waypoint_map = torch.stack(waypoint_map).reshape([-1, len(waypoints), H, W])

                    waypoint_maps_downsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))]
                    waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled

                    traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypoint_maps_downsampled)]

                    pred_traj_map = model.pred_traj(traj_input)
                    pred_traj = model.softargmax(pred_traj_map)
                    future_samples.append(pred_traj)
                future_samples = torch.stack(future_samples)

                print("future samples")
                print(future_samples.shape)

                if vis: 
                    # import ipdb
                    # ipdb.set_trace()
                
                    future_samples_batch = np.squeeze(future_samples[:, 0, :, :].cpu().numpy())
                
                    plt.scatter(future_samples_batch[:,:,0]/resize, future_samples_batch[:,:,1]/resize, s=5, marker=".", c='r', alpha=0.5)
                    plt.show()

                gt_goal = gt_future[:, -1:]

                # converts ETH/UCY pixel coordinates back into world-coordinates
                if dataset_name == 'eth':
                    waypoint_samples = image2world(waypoint_samples, scene, homo_mat, resize)
                    pred_traj = image2world(pred_traj, scene, homo_mat, resize)
                    gt_future = image2world(gt_future, scene, homo_mat, resize)

                val_FDE.append(((((gt_goal - waypoint_samples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
                val_ADE.append(((((gt_future - future_samples) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])

                if i > 30:
                    vis = False
        val_ADE = torch.cat(val_ADE).mean()
        val_FDE = torch.cat(val_FDE).mean()

    return val_ADE.item(), val_FDE.item()