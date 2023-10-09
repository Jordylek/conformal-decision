import numpy as np

from . import spline_planner_utils
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from skimage.transform import resize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time
import numba
import cv2

import pdb

class SplinePlanner:
    # Construct SplinePlanner object.
    def __init__(self, num_waypts, horizon,
                 max_linear_vel, max_angular_vel,
                 gmin, gmax, gnums, goal_rad, human_rad,
                lr, alpha, binary_env_img_path=None):

        self.num_waypts_ = num_waypts
        self.horizon_ = horizon
        self.max_linear_vel_ = max_linear_vel
        self.max_angular_vel_ = max_angular_vel
        self.gmin_ = gmin
        self.gmax_ = gmax
        self.gnums_ = gnums
        self.sd_obs_ = None
        self.sd_goal_ = None
        self.goal_rad = goal_rad
        self.human_rad = human_rad
        self.lamda = 0
        self.lr = lr
        self.alpha = alpha
        self.alphat = alpha

        self.X, self.Y = np.meshgrid(np.linspace(gmin[0], gmax[0], gnums[0]), \
                            np.linspace(gmin[1], gmax[1], gnums[1]))

        self.disc_xy_ = np.column_stack((self.X.flatten(), self.Y.flatten()))

        if binary_env_img_path is not None:
            print("[Spline Planner] creating obstacle signed distance function...")
            # load the binary env img
            binary_env_img = cv2.imread(binary_env_img_path, cv2.IMREAD_GRAYSCALE)

            # get raw signed distance
            sd_obs = spline_planner_utils.sd_binary_image(binary_env_img)

            # shape the signed distance
            obstacle_thresh = -100. # distance "deep" into obstacle after which you always get this cost
            freespace_thresh = 10. # distance "away" from obstacles after which you get no cost

            sd_obs[sd_obs > freespace_thresh] = 0
            sd_obs[sd_obs > 0] = -1./sd_obs[sd_obs > 0] # add a small buffer around obstacles where you get small penalty
            sd_obs[sd_obs < obstacle_thresh] = obstacle_thresh

            self.sd_obs_ = resize(sd_obs, (gnums[0], gnums[1]))
            self.sd_obs_interp_ = NearestNDInterpolator(list(zip( self.X.flatten(), self.Y.flatten()  )), self.sd_obs_.flatten())
            self.sd_obs_interp_ = resize(self.sd_obs_interp_(np.column_stack((self.X.flatten(), self.Y.flatten()))), (binary_env_img.shape[0], binary_env_img.shape[1]))
            print("[Spline Planner] created signed distance function interpolator!")

        self.qh = None
        #self.ax = plt.axes()

    def r_goal(self, traj, goal):
        return -1000*np.linalg.norm(traj - goal[:2, None]) # First two coordinates are position, last two are v/angle

    def r_goal_sd(self, goal): # deprecated
        return -1 * spline_planner_utils.sd_circle(self.X, self.Y, goal[0:2], self.goal_rad)

    # Plans a path from start to goal.
    def plan(self, start, goal, pred_splines, method, worst_residuals=None):
        opt_spline, opt_reward, loss = plan_numba(np.array(start), goal, pred_splines, np.array(self.gmin_), np.array(self.gmax_), self.disc_xy_, self.horizon_, self.num_waypts_, self.max_linear_vel_, self.max_angular_vel_, self.human_rad, self.sd_obs_interp_, self.lamda, worst_residuals, self.alphat, method)
        if opt_spline is None:
            loss = 0
            opt_spline = np.stack([
                start[0] * np.ones(self.horizon_),
                start[1] * np.ones(self.horizon_),
                np.zeros(self.horizon_),
                np.zeros(self.horizon_),
                start[2] * np.ones(self.horizon_)
            ])
        if method in ['conformal controller', 'proactive conformal controller']:
            self.lamda += self.lr * (loss + self.human_rad)
            self.lamda = max(self.lamda, 0)
        if method == 'aci':
            self.alphat -= self.lr * (loss - self.alpha)
        return opt_spline

# Jitted version of the plan function
@numba.njit
def plan_numba(start, goal, pred_splines, gmin, gmax, disc_xy_, horizon_, num_waypts_, max_linear_vel_, max_angular_vel_, human_rad, sd_obs_interp, lamda, worst_residuals, alphat, method):
    opt_reward = -1e15
    opt_spline = None
    curr_spline = None
    opt_loss = None

    if method == 'aci':
        if alphat <= 1/worst_residuals.shape[0]:
            conformal_quantile = np.infty
            return None, None, 0
        elif alphat >= 1:
            conformal_quantile = 0
        else:
            conformal_quantile = np.quantile(worst_residuals[:-1], 1-alphat)
        last_residual = worst_residuals[-1]

    for ti in range(len(disc_xy_)):

        candidate_goal = disc_xy_[ti, :]
        candidate_goal = np.append(candidate_goal, [goal[2], goal[3]])

        arrs = spline(start, candidate_goal, horizon_, num_waypts_)

        # Determine the common shape and create an empty result array
        ## Numba version
        shape = (len(arrs),) + arrs[0].shape
        curr_spline = np.empty(shape, dtype=arrs[0].dtype)
        # Assign each array to the appropriate slice of the result
        for i, arr in enumerate(arrs):
            curr_spline[i] = arr
        ## Reg version
        #curr_spline = np.stack(arrs, axis=0)

        curr_spline[0] = np.clip(curr_spline[0], gmin[0], gmax[0])
        curr_spline[1] = np.clip(curr_spline[1], gmin[1], gmax[1])

        feasible_horizon = compute_dyn_feasible_horizon(curr_spline,
                                                        max_linear_vel_,
                                                        max_angular_vel_,
                                                        horizon_)

        if feasible_horizon <= horizon_:
            if method == 'conformal controller':
                reward, loss = eval_reward_numba_conformal_controller(goal, curr_spline[:2, :], curr_spline[2,:], pred_splines, human_rad, sd_obs_interp, lamda, False)
            if method == 'proactive conformal controller':
                reward, loss = eval_reward_numba_conformal_controller(goal, curr_spline[:2, :], curr_spline[2,:], pred_splines, human_rad, sd_obs_interp, lamda, True)
            if method == 'aci':
                reward, loss = eval_reward_numba_aci(goal, curr_spline[:2, :], curr_spline[2,:], pred_splines, human_rad, sd_obs_interp, conformal_quantile, last_residual)
            if method == 'conservative':
                reward, loss = eval_reward_numba_conservative(goal, curr_spline[:2, :], curr_spline[2,:], pred_splines, human_rad, sd_obs_interp)

            if reward > opt_reward:
                opt_reward = reward
                opt_spline = curr_spline
                opt_loss = loss

    return opt_spline, opt_reward, opt_loss

# Computes dynamically feasible horizon (given dynamics of car).
@numba.njit
def compute_dyn_feasible_horizon(spline,
                          max_linear_vel,
                          max_angular_vel,
                          final_t):

  # Compute max linear and angular speed.
  plan_max_lin_vel = np.max(spline[2]);
  plan_max_angular_vel = np.max(spline[3]);

  # Compute required horizon to acheive max speed of planned spline.
  feasible_horizon_speed = final_t * plan_max_lin_vel / max_linear_vel

  # Compute required horizon to acheive max angular vel of planned spline.
  feasible_horizon_angular_vel = final_t * plan_max_angular_vel / max_angular_vel

  feasible_horizon = max(feasible_horizon_speed, feasible_horizon_angular_vel)

  return feasible_horizon

# Evaluates the total reward along the trajectory.
@numba.njit
def eval_reward_numba_conformal_controller(goal, traj, vel, pred_splines, human_rad, sd_obs_interp, lamda, proactive):
  goal_r = -1.*np.linalg.norm(traj - goal[:2, None])

  obs_r = 0
  xs = traj[0]
  ys = traj[1]
  for i in range(xs.shape[0]):
    obs_r += 1000*sd_obs_interp[int(ys[i])-1, int(xs[i])-1]


  # NOTE: assumes that prediction length and time aligns!
  differences = pred_splines - traj[None, :]
  norms = np.sqrt((differences ** 2).sum(axis=1))
  # Radius of gaussian with std = human_rad/2
  #rad = spline_planner_utils.confidence_interval_half_width_numba(alphat, human_rad/100, 1)
  human_r = 1000*lamda*norms.min()#-10000*lamda*np.any(norms <= human_rad) # Bad if we are within a CI
  #print(goal_r, human_r, obs_r)

  if proactive:
    loss = -norms.min()
  else:
    loss = -norms[:,0].min()

  reward = goal_r + obs_r + human_r

  return reward, loss

@numba.njit
def eval_reward_numba_aci(goal, traj, vel, pred_splines, human_rad, sd_obs_interp, conformal_quantile, last_residual):
  goal_r = -1.*np.linalg.norm(traj - goal[:2, None])

  obs_r = 0
  xs = traj[0]
  ys = traj[1]
  for i in range(xs.shape[0]):
      obs_r += 10000*sd_obs_interp[int(ys[i])-1, int(xs[i])-1]


  # NOTE: assumes that prediction length and time aligns!
  differences = pred_splines - traj[None, :]
  norms = np.sqrt((differences ** 2).sum(axis=1))
  # Radius of gaussian with std = human_rad/2
  #rad = spline_planner_utils.confidence_interval_half_width_numba(alphat, human_rad/100, 1)
  human_r = -10000*np.any(norms <= conformal_quantile) # Bad if we are within a CI
  #print(goal_r, human_r, obs_r)

  loss = 0 if last_residual <= conformal_quantile else 1

  reward = goal_r + obs_r + human_r

  return reward, loss

@numba.njit
def eval_reward_numba_conservative(goal, traj, vel, pred_splines, human_rad, sd_obs_interp):
  goal_r = -1.*np.linalg.norm(traj - goal[:2, None])

  obs_r = 0
  xs = traj[0]
  ys = traj[1]
  for i in range(xs.shape[0]):
    obs_r += 1000*sd_obs_interp[int(ys[i])-1, int(xs[i])-1]


  # NOTE: assumes that prediction length and time aligns!
  differences = pred_splines - traj[None, :]
  norms = np.sqrt((differences ** 2).sum(axis=1))
  # Radius of gaussian with std = human_rad/2
  #rad = spline_planner_utils.confidence_interval_half_width_numba(alphat, human_rad/100, 1)
  human_r = 1000*norms.min()
  #print(goal_r, human_r, obs_r)

  loss = 0

  reward = goal_r + obs_r + human_r

  return reward, loss



#   Computes a 3rd order spline of the form:
#      p(t) = a3(t/final_t)^3 + b3(t/final_t)^2 + c3(t/final_t) + d3
#      x(p) = a1p^3 + b1p^2 + c1p + d1
#      y(p) = a2p^2 + b2p^2 + c2p + d2
@numba.njit
def spline(start, goal, final_t, num_waypts):

  # Get coefficients for the spline.
  (xc, yc, pc) = gen_coeffs(start, goal, final_t);

  # Unpack x coeffs.
  a1 = xc[0]
  b1 = xc[1]
  c1 = xc[2]
  d1 = xc[3]

  # Unpack y coeffs.
  a2 = yc[0]
  b2 = yc[1]
  c2 = yc[2]
  d2 = yc[3]

  # Unpack p coeffs.
  a3 = pc[0]
  b3 = pc[1]
  c3 = pc[2]
  d3 = pc[3]

  # Compute state trajectory at time steps using coefficients.
  xs = np.zeros(num_waypts)
  ys = np.zeros(num_waypts)
  xsdot = np.zeros(num_waypts)
  ysdot = np.zeros(num_waypts)
  ps = np.zeros(num_waypts)
  psdot = np.zeros(num_waypts)
  ths = np.zeros(num_waypts)

  # Compute the control: u1 = linear vel, u2 = angular vel
  u1_lin_vel = np.zeros(num_waypts)
  u2_ang_vel = np.zeros(num_waypts)

  # Compute timestep between each waypt.
  dt = final_t/(num_waypts-1.)
  idx = 0
  t = 0
  #print("numwaypts: ", num_waypts)
  #print("dt: ", dt)
  #print("final_t: ", final_t)

  while (idx < num_waypts):
    tnorm = t/final_t

    # Compute (normalized) parameterized time var p and x,y and time derivatives of each.
    ps[idx]   = a3 * tnorm**3   + b3 * tnorm**2   + c3 * tnorm   + d3;
    xs[idx]   = a1 * ps[idx]**3 + b1 * ps[idx]**2 + c1 * ps[idx] + d1;
    ys[idx]   = a2 * ps[idx]**3 + b2 * ps[idx]**2 + c2 * ps[idx] + d2;
    xsdot[idx]  = 3. * a1 * ps[idx]**2 + 2 * b1 * ps[idx] + c1;
    ysdot[idx]  = 3. * a2 * ps[idx]**2 + 2 * b2 * ps[idx] + c2;
    psdot[idx]  = 3. * a3 * tnorm**2 + 2 * b3 * tnorm + c3;
    ths[idx] = np.arctan2(ysdot[idx], xsdot[idx]);

    # Compute speed (wrt time variable p).
    speed = np.sqrt(xsdot[idx]**2 + ysdot[idx]**2);

    xsddot = 6. * a1 * ps[idx] + 2. * b1
    ysddot = 6. * a2 * ps[idx] + 2. * b2

    # Linear Velocity (real-time):
    #    u1(t) = u1(p(t)) * dp(tnorm)/dtorm * dtnorm/dt
    u1_lin_vel[idx] = speed * psdot[idx] / final_t;

    # Angular Velocity (real-time):
    #    u2(t) = u2(p(t)) * dp(tnorm)/dtorm * dtnorm/dt
    u2_ang_vel[idx] = (xsdot[idx]*ysddot - ysdot[idx]*xsddot)/speed**2 * psdot[idx] / final_t;

    idx = idx + 1
    t = t + dt

  curr_spline = [xs, ys, u1_lin_vel, u2_ang_vel, ths]

  return curr_spline

@numba.njit
def gen_coeffs(start, goal, final_t):
  # Extract states.
  x0 = start[0]
  y0 = start[1]
  th0 = start[2]
  v0 = start[3]

  xg = goal[0]
  yg = goal[1]
  thg = goal[2]
  vg = goal[3]

  # Set heuristic coefficients.
  f1 = v0 + np.sqrt((xg - x0)*(xg - x0) + (yg - y0)*(yg - y0))
  f2 = f1

  # Compute x(p(t)) traj coeffs.
  d1 = x0
  c1 = f1*np.cos(th0)
  a1 = f2*np.cos(thg) - 2.*xg + c1 + 2.*d1
  b1 = 3.*xg - f2*np.cos(thg) - 2.*c1 - 3.*d1

  # Compute y(p(t))traj coeffs.
  d2 = y0
  c2 = f1*np.sin(th0)
  a2 = f2*np.sin(thg) - 2.*yg + c2 + 2.*d2
  b2 = 3.*yg - f2*np.sin(thg) - 2.*c2 - 3.*d2

  # Compute p(t) coeffs.
  d3 = 0.0;
  c3 = (final_t * v0) / f1
  a3 = (final_t * vg) / f2 + c3 - 2.
  b3 = 1. - a3 - c3

  xc = [a1, b1, c1, d1]
  yc = [a2, b2, c2, d2]
  pc = [a3, b3, c3, d3]

  return (xc, yc, pc)


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
