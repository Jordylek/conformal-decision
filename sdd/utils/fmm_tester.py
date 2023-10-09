import spline_planner_utils
import cv2
import numpy as np 
import matplotlib.pyplot as plt

filename = '../nexus_4_frame_bin.png'

# read image as grey scale
grey_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
print(grey_img)

# get raw signed distance
sd_obs = spline_planner_utils.sd_binary_image(grey_img)

# shape the signed distance
obstacle_thresh = -100
freespace_thresh = 10
close_to_obs_cost = -10
sd_obs[sd_obs > freespace_thresh] = 0
sd_obs[sd_obs > 0] = -1./sd_obs[sd_obs > 0]
sd_obs[sd_obs < obstacle_thresh] = obstacle_thresh
plt.imshow(sd_obs, cmap='RdBu')
plt.colorbar()

plt.savefig('../sd_obs_func.png')
#cv2.imwrite('../sd_obs_func.png', sd_obs)

# read image as numpy array
# rgb_img = np.load(filename)

# threshold for counting a pixel value as "obstacle"
#thresh = 200
#rgb_img = (255 * rgb_img).astype('uint8')
#binarize_img = spline_planner_utils.binarize_img(rgb_img, thresh, save_img=True)
#cv2.imwrite('../rgb_img_check.png', rgb_img)


