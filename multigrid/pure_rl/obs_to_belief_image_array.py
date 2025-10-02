import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def obs_to_belief_image_array(belief_update_observer, filename, obs, add_noise=True):
    """
    Render the environment and save the visualization as an image array.
    
    Parameters:
    filename (str): The name of the file to save the visualization.
    obs (dict): The observation dictionary containing the observer and goal positions.
    
    Returns:
    np.ndarray: Image array (H, W, 3).
    """
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    sample_goal = next(iter(belief_update_observer.actor_belief))
    sample_grid = belief_update_observer.actor_belief[sample_goal][0]
    total_belief = np.zeros_like(sample_grid)
    
    # Sum across all goals and behavior types
    for goal, belief_list in belief_update_observer.actor_belief.items():
        for behavior_grid in belief_list:
            regular_space_grid = np.exp(np.clip(behavior_grid, -700, 700))
            total_belief += regular_space_grid

    belief_sum = np.sum(total_belief, axis=2)
    log_belief_sum = np.log(belief_sum + 1e-10) # shape (H,W)
    # convert to W x H for matplotlib
    log_belief_sum = log_belief_sum.T
    vmin = np.min(log_belief_sum)
    vmax = np.max(log_belief_sum)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(log_belief_sum, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    goal_colors = ['yellow', 'green', 'cyan', 'magenta', 'orange']
    goal_probs = [belief_update_observer.goal_belief[goal] for goal in belief_update_observer.goals]
    goal_text = '\n'.join([f'Goal {i+1} ({goal_colors[i % len(goal_colors)]}): {prob:.2f}' 
                           for i, prob in enumerate(goal_probs)])
    ax.set_title(goal_text)
    
    obstacles = np.where(belief_update_observer.env.base_grid != 0)
    ax.scatter(obstacles[0], obstacles[1], c='black', marker='s', label='Obstacle')
    
    observer_pos = obs['observer_pos']
    ax.scatter(observer_pos[0], observer_pos[1], c='blue', marker='o', label='Observer')
    
    for i, goal in enumerate(belief_update_observer.goals):
        ax.scatter(goal[0], goal[1], c=goal_colors[i % len(goal_colors)], marker='*', label='Goal')
    
    if "target_pos" in obs:
        target_pos = obs["target_pos"]
        ax.scatter(target_pos[0], target_pos[1], c='red', marker='x', label='Target')
    
    # Remove axis ticks for a cleaner image
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.legend(loc='upper right', fontsize='x-small')

    # Save to file if a filename is supplied
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    # Convert to numpy array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
    # call transpose to convert ARGB to RGBA
    image_array = image_array.reshape((height, width, 4)).astype('uint8')
    image_array = image_array[:, :, [1, 2, 3]]  # convert to RGB
    # transpose width and height
    # add some noise 
    if add_noise:
        noise = np.random.normal(0, 10, image_array.shape).astype(np.uint8)
        image_array = cv2.addWeighted(image_array, 0.9, noise, 0.1, 0)

    img_resized = cv2.resize(image_array, (124, 124), interpolation=cv2.INTER_AREA)
    
    
    plt.close(fig)
    return img_resized, log_belief_sum


def preprocess_obs_for_rl_policy(belief_update_observer, obs):
    """
    Preprocess the observation for RL policy input.
    
    Parameters:
    belief_update_observer: The belief update observer object.
    obs (dict): The observation dictionary containing the observer and goal positions.

    Returns:
    np.ndarray: Processed image array (H, W, 3).
    """
    belief_img, _ = obs_to_belief_image_array(belief_update_observer, None, obs, add_noise=True)
    belief_img = (belief_img.astype(np.float32) / 128.0) - 1.0
    
    return belief_img
   