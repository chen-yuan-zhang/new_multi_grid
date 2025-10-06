from tqdm.auto import tqdm
from PIL import Image
import time
from pathlib import Path

import multiprocessing as mp 
import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger
import cv2 
from multigrid.core.grid import Grid
from multigrid.core.agent import Agent

from collections import OrderedDict, defaultdict

class FIFOCache:
    def __init__(self, capacity: int):
        """
        Initialize a FIFO cache with a specified capacity.
        
        Args:
            capacity: Maximum number of items the cache can hold
        """
        self.capacity = capacity
        self.cache = OrderedDict()  # Preserves insertion order
    
    def get(self, key):
        """
        Retrieve an item from the cache by key.
        
        Args:
            key: The key of the item to retrieve
            
        Returns:
            The value associated with the key, or None if not found
        """
        if key not in self.cache:
            return None
        return self.cache[key]
    
    def put(self, key, value):
        """
        Add or update an item in the cache.
        
        If the cache is full, the oldest item (first inserted) will be evicted.
        
        Args:
            key: The key of the item to add/update
            value: The value to associate with the key
        """
        # If key exists, remove it first to update its position (optional behavior)
        if key in self.cache:
            del self.cache[key]
        # If cache is full, remove the oldest item
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # last=False removes the first inserted item
        
        # Add the new item
        self.cache[key] = value
    
    def __str__(self):
        """String representation of the cache"""
        return str(dict(self.cache))

IMAGE_FIFO_CACHE = FIFOCache(capacity=10000)

DEBUG_MODE = False

# IMPORTANT chart rules for this notebook (kept simple):
# - Use matplotlib (not seaborn)
# - Single plot per figure (no subplots)
# - Do not set any specific colors/styles unless you want to later

# Action vocabulary (ordered)
ACTION_LABELS = ["right", "down", "left", "up"]

ACTION_LABEL_IDS = None

SYSTEM_PROMPT="""You are a visuomotor policy for action prediction.
Given (1) the current observation as an image, (2) a goal as an image icon, and
(3) a behavior type, you must output
ONLY the next action as a canonical token from the environment's action space.

Rules:
- Return exactly one action string. No explanations, no punctuation.
- Prefer actions consistent with the stated behavior type. If ambiguous, choose the action
  that maximizes expected progress toward the goal under that behavior.
- Valid action set: {right, down, left, up}
"""

USER_PROMPT = """Task: Predict the next action.


Behavior type: {behavior_description}

Observation: see the image
Goal: see the place with white dot

Format:
Return ONE action token from the allowed set. No extra words.
"""

USER_PROMPT_DEPRECATED = """Task: Predict the next action.


Behavior type: {behavior_description}

Observation: see the first image
Goal: see the second image

Format:
Return ONE action token from the allowed set. No extra words.
"""

# Load model and tokenizer
from unsloth import FastVisionModel
from transformers import AutoTokenizer

try:
    ACTOR_PREDICTOR_MODEL_PATH = os.environ["ACTOR_PREDICTOR_MODEL_PATH"]
except KeyError:
    print("Please set ACTOR_PREDICTOR_MODEL_PATH environment variable.")
    ACTOR_PREDICTOR_MODEL_PATH = None 
    # raise ValueError("Please set ACTOR_PREDICTOR_MODEL_PATH environment variable.")

ACTOR_PREDICTOR_MODEL = None
ACTOR_PREDICTOR_TOKENIZER = None 



def load_actor_predictor_model():
    global ACTOR_PREDICTOR_MODEL
    if ACTOR_PREDICTOR_MODEL is None:
        logger.info("Loading actor predictor model...")
        ACTOR_PREDICTOR_MODEL, ACTOR_PREDICTOR_TOKENIZER = FastVisionModel.from_pretrained(
            ACTOR_PREDICTOR_MODEL_PATH,
            load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(ACTOR_PREDICTOR_MODEL)
        logger.info("Actor predictor model loaded.")
    return ACTOR_PREDICTOR_MODEL, ACTOR_PREDICTOR_TOKENIZER

def convert_to_conversation(sample):
    img_pil = sample['image']
    if 'goal_icon' in sample:
        goal_icon_pil = sample['goal_icon']
        conversation_test = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            },
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : USER_PROMPT.format(behavior_description=sample["behavior_description"])},
                {"type" : "image", "image" : img_pil},
                {"type": "image", "image": goal_icon_pil}
            ]
            }
        ]
        return {"messages" : conversation_test }
    else:
        conversation_test = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            },
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : USER_PROMPT.format(behavior_description=sample["behavior_description"])},
                {"type" : "image", "image" : img_pil},
            ]
            }
        ]
        return {"messages" : conversation_test }

def process_message(message):
    global ACTOR_PREDICTOR_MODEL, ACTOR_PREDICTOR_TOKENIZER, ACTION_LABEL_IDS
    assert ACTOR_PREDICTOR_MODEL is not None and ACTOR_PREDICTOR_TOKENIZER is not None, "Model and tokenizer must be loaded."
    
    if ACTION_LABEL_IDS is None:
        # Map action labels to token IDs
        ACTION_LABEL_IDS = [ACTOR_PREDICTOR_TOKENIZER.tokenizer.convert_tokens_to_ids(label) for label in ACTION_LABELS]
        logger.info(f"Action labels mapped to token IDs: {dict(zip(ACTION_LABELS, ACTION_LABEL_IDS))}")
    
    images, texts = [], []
    messages = message["messages"]

    images.append(messages[-1]["content"][-2]["image"])
        
    images.append(messages[-1]["content"][-1]["image"])
        
    texts.append(ACTOR_PREDICTOR_TOKENIZER.apply_chat_template(messages, add_generation_prompt=True))
    
    inputs = ACTOR_PREDICTOR_TOKENIZER(
        images,
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    
    return inputs

GOAL_ICON_CACHE = dict()

def get_location_icon(location, map_size: int, image_array: np.ndarray):
    """
    Split an image into a map_size x map_size grid of equal patches and
    return the patch at grid location (row, col).

    Works for 2D (H, W) and 3D (H, W, C) arrays.
    """
    
    if image_array.ndim not in (2, 3):
        raise ValueError("image_array must be 2D (H, W) or 3D (H, W, C)")

    H, W = image_array.shape[:2]
    if H % map_size or W % map_size:
        raise ValueError(f"Image ({H}x{W}) not divisible into {map_size}x{map_size} grid.")

    c, r = location
    if not (0 <= r < map_size and 0 <= c < map_size):
        raise IndexError(f"location {location} out of bounds for {map_size}x{map_size} grid.")

    ph, pw = H // map_size, W // map_size  # patch height/width

    if image_array.ndim == 2:
        patches = image_array.reshape(map_size, ph, map_size, pw).swapaxes(1, 2)
        # shape: (map_size, map_size, ph, pw)
    else:
        C = image_array.shape[2]
        patches = image_array.reshape(map_size, ph, map_size, pw, C).swapaxes(1, 2)
        # shape: (map_size, map_size, ph, pw, C)


    return patches[r, c]

CALLING_COUNTER = 0
TIME_CHECKPOINT = time.time()


def local_render(grid, width, height, pos_state, tile_size, goal_data=None):
    highlight_mask = np.zeros(shape=(width, height), dtype=bool)
    agent_pos, agent_dir = pos_state
    # Get agent locations
    # For overlapping agents, non-terminated agents get priority
    location_to_agent = defaultdict(type(None))
    # create a dummy agent for rendering purpose
    dummy_agent = Agent(1)
    dummy_agent.color = 'green' # green is the actor agent color
    dummy_agent.dir = agent_dir
    location_to_agent[tuple(agent_pos)] = dummy_agent # 1 means the actor agent 

    # Initialize pixel array
    width_px = width * tile_size
    height_px = height * tile_size
    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for j in range(0, height):
        for i in range(0, width):
            assert highlight_mask is not None
            cell = grid.get(i, j)
            tile_img = Grid.render_tile(
                cell,
                agent=location_to_agent[i, j],
                highlight=highlight_mask[i, j],
                tile_size=tile_size,
            )

            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img
    # if goal_data is not None, then add a white color circle at the goal position
    if goal_data is not None:
        goal_x, goal_y = goal_data
        goal_x = int(goal_x)
        goal_y = int(goal_y)
        # draw a white circle at the center of the goal cell
        center_x = goal_x * tile_size + tile_size // 2
        center_y = goal_y * tile_size + tile_size // 2
        cv2.circle(img, (center_x, center_y), tile_size // 4, (255, 255, 255), -1) 

    return img


def neuro_predict(env, goal, behavior_type, successors, pos_state):
    """Predict action probabilities using the neuro predictor model.
    Args:
        env: The environment object with grid and agents.
        goal: The target goal location (row, col).
        behavior_type: Description of the behavior type (e.g., "like wall", "hate wall").
    Returns:
        action_probs: A numpy array of shape (num_actions,) with probabilities for each action. 
    """
    global ACTOR_PREDICTOR_MODEL, ACTOR_PREDICTOR_TOKENIZER, DEBUG_MODE
    
    global CALLING_COUNTER, TIME_CHECKPOINT
    CALLING_COUNTER += 1
    agent_pos, agent_dir = pos_state
    if CALLING_COUNTER % 50 == 0:
        current_time = time.time()
        elapsed = current_time - TIME_CHECKPOINT
        TIME_CHECKPOINT = current_time
        
        print(f"[neuro_predict] Called {CALLING_COUNTER} times. Time for last 50 calls: {elapsed:.2f}s")
    
     # Load model if not already loaded
    
    
    if ACTOR_PREDICTOR_MODEL is None or ACTOR_PREDICTOR_TOKENIZER is None:
        ACTOR_PREDICTOR_MODEL, ACTOR_PREDICTOR_TOKENIZER = load_actor_predictor_model()
        
    map_size = env.grid_size
    
    
    # the_image_key = (id(env), agent_pos, agent_dir)
    id_env = tuple(env.base_grid.flatten().tolist())
    the_image_key = (id_env, agent_pos, agent_dir)
    if the_image_key in IMAGE_FIFO_CACHE.cache:
        the_image = IMAGE_FIFO_CACHE.get(the_image_key)
    else:
        
        width, height = env.width, env.height
        grid = env.grid
        tile_size = 13  # Increased tile size for better resolution
        the_image = local_render(grid, width, height, pos_state, tile_size, goal_data=goal)
        IMAGE_FIFO_CACHE.put(the_image_key, the_image)
        
    # # debug image save 
    # save_dir =  '/home/sukai/Project/chenyuan_project/new_multi_grid/debug_images'
    # Path(save_dir).mkdir(parents=True, exist_ok=True)
    # debug_image_path = os.path.join(save_dir, f"debug_image_calling_count_{CALLING_COUNTER}_pos_{agent_pos[0]}_{agent_pos[1]}_dir_{agent_dir}.png")
    # the_image_pil = Image.fromarray(the_image)
    # the_image_pil.save(debug_image_path)
    
    # # --- end of debug ---
    
    # goal_desc_key = (id(env), goal) DEPRECATED
    # goal_desc_key = (id_env, goal, map_size)
    # if goal_desc_key in IMAGE_FIFO_CACHE.cache:
    #     goal_desc = IMAGE_FIFO_CACHE.get(goal_desc_key)
    # else:
    #     goal_desc = get_location_icon(goal, map_size, the_image)
    #     IMAGE_FIFO_CACHE.put(goal_desc_key, goal_desc)
        
    # noise = np.random.normal(0, 10, the_image.shape).astype(np.uint8)
    # image_aug = cv2.addWeighted(the_image, 0.9, noise, 0.1, 0)
    
    
    # * convert_to_conversation
    sample = {
        "image": the_image,
        # "goal_icon": goal_desc,
        "behavior_description": int(behavior_type)
    } 
    
    message = convert_to_conversation(sample)
    
    inputs = process_message(message)
    
    # * Inference Step 
    with torch.inference_mode():
        logits = ACTOR_PREDICTOR_MODEL(**inputs).logits[:, -1, :]  # Logits for next token
        
        # DEBUG
        if DEBUG_MODE:
            prob = torch.softmax(logits, dim=-1)
            topk = torch.topk(prob, k=5) # shape: [B, 5]
            top_tokens = topk.indices
            top_probs = topk.values
            print("Top tokens and probs:")
            for idx, (tok, p) in enumerate(zip(top_tokens[0], top_probs[0])):
                print(f"Top {idx + 1}: {ACTOR_PREDICTOR_TOKENIZER.decode(tok)}, {float(p)}")
                
        
        # get logits for action tokens only
        action_logits = logits[:, ACTION_LABEL_IDS]
        action_probs = torch.softmax(action_logits, dim=-1) # shape (1, num_actions)
        
        # convert to cpu numpy
        # convert from BFloat16 to float32 first
        action_probs = action_probs.to(torch.float32)
        action_probs = action_probs.cpu().numpy().flatten()
        # logger.info(f"Action probabilities: {dict(zip(ACTION_LABELS, action_probs))}")

        
        # * convert from up right down left to forward left right using the information of successors
        agent_dir = int(agent_dir)
        # convert agent_dir to text 
        agent_dir_text = ACTION_LABELS[agent_dir]
        
        tran_probs_dict = dict()
        
        # process successor info 
        successor_text_dict = dict()
        for action, succ in successors:
            successor_text_dict[action.name] = succ
        for action_id, action in enumerate(ACTION_LABELS):
            if action == agent_dir_text: # means forward 
                if successor_text_dict.get('forward') is not None:
                    tran_probs_dict[successor_text_dict['forward']] = action_probs[action_id]
            elif (agent_dir + 1) % 4 == action_id: # means right 
                if successor_text_dict.get('right') is not None:
                    tran_probs_dict[successor_text_dict['right']] = action_probs[action_id]
            elif (agent_dir - 1) % 4 == action_id: # means left
                if successor_text_dict.get('left') is not None:
                    tran_probs_dict[successor_text_dict['left']] = action_probs[action_id]
            elif (agent_dir + 2) % 4 == action_id: # means stay
                if successor_text_dict.get('stay') is not None:
                    tran_probs_dict[successor_text_dict['stay']] = action_probs[action_id]
            else:
                raise ValueError("Invalid action direction mapping.")
            
    return tran_probs_dict