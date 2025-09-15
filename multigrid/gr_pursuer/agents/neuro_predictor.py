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
    raise ValueError("Please set ACTOR_PREDICTOR_MODEL_PATH environment variable.")

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

def get_location_icon(location, map_size: int, image_array: np.ndarray, env):
    """
    Split an image into a map_size x map_size grid of equal patches and
    return the patch at grid location (row, col).

    Works for 2D (H, W) and 3D (H, W, C) arrays.
    """
    
    env_id = id(env)
    if env_id not in GOAL_ICON_CACHE:
        GOAL_ICON_CACHE[env_id] = dict()
        
    if location in GOAL_ICON_CACHE[env_id]:
        return GOAL_ICON_CACHE[env_id][location]
    
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

    # save to cache 
    GOAL_ICON_CACHE[env_id][location] = patches[r, c]

    return patches[r, c]

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
    if ACTOR_PREDICTOR_MODEL is None or ACTOR_PREDICTOR_TOKENIZER is None:
        ACTOR_PREDICTOR_MODEL, ACTOR_PREDICTOR_TOKENIZER = load_actor_predictor_model()
        
    map_size = env.grid_size
    the_image = env.grid.render(tile_size=32, agents=env.unwrapped.agents[1:], highlight_mask=None)
    goal_desc = get_location_icon(goal, map_size, the_image, env)
    
    noise = np.random.normal(0, 10, the_image.shape).astype(np.uint8)
    image_aug = cv2.addWeighted(the_image, 0.9, noise, 0.1, 0)
    
    
    # * convert_to_conversation
    sample = {
        "image": image_aug,
        "goal_icon": goal_desc,
        "behavior_description": behavior_type
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
                
            breakpoint()
        
        # get logits for action tokens only
        action_logits = logits[:, ACTION_LABEL_IDS]
        action_probs = torch.softmax(action_logits, dim=-1) # shape (1, num_actions)
        
        # convert to cpu numpy
        # convert from BFloat16 to float32 first
        action_probs = action_probs.to(torch.float32)
        action_probs = action_probs.cpu().numpy().flatten()
        # logger.info(f"Action probabilities: {dict(zip(ACTION_LABELS, action_probs))}")

        
        # * convert from up right down left to forward left right using the information of successors
        agent_pos, agent_dir = pos_state
        agent_dir = int(agent_dir)
        # convert agent_dir to text 
        agent_dir_text = ACTION_LABELS[agent_dir]
        
        tran_probs_dict = dict()
        
        # process successor info 
        successor_text_dict = dict()
        for action, succ in successors:
            successor_text_dict[action.value] = succ
        
        for action_id, action in enumerate(ACTION_LABELS):
            if action == agent_dir_text: # means forward 
                tran_probs_dict['forward'] = action_probs[action_id]
            elif (agent_dir + 1) % 4 == action_id: # means right 
                tran_probs_dict['right'] = action_probs[action_id]
            elif (agent_dir - 1) % 4 == action_id: # means left
                tran_probs_dict['left'] = action_probs[action_id]
            elif (agent_dir + 2) % 4 == action_id: # means stay 
                tran_probs_dict['stay'] = action_probs[action_id]
            else:
                raise ValueError("Invalid action direction mapping.")
            
    return tran_probs_dict