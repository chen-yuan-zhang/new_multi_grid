import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Optional: set GPU device ID

from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel, is_bfloat16_supported
import torch
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from PIL import Image
import wandb
import multiprocessing as mp
from tqdm.auto import tqdm
import random
from pathlib import Path
import numpy as np

SYSTEM_PROMPT = """You are a visuomotor policy for action prediction.
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


# All possible actions
ALL_ACTIONS = ["right", "down", "left", "up"]

dataset_path = "/home/sukai/Project/chenyuan_project/new_multi_grid/data/training_data/actor_moves_dataset_single_image"
sft_model_path = "/home/sukai/Project/chenyuan_project/new_multi_grid/data/trained_models/neuro_predictor_float16"  # Path to your SFT trained model


def reward_function(prompts, completions, answer, **kwargs):
    """
    Reward function for GRPO - checks if generated action matches the correct action
    
    Args:
        prompts: List of prompt conversations (not used directly, but required by GRPO)
        completions: List of completion dictionaries, each with 'content' field containing the generated text
        answer: List of correct action strings from the dataset
        **kwargs: Additional keyword arguments from the dataset
    
    Returns:
        List of reward scores (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    
    # Extract generated text from completions
    # completions is a list of lists, where each inner list contains message dicts
    for completion, correct_answer in zip(completions, answer):
        # Get the generated text from the completion
        # completion[0]['content'] contains the actual generated action
        if isinstance(completion, list) and len(completion) > 0:
            generated_text = completion[0].get('content', '').strip().lower()
        else:
            generated_text = str(completion).strip().lower()
        
        # Get correct action
        correct_action = correct_answer.strip().lower()
        
        # Check if the generated action is in the valid action set
        if generated_text in [a.lower() for a in ALL_ACTIONS]:
            # Give reward of 1.0 if correct, 0.0 if incorrect
            reward = 1.0 if generated_text == correct_action else 0.0
        else:
            # Penalize invalid actions (hallucinations, explanations, etc.)
            reward = -0.5
        
        rewards.append(reward)
    
    # Optional: Print first example for debugging
    if len(prompts) > 0 and len(completions) > 0:
        print('-'*50)
        print(f"Generated: {completions[0]}")
        print(f"Expected: {answer[0]}")
        print(f"Reward: {rewards[0]}")
        print('-'*50)
    
    return rewards


def convert_to_grpo_format(sample):
    """Convert a sample to GRPO format with prompt, images, and answer"""
    img_full_path = os.path.join(dataset_path, sample["image_path"])
    
    # Load images
    obs_image = Image.open(img_full_path)
    if obs_image.mode != 'RGB':
        obs_image = obs_image.convert('RGB')
    
    # Create prompt in GRPO format
    prompt = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : USER_PROMPT.format(behavior_description=sample["behavior_description"])},
            {"type" : "image"},
        ]
        },
    ]
    
    return {
        "prompt": prompt,
        "image": obs_image.resize((512, 512)),  
        "answer": sample["action"]  # The correct action
    }


if __name__ == "__main__":
    # Configuration
    use_sft_model = True  # Set to True if you want to load your SFT model, False to start from base
    max_seq_length = 2048
    
    project_name = "qwen2.5_vlm_grpo_action_prediction"
    wandb.init(project="chenyuan_action_prediction", name=project_name)

    # Load model - either from your SFT checkpoint or base model
    if use_sft_model and os.path.exists(sft_model_path):
        print(f"Loading SFT model from {sft_model_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=sft_model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            use_gradient_checkpointing="unsloth",
            gpu_memory_utilization=0.6
        )
     
        
    else:
        raise ValueError("Please provide a valid path to your SFT model.")
    # Load and convert datasets to GRPO format
    print("Loading training dataset...")
    dataset = load_dataset(dataset_path, split="validation")
    a = list(dataset)
    
    print("Converting training dataset to GRPO format...")
    with mp.Pool(16) as pool:
        train_grpo_list = list(
            tqdm(pool.imap(convert_to_grpo_format, a), total=len(a))
        )
    
    # Convert list to HuggingFace Dataset
    print("Creating HuggingFace Dataset from converted data...")
    train_grpo_dataset = Dataset.from_list(train_grpo_list)
    
    # apply chat template 
    train_grpo_dataset = train_grpo_dataset.map(
        lambda example: {
            "prompt": tokenizer.apply_chat_template(
                example["prompt"],
                tokenize = False,
                add_generation_prompt = True, # Must add assistant
            )
        }
    )
    
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 1.0,
        top_k = 50,
        max_tokens = 1024,
    )

    outputs = model.fast_generate(
        {
            "prompt": train_grpo_dataset[100]["prompt"],
            "multi_modal_data": {"image": train_grpo_dataset[100]["image"]}
        },
        sampling_params,
    )
    print(outputs[0].outputs[0].text)
    input("Press Enter to continue...")
    
    # Enable training mode
    FastVisionModel.for_training(model)
    # display trainable parameters
    model.print_trainable_parameters()
    input("Press Enter to continue...")

    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 20,
        log_completions = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = 1024,
        max_completion_length = 1024,
        num_train_epochs = 2, # Set to 1 for a full training run
        save_steps = 60,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = "./grpo_model_output",

        # Below enables GSPO:
        importance_sampling_level = "sequence",
        mask_truncated_completions = False,
        loss_type = "dr_grpo",
    )
    
    trainer = GRPOTrainer(
        model = model,
        args = training_args,
        # Pass the processor to handle multimodal inputs
        processing_class = tokenizer,
        reward_funcs = [
            reward_function
        ],
        train_dataset = train_grpo_dataset,
    )

    trainer.train()
        
    # Save the final model
    print("Saving GRPO model...")
    model.save_pretrained("lora_model_grpo")
    tokenizer.save_pretrained("lora_model_grpo")
    
    if wandb.run:
        wandb.finish()
    
    print("GRPO training complete!")