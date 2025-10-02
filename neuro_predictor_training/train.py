from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, EarlyStoppingCallback
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import os 
from PIL import Image
from transformers import TextStreamer
from copy import deepcopy
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import wandb
import multiprocessing as mp
from tqdm.auto import tqdm
import random
from pathlib import Path

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
if_shuffle = False
USER_PROMPT = """Task: Predict the next action.


Behavior type: {behavior_description}

Observation: see the first image
Goal: see the second image

Format:
Return ONE action token from the allowed set. No extra words.
"""

dataset_path = "/home/sukai/Project/chenyuan_project/new_multi_grid/data/training_data/actor_moves_dataset"


def convert_to_conversation(sample):
    img_full_path = os.path.join(dataset_path, sample["image_path"])
    # image_obj = Image.open(img_full_path)
    goal_icon_full_path = Path(img_full_path).with_name(f"icon_{Path(img_full_path).name}")
    # goal_icon_obj = Image.open(goal_icon_full_path)
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
        { "role": "user",
          "content" : [
                {"type" : "text",  "text"  : USER_PROMPT.format(behavior_description=sample["behavior_description"])},
                {"type" : "image", "image" : f"file://{img_full_path}"},
                {"type": "image", "image": f"file://{goal_icon_full_path}"}
            ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["action"]} ]
        },
    ]
    return { "messages" : conversation }


if __name__ == "__main__":
    project_name = "qwen2.5_vlm_prediction_single_image" + "no_shuffle" if not if_shuffle else "shuffle"
    wandb.init(project="chenyuan_action_prediction", name=project_name)

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 128,          # Increased from 32 to 128 for higher capacity
        lora_alpha = 256, # Increased proportionally (2 * r)
        lora_dropout = 0.1, # Slightly increased to prevent overfitting with higher rank
        bias = "lora_only", # "none", "all", "lora_only"
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    
    dataset = load_dataset(dataset_path, split="train")
    a = list(dataset)
    
    with mp.Pool(16) as pool:
        # Use tqdm to visualize progress
        converted_dataset = list(
            tqdm(pool.imap(convert_to_conversation, a), total=len(a))
        )

    # shuffle
    if if_shuffle:
        random.shuffle(converted_dataset)
        
    # validation set
    val_dataset = load_dataset(dataset_path, split="validation")
    val_a = list(val_dataset)
    with mp.Pool(16) as pool:
        # Use tqdm to visualize progress
        val_converted_dataset = list(
            tqdm(pool.imap(convert_to_conversation, val_a), total=len(val_a))
        )
    


    # train the model 
    FastVisionModel.for_training(model) # Enable for training!

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = converted_dataset,
        eval_dataset = val_converted_dataset,  # Add validation dataset
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)],
        args = SFTConfig(
            per_device_train_batch_size = 12,   # Increased from 8 to 12 to utilize more VRAM
            per_device_eval_batch_size = 12,    # Match training batch size
            gradient_accumulation_steps = 1,    # Reduced to 1 since we have large batch size
            warmup_steps = 20,                  # Increased warmup for larger batch size
            num_train_epochs = 10, # Increase epochs since early stopping will handle this
            learning_rate = 1.5e-4,             # Slightly reduced LR for larger batch size
            logging_steps = 50,                 # More frequent logging
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs_noshuffle" if not if_shuffle else "outputs",
            report_to = "wandb",     # For Weights and Biases
            save_total_limit=2,                 # Keep more checkpoints with more VRAM
            save_strategy="steps",
            save_steps=250,                     # More frequent saves for better recovery
            
            # Memory optimization settings for 24GB VRAM
            dataloader_pin_memory=True,         # Pin memory for faster data transfer
            dataloader_num_workers=4,           # Parallel data loading
            fp16=False,                         # Keep bf16 from Unsloth (better than fp16)
            
            # Evaluation and early stopping settings
            eval_strategy="steps",          # Evaluate every eval_steps
            eval_steps=250,                 # More frequent evaluation (was 500)
            eval_accumulation_steps=1,      # Accumulate eval batches to save memory
            load_best_model_at_end=True,    # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Use validation loss as the metric
            greater_is_better=False,        # Lower loss is better
          
        ),
    )
    
    trainer_stats = trainer.train()
    
    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
    
    if wandb.run:
        # close it
        wandb.finish()