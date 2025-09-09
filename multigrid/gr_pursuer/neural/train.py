import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import hydra
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .model import TransformerPredictor, CosineWarmupScheduler, PositionalEncoding
from .logger import Logger, WanDBLogger
from .dataset import TargetDataset, load_scenarios, collate_fn
from .metrics import metric_funcs, compute_metrics, CumulativeMetrics

loggers = {
    "base": Logger,
    "wandb": WanDBLogger,
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None: 

    SIZE = cfg["env"]["size"]
    OBSERVATION_WINDOW = cfg["model"]["observation_window"]
    NEPOCHS = cfg["model"]["nepochs"]
    LR = cfg["model"]["lr"]
    NUM_LAYERS = cfg["model"]["num_layers"]
    NUM_HEADS = cfg["model"]["num_heads"]

    HIDDEN_COST_TYPE = cfg["env"]["hidden_cost_type"]
    GOAL_GT = cfg["env"]["goal_gt"]
    DATA_LENGTH = cfg["data"]["length"]
    LAYOUT_SHUFFLE = cfg["data"]["layout_shuffle"]
    DATA_PATH = Path(cfg["data"]["path"].format(SIZE, HIDDEN_COST_TYPE, DATA_LENGTH, int(LAYOUT_SHUFFLE)))
    save_path = cfg["model"]["save_path"]
    tmp_epoch_save_path = cfg["model"]["tmp_epoch_save_path"]

    # Add additional parameters config:
    parameters = OmegaConf.to_container(cfg, resolve=True)
    parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the logger
    metrics = ["partition"] + list(metric_funcs.keys())
    logger = loggers[cfg.logger.type](
        parameters, show_keys=["epoch", "loss_train", "f1_train", "loss_valid", "f1_valid"]
    )
    
    classes = [0,1,2]

    scenarios = load_scenarios(DATA_PATH, OBSERVATION_WINDOW)#.groupby("PARTITION").sample(2)
    dataset = TargetDataset(DATA_PATH, scenarios, OBSERVATION_WINDOW)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                            collate_fn=lambda x: collate_fn(x, DATA_PATH, SIZE, GOAL_GT))

    # Datasets
    print("Loading Data Loaders")
    train_dataset = TargetDataset(DATA_PATH, scenarios[scenarios["PARTITION"]=="TRAIN"].reset_index(), OBSERVATION_WINDOW)
    valid_dataset = TargetDataset(DATA_PATH, scenarios[scenarios["PARTITION"]=="VALID"].reset_index(), OBSERVATION_WINDOW)
    test_dataset = TargetDataset(DATA_PATH, scenarios[scenarios["PARTITION"]=="TEST"].reset_index(), OBSERVATION_WINDOW)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, 
                            collate_fn=lambda x: collate_fn(x, DATA_PATH, SIZE, GOAL_GT))
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, 
                            collate_fn=lambda x: collate_fn(x, DATA_PATH, SIZE, GOAL_GT))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            collate_fn=lambda x: collate_fn(x, DATA_PATH, SIZE, GOAL_GT))

    # Dynamically determine input dimension
    state_dim = 6
    action_dim = 3
    goals_dim = 2 if GOAL_GT else 6
    grid_dim = SIZE**2
    mlp_grid_dim = [grid_dim, 256, 64, 30]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(state_dim=state_dim, goals_dim=goals_dim, mlp_grid_dim=mlp_grid_dim, 
                                 window=OBSERVATION_WINDOW, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, 
                                 output_dim=action_dim, enable_positional_encoding=cfg["model"]["positional_encoding"]).to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = None
    if cfg["model"]["scheduler"] == "cosine":
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, 
                                            warmup=cfg["model"]["warmup_steps"], max_iters=len(train_dataloader)*NEPOCHS)

    # Create tmp model save folder
    if not os.path.exists(tmp_epoch_save_path):
        os.makedirs(tmp_epoch_save_path)

    # Training Loop
    print(f"Starting training with {DATA_PATH}")
    epoch_stats = []
    
    for epoch in range(NEPOCHS):
        model.train()
        train_loss = 0
        train_metrics = CumulativeMetrics(classes, posfix="train")
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["action"])
            loss.backward()
            optimizer.step()
            if lr_scheduler: lr_scheduler.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=train_loss / (progress_bar.n + 1))

            probs = torch.nn.functional.softmax(outputs, dim=1)
            y_pred = torch.argmax(probs, dim=1).cpu().numpy()
            y_true = torch.argmax(batch["action"], dim=1).cpu().numpy()
            train_metrics.update(y_true, y_pred)
       

        model.eval()
        val_loss = 0
        val_metrics = CumulativeMetrics(classes, posfix="valid")
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in progress_bar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch)
                loss = criterion(outputs, batch["action"]).item() 
                val_loss += loss 
                progress_bar.set_postfix(loss=val_loss, avg_loss=val_loss / (progress_bar.n + 1))

                probs = torch.nn.functional.softmax(outputs, dim=1)
                y_pred = torch.argmax(probs, dim=1).cpu().numpy()
                y_true = torch.argmax(batch["action"], dim=1).cpu().numpy()
                val_metrics.update(y_true, y_pred)
 
        train_metrics = train_metrics.compute()
        val_metrics = val_metrics.compute()
        epoch_log = {
            "epoch": epoch, 
            "loss_train": train_loss/len(train_dataloader), 
            "loss_valid": val_loss/len(valid_dataloader)
        }
        epoch_log.update(train_metrics)
        epoch_log.update(val_metrics)
        epoch_stats.append(epoch_log)
        logger.log_epoch(epoch_log)

        # Save the model
        torch.save(model.state_dict(), f"{tmp_epoch_save_path}/{epoch_log['epoch']}.pth")
    
    best_epoch = max(epoch_stats, key=lambda x: x["f1_valid"])["epoch"]
    # Move model
    best_model_path = f"{save_path}/{logger.run_name}.pth"
    os.rename(f"{tmp_epoch_save_path}/{best_epoch}.pth", best_model_path)
    # Remove tmp folder
    shutil.rmtree(tmp_epoch_save_path)
    # # Load the best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    # torch.save(model.state_dict(), f"{save_path}/{logger.run_name}.pth")

    # Evaluate the model  
    print("Evaluating the model")


    model.eval()
    test_metrics = CumulativeMetrics(classes, posfix="test")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            y_pred = torch.argmax(probs, dim=1).cpu().numpy()
            y_true = torch.argmax(batch["action"], dim=1).cpu().numpy()
            test_metrics.update(y_true, y_pred)

        # metrics = test_metrics.compute()
        # metrics["partition"] = "TEST"
        logger.log_metrics(test_metrics.compute())

        # dataset.index["probs"] = predictions
        # data_pred = pd.merge(data.reset_index(drop=False), dataset.index, on=["index"])
        # data_pred["pred"] = data_pred["probs"].apply(lambda x: np.argmax(x))

    # test_metrics = CumulativeMetrics(classes)
    # for p in ["TRAIN", "VALID", "TEST"]:
    #     data_pred_p = data_pred[data_pred["PARTITION"] == p]
    #     y_pred = data_pred_p["pred"].values
    #     y_true = data_pred_p["action"].astype(int).values

    #     test_metrics.update(y_true, y_pred)
    #     metrics = {"partition": p}
    #     metrics.update(test_metrics.compute())
    #     # metrics.update(compute_metrics(y_true, y_pred))       
    #     logger.log_metrics(metrics)

    logger.close()
    print("FINISHED RUNNING")

if __name__ == "__main__":
    main()