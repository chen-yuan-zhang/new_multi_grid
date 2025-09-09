import os
import json
import yaml
import wandb
import logging
import numpy as np

from csv import writer
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
# from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, cfg, show_keys=None) -> None:

        self.columns = None
        self.metrics = None
        self.show_keys = show_keys 
        # Time run starts
        self.start_time = datetime.now()   
        # Set the run name
        
        # Config parameters
        self.run_name = cfg["experiment"] + "-" + str(self.start_time).replace(" ", "_")
        logs_path = cfg["logger"]["folder_path"]
        self.log_name = os.path.join(logs_path, self.run_name)    
       
        # Create log folder
        if not os.path.exists(self.log_name): os.makedirs(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

        # Set Files
        ## Episode File
        self.epoch_file_path = os.path.join(self.log_name, 'epochs.csv')  
        
        ## Metrics File
        self.metrics_file_path = os.path.join(self.log_name, 'metrics.csv')
        
    def log_epoch(self, data):

        # Create the file if it does not exist
        if self.columns is None:
            self.columns = list(data.keys())
            with open(self.epoch_file_path, 'w') as file:
                epoch_log = writer(file)
                epoch_log.writerow(
                    self.columns
                )

        time = str(datetime.now()-self.start_time)        
            
        line = " ".join([f"{c}: {data[c]:5f} |" for c in self.columns 
                         if self.show_keys is not None and c in self.show_keys])
        logging.info(line)
            
        with open(self.epoch_file_path, 'a') as file:
            epoch_log = writer(file) 
            epoch_log.writerow(
                [data[c] for c in self.columns]
            )

    def log_metrics(self, data):

        if self.metrics is None:
            self.metrics = list(data.keys())
            with open(self.metrics_file_path, 'w') as file:
                metrics_log = writer(file)
                metrics_log.writerow(
                    self.metrics
                )

        line = " ".join([f"{m}: {data[m]} |" for m in self.metrics])
        logging.info(line)

        with open(self.metrics_file_path, 'a') as file:
            metrics_log = writer(file) 
            metrics_log.writerow(
                [data[m] for m in self.metrics]
            )
        
    def close(self):
        pass
        


class TensorboardLogger(Logger):
    def __init__(self, cfg, run_name):
        super().__init__(cfg, run_name)

        self.tf_writer = None
        self.writer = SummaryWriter(self.log_name)
    
    def log_epoch(self, step, rewards, actor_loss, critic_loss, entropy, epsilon):

        if not self.enable_log_step:
            return

        if actor_loss:
            self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="epsilon",scalar_value=epsilon, global_step=step)
        self.writer.add_scalar(tag="step_rewards",scalar_value=rewards, global_step=step)


class WanDBLogger(Logger):

    def __init__(self, cfg, show_keys):

        super().__init__(cfg, show_keys)

        project = cfg["logger"]["wandb_project"]
        # start a new wandb run to track this script
        wandb.init(
            project=project,
            name=self.run_name,
            # track hyperparameters and run metadata
            config=cfg
        )      
                
    def log_epoch(self, data):
        super().log_epoch(data)
        wandb.log(data) 

    def log_metrics(self, data):
        super().log_metrics(data)

        # formatted_data = {}
        # for key, value in data.items():
        #     if key == "partition":
        #         continue
        #     nkey = data["partition"] + "_" + key
        #     formatted_data[nkey] = value

        wandb.log(data)

    def close(self):

        super().close()
        wandb.finish()