# Sources used:
# https://medium.com/@yennhi95zz/logging-the-effective-management-of-machine-learning-systems-e1eb04e74eb5

import argparse
import logging
from datasets import CBISDDSMDataset, MIASDataset, load_cbis_ddsm_split, balance_cbis_ddsm_class_weights, get_dataset_labels
from preprocessing import apply_transforms
from transfer_learning_trainer import progressive_model_training
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from itertools import product
from datetime import datetime
import json
import random
import numpy as np
import multiprocessing
from torch import optim

def main():
    workflow_start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        "batch_size": 32,
        "epochs": 50,
        "num_classes": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "checkpoints": "./checkpoints",
        "results_path": "./results",
        "logs_path": "./logs"
    }

    ### Setting up logging ###
    log_path = os.path.join(config["logs_path"], f"training_log_{workflow_start_timestamp}.log")
    setup_logging(log_path)

    ### Set random seeds for reproducibility
    set_random_seeds(42)

    logging.info("-----Transfer learning ML workflow started-----")
    
    ### Model training main parameters ####
    Models = ["resnet50", "densenet121", "inception_v3"] 
    classifier_lr = [1e-4]
    backbone_lr = [1e-6]
    Optimizers = ["adam", "adamw", "sgd"]
    Depth = [0] 

    ### Result trackers ###
    best_models = {}
    results = []

    ### Checkpoint and result directories ###
    create_dirs(config)

    ### Load datasets ### 
    try:
        train_df_path = "./data/processed/combined_training_set_mapped.csv"
        val_df_path = "./data/processed/combined_test_set_mapped.csv"
        train_df, val_df, test_df = load_cbis_ddsm_split(train_df_path,
                                                                    val_df_path,
                                                                    val_split_ratio=0.25,
                                                                    random_state=42)
        logging.info("Datasets successfully loaded.")
    except Exception as e:
        logging.error(f"Dataset loading failed with error {str(e)}")
        return

    #### Systematic model training preparation ####

    total_combinations = len(Models) * len(classifier_lr) * len(backbone_lr) * len(Optimizers) * len(Depth)
    combination_count = 0
    training_start = datetime.now()
    logging.info(f"----Systematic training start | {total_combinations} configurations | {training_start.strftime('%Y-%m-%d %H:%M:%S')}----")


    #### Systematic model training ####
    for model_name in Models:

        logging.info(f"Initiating systematic model training for {model_name}.")
        model_training_start = datetime.now()
        
        ### Load transformations and datasets ###
        train_transform, val_transform = apply_transforms(model_name)
        train_dataset = CBISDDSMDataset(train_df, transform=train_transform)
        val_dataset = CBISDDSMDataset(val_df, transform=val_transform)

        # get balanced class weights
        dataset_labels = get_dataset_labels(train_df)
        class_weights = balance_cbis_ddsm_class_weights(dataset_labels, config["device"])

        train_loader = DataLoader(train_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=True,
                                num_workers=12,
                                pin_memory=True)
        
        val_loader = DataLoader(val_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=False,
                                num_workers=12,
                                pin_memory=True)

        ### Store best performing model constellation ###
        model_best_f1 = 0.0
        model_best_config = None
        model_best_history= None
        model_best_states = None
        model_best_metrics = {}

        ### Execute training constellations ###
        for class_lr, back_lr, optimizer_name, depth in product(classifier_lr, backbone_lr, Optimizers, Depth):
           
            combination_count += 1
            logging.info(f"**** Config {combination_count}/{total_combinations} | Classifier LR:{class_lr} | Backbone LR: {back_lr}| Optimizer:{optimizer_name} | Layers unfrozen:{depth} ****")

            history, best_model_state, best_metrics = progressive_model_training(model_name=model_name,
                    classifier_lr=class_lr,
                    backbone_lr=back_lr,
                    optimizer_name=optimizer_name,
                    training_depth=depth,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    class_weigths=class_weights)

            results.append({"model_name": model_name,
                    "classifier_lr": class_lr,
                    "backbone_lr": back_lr,
                    "optimizer": optimizer_name,
                    "depth": depth,
                    "history": history,
                    "best_metrics": best_metrics})

            # Update best for this model
            if best_metrics["f1"] > model_best_f1:
                model_best_f1 = best_metrics["f1"]
                model_best_config = {
                    "classifier_lr": class_lr,
                    "backbone_lr": back_lr,
                    "optimizer": optimizer_name,
                    "depth": depth
                }
                model_best_states = best_model_state
                model_best_history = history
                logging.info(f"\nNew best configuration for {model_name} - F1: {model_best_f1:.4f}")
                
                save_best_model(model_name=model_name,
                    model_states=model_best_states,
                    metrics=model_best_metrics,
                    model_best_config=model_best_config,
                    config=config)

            model_time = datetime.now() - model_training_start
            logging.info(f"{model_name} training completed | Total time: {model_time.total_seconds()/60:.1f} minutes.")
    
        best_models[model_name] = {
            "config": model_best_config,
            "metrics": {"f1": model_best_f1},
            "history": model_best_history
        }
        
    # Store all training results
    results_file = os.path.join(config["results_path"], f"training_results_{workflow_start_timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Best models summary
    best_models_file = os.path.join(config["results_path"], f"training_results_best_models.json")
    with open(best_models_file, 'w') as f:
        json.dump(best_models, f, indent=2)

def set_random_seeds(seed):
    """ Set random seed for used components to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_logging(path):
    """Set up logging for the project (on debug level)."""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers = [logging.FileHandler(path), logging.StreamHandler()])
    logging.info(f"Log file: {path}")

def create_dirs(config):
    """Create directories for checkpoints, results and logs."""
    os.makedirs(config["checkpoints"], exist_ok=True)
    os.makedirs(config["results_path"], exist_ok=True)
    os.makedirs(config["logs_path"], exist_ok=True)

def save_best_model(model_name:str, model_states:dict, metrics:dict, model_best_config:dict, config:dict):
    """Saves model checkpoint."""

    output_path = os.path.join(config["checkpoints"], f"{model_name}_best.pth")

    torch.save({
        "model_name": model_name,
        "model_state_dict": model_states,
        "metrics": metrics,
        "model_best_config": model_best_config,
        "config": config
    }, output_path)

    logging.info(f"Best model version for {model_name} saved.")
    
if __name__ == "__main__":
    multiprocessing.freeze_support()

    main()
