# Sources used:
# https://medium.com/@yennhi95zz/logging-the-effective-management-of-machine-learning-systems-e1eb04e74eb5

import argparse
import logging
from datasets import CBISDDSMDataset, MIASDataset, load_cbis_ddsm_split
from preprocessing import apply_transforms
from transfer_learning_trainer import prepare_model, train_model, load_model, unfreeze_layer, model_training_2_phase, freeze_model_params
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from itertools import product
from datetime import datetime
import json
import multiprocessing

def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    train_classifier = True
    train_backbone = True
    run_validation = False

    config = {
        "batch_size": 32,
        "epochs": 10,
        "num_classes": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "checkpoints_path": "./checkpoints",
        "results_path": "./results",
        "logs_path": "./logs"
    }

    #### Setting up logging ####
    workflow_start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config["logs_path"], f"training_log_{workflow_start_timestamp}.log")
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers = [logging.FileHandler(log_path), logging.StreamHandler()])
    logging.info(f"Log file: {log_path}")

    logging.info("Transfer learning ML workflow started")
    logging.info(f"Train Classifier: {train_classifier}")
    logging.info(f"Train Backbone: {train_backbone}")
    logging.info(f"Run validation: {run_validation}")
    
    ### Model training main parameters ####
    Models = ["resnet50"] # , "densenet121", "inception_v3"
    classifier_lr = [1e-3] #, 1e-4]
    backbone_lr = [1e-5] #, 1e-6]
    Optimizers = ["adam"] #, "adamw", "sgd
    Depth = [1] #, 2] # Depth = ["classifier_only", "last_layer", "last_2_layers"]

    best_models = {
        "resnet50": {"classifier_lr": 0.0, "backbone_lr": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0},
        "densenet121": {"classifier_lr": 0.0, "backbone_lr": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0},
        "inception_v3": {"classifier_lr": 0.0, "backbone_lr": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0}
    }

    #### Checkpoint and result directories ####
    os.makedirs(config["checkpoints_path"], exist_ok=True)
    os.makedirs(config["results_path"], exist_ok=True)
    os.makedirs(config["logs_path"], exist_ok=True)

    #### Load datasets ####   --> redo splits to match midterm feedback 60:20:20
    logging.info("Loading datasets...")
    try:
        train_df_path = "./data/processed/combined_training_set_mapped.csv"
        val_df_path = "./data/processed/combined_test_set_mapped.csv"
        train_dataset, val_dataset, test_dataset = load_cbis_ddsm_split(train_df_path,
                                                                    val_df_path,
                                                                    val_split_ratio=0.25,
                                                                    random_state=42)
        logging.info("Datasets successfully loaded.")
    except Exception as e:
        logging.error(f"Dataset loading failed with error {str(e)}")
        return

    #### Systematic model training preparation ####
    overall_results = []
    classifier_results = []
    total_combinations = len(Models) * len(classifier_lr) * len(backbone_lr) * len(Optimizers) * len(Depth)
    combination_count = 0
    training_start = datetime.now()
    logging.info(f"Systematic training start | {total_combinations} configurations | {training_start.strftime('%Y-%m-%d %H:%M:%S')}***")

    #### Systematic model training ####
    for model_name in Models:
        model_training_start = datetime.now()
        logging.info(f"Model: {model_name}")
        
        ### Load transformations and datasets ###
        train_transform, val_transform = apply_transforms(model_name)
        train_dataset = CBISDDSMDataset(train_dataset, transform=train_transform)
        val_dataset = CBISDDSMDataset(val_dataset, transform=val_transform)

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
        model_best_result = None
        model_best_states = None
        classifier_best_f1 = 0.0
        classifier_best_states = None

        
        ### Execute training constellations ###
        for class_lr, back_lr, optimizer, depth in product(classifier_lr, backbone_lr, Optimizers, Depth):
            combination_count += 1
            logging.info(f"Config {combination_count}/{total_combinations} | Classifier LR:{class_lr} | Backbone LR: {back_lr}| Optimizer:{optimizer} | Layers unfrozen:{depth}")

            ### Phase 1 - Replace and train classifier layer ####
            logging.info("Starting classifier block training")

            model = prepare_model(model_name, config["num_classes"], "classifier_only")
            result, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_f1 = model_training_2_phase(model= model, 
                model_name=model_name,
                optimizer_name=optimizer, 
                device=config["device"], 
                classifier_only=train_classifier,
                training_depth=depth, 
                learning_rate=class_lr,                                   
                train_dataloader=train_loader, 
                validation_dataloader=val_loader, 
                config=config)
                    
            logging.info(f"Classifier training results | F1:{best_f1:.4f} | Epoch:{best_epoch}")
            overall_results.append(result)
            checkpoint_path = os.path.join(config["checkpoints_path"], f"{model_name}_classifier_{combination_count}.pth")
            
            if best_f1 > classifier_best_f1:
                output_path = os.path.join(config["checkpoints_path"], f"{model_name}_best.pth")
                classifier_best_f1 = best_f1
                classifier_best_states = {
                    "model_state_dict": best_model_state_dict,
                    "optimizer_state_dict": best_optimizer_state_dict,
                    "epoch": best_epoch,
                    "f1_score": best_f1
            }

            torch.save({
                    "epoch": classifier_best_states["epoch"],
                    "model_state_dict": classifier_best_states["model_state_dict"],
                    "optimizer_state_dict": classifier_best_states["optimizer_state_dict"],
                    "val_f1": classifier_best_states["f1_score"],
                    "config": config
                    }, checkpoint_path)
            logging.info(f"Best classifier model for {model_name} saved.")

            ### Phase 2 - Load checkpoint and train backbone training ###

            # Load previously saved checkpoint
            model = load_model(model_name, 
                                checkpoint_path,
                                config["device"], 
                                config["num_classes"])
            
            # freeze modle parameters
            model = freeze_model_params(model)
            # unfreeze layer
            model = unfreeze_layer(model, model_name, depth)

            # Train backbone
            result, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_f1 = model_training_2_phase(model= model, 
                model_name=model_name,
                optimizer_name=optimizer, 
                device=config["device"], 
                classifier_only=False,
                training_depth=depth, 
                learning_rate=back_lr,                                   
                train_dataloader=train_loader, 
                validation_dataloader=val_loader, 
                config=config)
            logging.info(f"Result | F1:{best_f1:.4f} | Epoch:{best_epoch}")
            overall_results.append(result)
                
            if best_f1 > model_best_f1:
                output_path = os.path.join(config["checkpoints_path"], f"{model_name}_best.pth")
                model_best_f1 = best_f1
                model_best_result = result
                model_best_states = {
                    "model_state_dict": best_model_state_dict,
                    "optimizer_state_dict": best_optimizer_state_dict,
                    "epoch": best_epoch,
                    "f1_score": best_f1
            }
            logging.info(f"New best for {model_name} during backbone training: F1={best_f1:.4f}")

            if model_best_states is not None:
                torch.save({
                    "epoch": model_best_states["epoch"],
                    "model_state_dict": model_best_states["model_state_dict"],
                    "optimizer_state_dict": model_best_states["optimizer_state_dict"],
                    "val_f1": model_best_states["f1_score"],
                    "config": config
                    }, output_path)
                logging.info(f"Best model for {model_name} saved.")
            else:
                logging.error(f"{model_name} training failed to save best performing model data.")

            best_models[model_name].update({
                "learning_rate": model_best_result["LR"],
                "optimizer": model_best_result["optimizer"],
                "depth": model_best_result["training_depth"],
                "train_loss": model_best_result["train_loss"],
                "train_acc": model_best_result["train_acc"],
                "val_acc": model_best_result["val_acc"],
                "val_loss": model_best_result["val_loss"],
                "val_recall": model_best_result["val_recall"],
                "val_precision": model_best_result["val_precision"],
                "f1_score": model_best_result["val_f1"],
                "AUC": model_best_result["val_roc_auc"],
                "val_specificity": model_best_result["val_specificity"],
                "epoch": model_best_states["epoch"]
            })
        
        #for i in range(3):
         #   logging.info(f"Best transfer learning parameters: model={model_best_result['model_name'][i]}, LR={model_best_result['LR'][i]}, Optimizer={model_best_result['optimizer'][i]}, Depth={model_best_result['training_depth'][i]}.")
        
        model_time = datetime.now() - model_training_start
        logging.info(f"{model_name} classifier training complete | Total time: {model_time.total_seconds()/60:.1f} minutes.")
    
    # Store all training results
    results_file = os.path.join(config["results_path"], f"training_results_{workflow_start_timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    # Best models summary
    best_models_file = os.path.join(config["results_path"], f"training_results_best_models.json")
    with open(best_models_file, 'w') as f:
        json.dump(best_models, f, indent=2)

    
    # run validation TBD
    

if __name__ == "__main__":
    multiprocessing.freeze_support()

    main()
