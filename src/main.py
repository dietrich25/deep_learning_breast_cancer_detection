# Sources used:
# https://medium.com/@yennhi95zz/logging-the-effective-management-of-machine-learning-systems-e1eb04e74eb5


import logging
from datasets import CBISDDSMDataset, MIASDataset
from preprocessing import apply_transforms
from train_classification_layer import prepare_model, train_model
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from itertools import product
from datetime import datetime
import json


def main():

    config = {
        "batch_size": 32,
        "epochs": 3,
        "num_classes": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "checkpoints_path": "./checkpoints",
        "results_path": "./results",
        "logs_path": "./logs"
    }

    # Setting up logging
    workflow_start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config["logs_path"], f"training_log_{workflow_start_timestamp}.log")
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers = [logging.FileHandler(log_path), logging.StreamHandler()])
    
    logging.info(f"Logging initialised, log file: {log_path}")

    # Model training main parameters
    Models = ["resnet50", "densenet121", "inception_v3"]
    #LR = [1e-4, 1e-3]
    LR = [1e-3]
    #Optimizers = ["adam", "adamw", "sgd"]
    Optimizers = ["adam"]
    #Depth = ["classifier_only", "last_layer", "last_2_layers"]
    Depth = ["classifier_only", "last_layer"]

    best_models = {
        "resnet50": {"learning_rate:": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0},
        "densenet121": {"learning_rate:": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0},
        "inception_v3": {"learning_rate:": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0}
    }

    # Checkpoint and result directories
    os.makedirs(config["checkpoints_path"], exist_ok=True)
    os.makedirs(config["results_path"], exist_ok=True)

    # Load datasets
    logging.info("Loading datasets...")
    try:
        train_mass_df = pd.read_csv("./data/processed/mass_case_description_train_set_mapped.csv")
        val_mass_df = pd.read_csv("./data/processed/mass_case_description_test_set_mapped.csv")
        logging.info("Datasets successfully loaded.")
    except Exception as e:
        logging.error(f"Dataset loading failed with error {str(e)}")
        return

    overall_results = []
    total_combinations = len(Models) * len(LR) * len(Optimizers) * len(Depth)
    logging.info(f"Training will be initialised for {total_combinations} configurations.")
    combination_count = 0

    training_start = datetime.now()
    logging.info(f"Systematic model training startet at {training_start.strftime("%Y-%m-%d %H:%M:%S")}")

    # Systematic model training
    for model_name in Models:
        model_training_start = datetime.now()
        logging.info(f"Model training workflow initiated for {model_name} family... at {model_training_start.strftime("%Y-%m-%d %H:%M:%S")}")
        
        train_transform, val_transform = apply_transforms(model_name)

        train_mass_dataset = CBISDDSMDataset(train_mass_df, transform=train_transform)
        val_mass_dataset = CBISDDSMDataset(val_mass_df, transform=val_transform)
        logging.debug("Training and validation datasets passed to the model.")

        train_loader = DataLoader(train_mass_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
        logging.debug("Training loader initialised.")
        
        val_loader = DataLoader(val_mass_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True)
        logging.debug("Validation loader initialised.")

        # store best performing model constellation
        model_best_f1 = 0.0
        model_best_result = None
        model_best_states = None
        
        for lr, optimizer, depth in product(LR, Optimizers, Depth):
            combination_count += 1
            logging.info(f"Training progress {combination_count}/{total_combinations}")
            logging.info(f"Training parameters: LR:{lr} Optimizer:{optimizer} Depth:{depth}")

            try:
                model = prepare_model(model_name, config["num_classes"], depth)
                logging.debug(f"Model initialised - {model_name}.")
                logging.info("Training workflow started...")
                result, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_f1 = train_model(model_name=model_name,
                                    learning_rate=lr, optimizer_name=optimizer, training_depth=depth, 
                                    model=model, 
                                    device=config["device"], 
                                    train_dataloader=train_loader, 
                                    validation_dataloader=val_loader, 
                                    config=config)
                logging.info(f"{model_name} training for parameter combination LR: {lr}; Optimizer:{optimizer}; Depth:{depth} finished with F1-score {best_f1}.")
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
                    logging.info(f"New best model for {model_name}: F1={best_f1:.2f}")
            
            except Exception as e:
                logging.error(f"Training failed for {model_name} with params LR:{lr}, Optimizer:{optimizer}, Depth:{depth}. Error: {str(e)}")
                continue

        if model_best_states is not None:
            torch.save({
                "epoch": model_best_states["epoch"],
                "model_state_dict": model_best_states["model_state_dict"],
                "optimizer_state_dict": model_best_states["optimizer_state_dict"],
                "val_f1": model_best_states["f1_score"],
                "config": config
                }, output_path)
            logging.info(f"Best model version for {model_name} saved.")

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

            logging.info(f"Best transfer learning parameters: LR={model_best_result['learning_rate']}, Optimizer={model_best_result['optimizer']}, Depth={model_best_result['training_depth']}.")
        else:
            logging.error(f"{model_name} training failed to save best performing model data.")
           
    # Store all training results
    results_file = os.path.join(config["results_path"], f"training_results_{model_name}.json")
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    # Best models summary
    best_models_file = os.path.join(config["results_path"], f"training_results_best_models.json")
    with open(best_models_file, 'w') as f:
        json.dump(best_models, f, indent=2)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
