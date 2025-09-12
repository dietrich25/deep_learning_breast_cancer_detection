# Sources used:
# https://medium.com/@yennhi95zz/logging-the-effective-management-of-machine-learning-systems-e1eb04e74eb5
# https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452/
# https://stackoverflow.com/questions/60812032/using-weightedrandomsampler-in-pytorch


import logging
import argparse
from datasets import CBISDDSMDataset, MIASDataset, load_cbis_ddsm_split, get_dataset_labels, load_mias_dataset, load_demo_dataset
from preprocessing import apply_transforms
from transfer_learning_trainer import single_phase_training
from evaluation import evaluate_model_performance
from utils import create_dirs, set_random_seeds, setup_logging, save_best_model, load_model
from torch.utils.data import DataLoader
import torch
import os
from itertools import product
from datetime import datetime
import json
import multiprocessing
from torch.utils.data import WeightedRandomSampler
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer learning breast cancer classification")
    parser.add_argument(
        "mode",
        choices=["training", "validation", "demo"],
        help="Select mode: 'training' to train models, 'validation' to test best models, 'demo' to run demo inference."
    )
    return parser.parse_args()

def main(mode):

    workflow_start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        "batch_size": 4,
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

    logging.info("----- Transfer learning ML workflow started -----")
    logging.info(f"Workflow started in {mode} mode.")
    
    ### Model training main parameters ####
    Models = ["resnet50", "densenet121", "inception_v3"] 
    classifier_lr = [1e-3, 1e-4]
    backbone_lr = [1e-4, 1e-5]
    Optimizers = ["adam","adamw","sgd"]
    Depth = [1, 2] 

    ### Result trackers ###
    best_models = {}
    results = []

    ### Checkpoint and result directories ###
    create_dirs(config)

    ### Load datasets ### 
    if mode == "training" or mode == "validation":
        try:
            train_df_path = "./data/processed/combined_training_set_mapped.csv"
            test_df_path = "./data/processed/combined_test_set_mapped.csv"
            mias_path = "./data/processed/mias_external_verification_set.csv"
            train_df, val_df, test_df = load_cbis_ddsm_split(train_df_path,
                                                            test_df_path,
                                                            val_split_ratio=0.25,
                                                            random_state=42)
                
            external_dataset = load_mias_dataset(mias_path)
                
            logging.info("Datasets successfully loaded.")
        except Exception as e:
            logging.error(f"Dataset loading failed with error {str(e)}")
            return
    else:
        demo_df = load_demo_dataset("./data/processed/combined_test_set_mapped.csv", n_samples=5, random_state=35)
        cols = ["patient_id", "left_or_right_breast", "abnormality_type", "pathology"]

        for i, row in demo_df[cols].head(5).iterrows():
            logging.info(
                f"Patient {row['patient_id']} | Side: {row['left_or_right_breast']} | Type: {row['abnormality_type']} | Pathology: {row['pathology']}"
            )

    #### Systematic model training preparation ####
    if mode == "training":
        total_combinations = len(Models) * len(classifier_lr) * len(backbone_lr) * len(Optimizers) * len(Depth)
        combination_count = 0
        training_start = datetime.now()
        logging.info(f"---- Systematic training start | {total_combinations} configurations | {training_start.strftime('%Y-%m-%d %H:%M:%S')} ----")


        #### Systematic model training ####
        for model_name in Models:

            logging.info(f"Initiating systematic model training for {model_name}.")
            model_training_start = datetime.now()
            
            ### Load transformations and datasets ###
            train_transform, val_transform = apply_transforms(model_name)
            train_dataset = CBISDDSMDataset(train_df, transform=train_transform)
            val_dataset = CBISDDSMDataset(val_df, transform=val_transform)

            # get balanced class weights - used only in progressive, multiphase training
            dataset_labels = get_dataset_labels(train_df)
            #class_weights = balance_cbis_ddsm_class_weights(dataset_labels, config["device"])

            targets = np.array(dataset_labels)
            class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
            weights = 1. / class_sample_count
            sample_weights = np.array([weights[t] for t in targets])
            sample_weights = torch.DoubleTensor(sample_weights)
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(sample_weights),
                                            replacement=True)

            train_loader = DataLoader(train_dataset, 
                                    batch_size=config["batch_size"],
                                    sampler=sampler,
                                    #shuffle=True,
                                    num_workers=10,
                                    pin_memory=True)
            
            val_loader = DataLoader(val_dataset, 
                                    batch_size=config["batch_size"],
                                    shuffle=False,
                                    num_workers=10,
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
                logging.info(f"---- Config {combination_count}/{total_combinations} | Model: {model_name} | Classifier LR:{class_lr} | Backbone LR: {back_lr}| Optimizer:{optimizer_name} | Layers unfrozen:{depth} ----")
                
                history, best_model_state, best_metrics = single_phase_training(
                    model_name=model_name,
                    classifier_lr=class_lr,
                    backbone_lr=back_lr,
                    optimizer_name=optimizer_name,
                    training_depth=depth,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config
                )

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
        
    ### Validation of best performing single models on unseen data ###
    if mode == "validation":
        logging.info("Initialising model performance validation on test datasets.")
        validation_results = []

        for model_name in Models:
            # Load model
            checkpoint_path = os.path.join(config["checkpoints"], f"{model_name}_best.pth")
            model = load_model(model_name=model_name,
                               checkpoint=checkpoint_path,
                               device=config["device"],
                               num_classes=config["num_classes"])
            
            # Load transformations and datasets
            _, val_transform = apply_transforms(model_name)
            test_dataset = CBISDDSMDataset(test_df, transform=val_transform)
            test_external = MIASDataset(external_dataset,transform=val_transform)
             
            test_loader = DataLoader(test_dataset,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True)
            external_loader = DataLoader(test_external,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True)
            
            # Loss function
            criterion = torch.nn.CrossEntropyLoss()

            ### Model testing on unseen data - CBIS-DDSM test set ###
            logging.info(f"{model_name} performance evaluation of CBIS-DDSM test dataset.")
            val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = evaluate_model_performance(model, 
                                                                                                                test_loader, 
                                                                                                              criterion, config["device"])
            validation_results.append({"model_name": model_name, "dataset": "cbis_ddsm_test",
                                    "accuracy": val_acc, "loss": val_loss, "recall": val_recall, "precision": val_precision,
                                    "F1": val_f1, "ROC_AUC": val_roc_auc, "specificity": val_specificity})
            
             ### Model testing on unseen data - Mini-MIAS external test set ###
            logging.info(f"{model_name} performance evaluation of MIAS external dataset.")
            val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = evaluate_model_performance(model, 
                                                                                                                external_loader, 
                                                                                                                criterion, config["device"])
            validation_results.append({"model_name": model_name, "dataset": "mias_test", "accuracy": val_acc, 
                                    "loss": val_loss, "recall": val_recall, "precision": val_precision,
                                    "F1": val_f1, "ROC_AUC": val_roc_auc, "specificity": val_specificity})

        # save results
        eval_file = os.path.join(config["results_path"], f"model_evaluation_results_{workflow_start_timestamp}.json")
        with open(eval_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
    
    if mode == "demo":
        for model_name in Models:
            _, val_transform = apply_transforms(model_name)
            demo_dataset = CBISDDSMDataset(demo_df, transform=val_transform)
            demo_loader = DataLoader(demo_dataset, batch_size=4, shuffle=False)

            checkpoint_path = os.path.join(config["checkpoints"], f"{model_name}_best.pth")
            model = load_model(model_name=model_name,
                               checkpoint=checkpoint_path,
                               device=config["device"],
                               num_classes=config["num_classes"])
            
            demo_loader = DataLoader(demo_dataset, batch_size=5, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = evaluate_model_performance(model, 
                                                                                                demo_loader, 
                                                                                                criterion, config["device"])

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    main(args.mode)
