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
        "epochs": 5,
        "learning_rate": 0.001,
        "num_classes": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "checkpoints_path": "./checkpoints",
        "results_path": "./results_path",
    }

    # Model training main parameters
    Models = ["resnet50", "densenet121", "inception_v3"]
    LR = [1e-4, 1e-3]
    Optimizers = ["adam", "adamw", "sgd"]
    Depth = ["classifier_only", "last_layer", "last_2_layers"]

    best_models = {
        "resnet50": {"learning_rate:": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0},
        "densenet121": {"learning_rate:": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0},
        "inception_v3": {"learning_rate:": 0.0, "optimizer": None, "depth": None, "accuracy": 0.0, "f1_score": 0.0, "AUC": 0.0}
    }

    # Checkpoint and result directories
    os.makedirs(config["checkpoints_path"], exist_ok=True)
    os.makedirs(config["results_path"], exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_mass_df = pd.read_csv("./data/processed/mass_case_description_train_set_mapped.csv")
    val_mass_df = pd.read_csv("./data/processed/mass_case_description_test_set_mapped.csv")

    overall_results = []

    total_combinations = len(Models) * len(LR) * len(Optimizers) * len(Depth)
    combination_count = 0

    # Systematic model training
    for model_name in Models:
        print(f"\n Model training workflow initiated for {model_name}...")
        train_transform, val_transform = apply_transforms(model_name)

        train_mass_dataset = CBISDDSMDataset(train_mass_df, transform=train_transform)
        val_mass_dataset = CBISDDSMDataset(val_mass_df, transform=val_transform)
        print(f"\n Datasets loaded successfully...")

        train_loader = DataLoader(train_mass_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
        
        val_loader = DataLoader(val_mass_dataset, 
                                batch_size=config["batch_size"],
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True)
        
        for lr, optimizer, depth in product(LR, Optimizers, Depth):
            combination_count += 1
            print(f"\n Training progress {combination_count}/{total_combinations}")
            print(f"\n Training parameters: LR:{lr} Optimizer:{optimizer} Depth:{depth}")

            overall_best_f1 = 0.0

            try:
                model = prepare_model(model_name, config["num_classes"], depth)
                result, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_f1 = train_model(model_name,
                                    lr, optimizer, depth, 
                                    model, 
                                    config["device"], 
                                    train_loader, 
                                    val_loader, 
                                    config)
                print(f"\n Training with parameter combination LR: {lr}; Optimizer:{optimizer}; Depth:{depth}")
                overall_results.append(result)

                if best_f1 > overall_best_f1:
                    output_path = os.path.join(config["checkpoints_path"], f"{model_name}_best.pth")
                    overall_best_f1 = best_f1
                    torch.save({
                        "epoch": best_epoch,
                        "model_state_dict": best_model_state_dict,
                        "optimizer_state_dict": best_optimizer_state_dict,
                        "val_f1": best_f1,
                        "config": config
                    },output_path)

                #save best performing model
                
            except Exception as e:
                print(f"Unexpected error raised: {str(e)}")
                continue
        
        # Store results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(config["results_path"], f"training_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2)

        # Training summary
        overall_results.sort(key=lambda x: x["best_val_f1"], reverse=True) # sort by validation accuracy

        print(f"\n Model training completed - Top 5 performer models:")
        for i, result in enumerate(overall_results[:5], 1):
            print(f"\n{i}. {result['model_name']} (LR={result['learning_rate']}, Optimizer={result['optimizer']}, Depth={result['training_depth']})")
            print(f"   Val Acc: {result['best_val_acc']:.4f}")
            print(f"   Val F1:  {result['best_val_f1']:.4f}")
            print(f"   Val AUC: {result['best_val_roc_auc']:.4f}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
