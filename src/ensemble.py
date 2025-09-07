# sources used:
# https://docs.pytorch.org/tutorials/intermediate/ensembling.html
# https://discuss.pytorch.org/t/how-to-ensemble-different-cnn-models-when-use-the-same-dataset/91285
# https://www.geeksforgeeks.org/machine-learning/voting-classifier/
# https://machinelearningmastery.com/voting-ensembles-with-python/

import torch
import torch.nn as nn
import pandas as pd
from preprocessing import apply_transforms
from datasets import CBISDDSMDataset,MIASDataset, load_mias_dataset
from evaluation import evaluate_hard_voting, evaluate_ensemble
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from utils import setup_logging, load_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from collections import Counter
from datetime import datetime
import logging

# Soft voting ensemble as in the feature prototype
class SoftVotingEnsemble(nn.Module):
    """
    Ensemble model using soft voting (average of softmax probabilities).

    Args:
        models (list[nn.Module]): List of pretrained models to ensemble.
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    # forward pass with soft voting
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensemble_probabilities = None

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                
                probabilities = F.softmax(output, dim=1)

                if ensemble_probabilities is None:
                    ensemble_probabilities = probabilities
                else:
                    ensemble_probabilities += probabilities
        
        # Average of probabilities
        ensemble_probabilities /= len(self.models)

        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probabilities + 1e-8)

        return ensemble_logits

class WeightedSoftVotingEnsemble(nn.Module):
    """
    Ensemble model using weighted soft voting.

    Args:
        models (list[nn.Module]): List of pretrained models to ensemble.
        weights (list[float], optional): Weights for each model. Defaults to equal weights.
    """

    def __init__(self, models: list[nn.Module], weights: list[float] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights and models must match in length"
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]  # Normalize to sum = 1

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        ensemble_probabilities = None

        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                output = model(x)
                probabilities = F.softmax(output, dim=1)

                if ensemble_probabilities is None:
                    ensemble_probabilities = weight * probabilities
                else:
                    ensemble_probabilities += weight * probabilities

        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probabilities + 1e-8)
        return ensemble_logits

class HardVotingEnsemble(nn.Module):
    """
    Ensemble model using hard voting (majority class prediction).

    Args:
        models (list[nn.Module]): List of pretrained models to ensemble.
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        all_preds = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)  # Get class predictions
                all_preds.append(preds)

        # Stack predictions
        all_preds = torch.stack(all_preds)

        # Transpose
        all_preds = all_preds.T 

        # Apply hard voting
        voted_preds = []
        for sample_preds in all_preds:
            most_common = Counter(sample_preds.tolist()).most_common(1)[0][0]
            voted_preds.append(most_common)

        return torch.tensor(voted_preds, device=x.device)

def log_metrics(metrics: dict, 
                dataset_name: str, 
                ensemble_method_name: str) -> None:
    """
    Log evaluation metrics for a dataset and ensemble method.

    Args:
        metrics (dict): Dictionary of computed evaluation metrics.
        dataset_name (str): Dataset identifier (e.g., "cbis-ddsm").
        ensemble_method_name (str): Ensemble method identifier (e.g., "softvoting").
    """
    val_acc         = metrics["accuracy"]
    val_precision   = metrics["precision"]
    val_recall      = metrics["recall"]
    val_f1          = metrics["f1"]
    val_roc_auc     = metrics["roc_auc"]
    val_specificity = metrics["specificity"]

    logging.info(f"---- Validation Metrics for {ensemble_method_name} Ensemble on {dataset_name} set:----")
    logging.info(f"Accuracy       : {val_acc}")
    logging.info(f"Precision      : {val_precision}")
    logging.info(f"Recall         : {val_recall}")
    logging.info(f"F1 Score       : {val_f1}")
    logging.info(f"ROC AUC        : {val_roc_auc}")
    logging.info(f"Specificity    : {val_specificity}")

def main() -> None:
    """
    Run evaluation workflow for pretrained CNN ensemble models.
    """

    config = {
        "batch_size": 16, # optimized for 512x512 images
        "num_classes": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "checkpoints_path": "./checkpoints",
        "results_path": "./results",
        "logs_path": "./logs"
    }

    ### Initialise logging ###
    workflow_start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config["logs_path"], f"ensemble_test_{workflow_start_timestamp}.log")
    setup_logging(log_path)

    model_names = ["resnet50", "densenet121", "inception_v3"]
    datasets = ["cbis-ddsm", "mini-mias"]
    ensemble_strategies = ["softvoting", "hardvoting", "weightedsoftvoting"]

    logging.debug(f"Starting ensemble model demo...")

    # CBIS-DDSM reserved test set
    cbis_test_set = pd.read_csv("./data/processed/combined_test_set_mapped.csv")
    # Mini-MIAS external validation dataset
    mias_path = "./data/processed/mias_external_verification_set.csv"

    ### Initialise dataloaders ###
    _, val_transform = apply_transforms("resnet50") # transformations has been unified
    val_mass_dataset = CBISDDSMDataset(cbis_test_set, transform=val_transform)
    val_loader = DataLoader(val_mass_dataset, 
                           batch_size=8,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)
    
    external_dataset = load_mias_dataset(mias_path)
    test_external = MIASDataset(external_dataset,transform=val_transform)
    external_loader = DataLoader(test_external,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True)

    ### Load model checkpoints ###
    models = []
    for model_name in model_names:
        checkpoint_path = os.path.join(config["checkpoints_path"], f"{model_name}_best.pth")
        # Load models
        model = load_model(model_name, checkpoint_path,config["device"], config["num_classes"])
        models.append(model)
    logging.debug(f"Individual model checkpoints loaded...")

    weights = [0.4, 0.3, 0.3]  # Approximate estimation based on model performance

    ### Ensemble integration ###
    for strategy in ensemble_strategies:
        metrics = []
        if strategy == "weightedsoftvoting":
            ensemble = WeightedSoftVotingEnsemble(models, weights)
            ensemble.to(config["device"])
        if strategy == "softvoting":
            ensemble = SoftVotingEnsemble(models)
            ensemble.to(config["device"])
        if strategy == "hardvoting":
            ensemble = HardVotingEnsemble(models)
            ensemble.to(config["device"])

        for set in datasets:
            if strategy == "softvoting" or strategy == "weightedsoftvoting":
                if set == "cbis-ddsm":
                    metrics = evaluate_ensemble(ensemble, val_loader, config["device"])
                else:
                    metrics = evaluate_ensemble(ensemble, external_loader, config["device"])
                
            else:
                if set == "cbis-ddsm":
                    metrics = evaluate_hard_voting(ensemble, val_loader, config["device"])
                else:
                    metrics = evaluate_hard_voting(ensemble, external_loader, config["device"])
            log_metrics(metrics, set, strategy)
                
    logging.info("Workflow ended")

if __name__ == "__main__":
    main()