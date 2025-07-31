# sources used:
# https://docs.pytorch.org/tutorials/intermediate/ensembling.html
# https://discuss.pytorch.org/t/how-to-ensemble-different-cnn-models-when-use-the-same-dataset/91285
# https://www.geeksforgeeks.org/machine-learning/voting-classifier/

import torch
import torch.nn as nn
import pandas as pd
from torchvision import models
from preprocessing import apply_transforms
from datasets import CBISDDSMDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from evaluation import evaluate_model_performance
from utils import setup_logging


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DemoEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    # forward pass with soft voting
    def forward(self, x):
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

def load_model(model_name, checkpoint, device, num_classes):
        if model_name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        if model_name == "densenet121":
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        if model_name == "inception_v3":
            model = models.inception_v3(weights=None)
            model.aux_logits = False
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    
        return model

def evaluate_ensemble(model, dataloader, device):

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss= 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    eval_loss = total_loss/len(dataloader)
    eval_acc = correct_predictions/total_predictions

    return eval_loss, eval_acc


        
def main():

    config = {
        "batch_size": 32,
        "num_classes": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "checkpoints_path": "./checkpoints",
        "logs_path": "./logs"
    }

    log_path = os.path.join(config["logs_path"], f"ensemble_test.log")
    setup_logging(log_path)

    model_names = ["resnet50", "densenet121", "inception_v3"]

    results = {
        "model_name": [], "LR": [], "training_depth": [], "optimizer": [],
        "train_loss": [], "train_acc": [],
        "val_acc": [], "val_loss": [], "val_recall": [], "val_precision": [],
        "val_f1": [], "val_roc_auc": [], "val_specificity": []
    }

    print(f"Starting ensemble model demo...")
    # validation dataset
    val_mass_df = pd.read_csv("./data/processed/mass_case_description_test_set_mapped.csv")

    _, val_transform = apply_transforms("resnet50") # for testing
    val_mass_dataset = CBISDDSMDataset(val_mass_df, transform=val_transform)
    val_loader = DataLoader(val_mass_dataset, 
                           batch_size=32,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)

    models = []

    for model_name in model_names:
        checkpoint_path = os.path.join(config["checkpoints_path"], f"{model_name}_best.pth")
        # Load models
        model = load_model(model_name, checkpoint_path,config["device"], config["num_classes"])
        models.append(model)
    print(f"\nIndividual model checkpoints loaded...")

    # Ensemble setup
    ensemble = DemoEnsemble(models)
    print(f"\nEnsemble model set...")
    ensemble.to(config["device"])

    criterion = torch.nn.CrossEntropyLoss()

    print("\nTesting ensemble on validation set...")
    val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = evaluate_model_performance(ensemble, val_loader, criterion, config["device"])


    results["model_name"].append("ensemble")
    results["val_acc"].append(val_acc)
    results["val_loss"].append(val_loss)
    results["val_recall"].append(val_recall)
    results["val_precision"].append(val_precision)
    results["val_f1"].append(val_f1)
    results["val_roc_auc"].append(val_roc_auc)
    results["val_specificity"].append(val_specificity)
    
    torch.save(results, 'ensemble_evaluation_results.pth')
    print(f"\nResults saved to 'ensemble_evaluation_results.pth'")

if __name__ == "__main__":
    main()