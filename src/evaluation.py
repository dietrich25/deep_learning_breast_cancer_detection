from utils import load_model
from preprocessing import apply_transforms
from datasets import CBISDDSMDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch
import logging
import time
# Model evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,roc_auc_score, confusion_matrix


def evaluate_model_performance(model, dataloader, criterion, device):

    start_time = time.time()
    
    # Set the model into evaluation mode
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    num_batches = len(dataloader)
    logging.debug(f"Starting model performance evaluation with {num_batches} batches...")

    # Run validation without gradient calculation
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Position data to the same device where the model is
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Calculate probabilities for class 1 and predictions
            probs = torch.softmax(outputs, dim=1)[:,1]
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / num_batches

    epoch_accuracy = accuracy_score(all_labels, all_preds)
    epoch_recall = recall_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_roc_auc = roc_auc_score(all_labels, all_probs)

    #specificity 
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    epoch_specificity = tn / (tn+fp) if (tn+fp) != 0 else float("nan")

    logging.info(f"Model validation completed: {epoch_time:.1f}s | Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f} | Acc: {epoch_accuracy:.4f}")
    logging.debug(f"Precision={epoch_precision:.4f} | Recall={epoch_recall:.4f} | Specificity={epoch_specificity:.4f} | ROC-AUC={epoch_roc_auc:.4f}")

    return epoch_accuracy, epoch_loss, epoch_recall, epoch_precision, epoch_f1, epoch_roc_auc, epoch_specificity


def load_and_evaluate_single_model(model_name:str, 
                                   test_df: pd.DataFrame,
                                   config: dict):
    
    _, val_transform = apply_transforms(model_name)
    test_dataset = CBISDDSMDataset(test_df, transform=val_transform)
             
    test_loader = DataLoader(test_dataset,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True)
    checkpoint_path = os.path.join(config["checkpoints"], f"{model_name}_best.pth")
    model = load_model(model_name, checkpoint_path,config["device"], config["num_classes"])

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = evaluate_model_performance(model, 
                                                                                                                test_loader, 
                                                                                                                criterion, config["device"])

    results = {
        "model": model_name,
        "accuracy": val_acc,
        "loss": val_loss,
        "recall": val_recall,
        "precision": val_precision,
        "f1_score": val_f1,
        "roc_auc": val_roc_auc,
        "specificity": val_specificity
    }

    return results