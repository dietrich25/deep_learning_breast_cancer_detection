from utils import load_model
from preprocessing import apply_transforms
from datasets import CBISDDSMDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch
import logging
import time
import torch.nn as nn
# Model evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,roc_auc_score, confusion_matrix

def evaluate_model_performance(model:nn.Module, 
                               dataloader:DataLoader, 
                               criterion:nn.Module, 
                               device:torch.device) -> tuple:
    """
    Evaluate a model's performance on a dataset using the given loss criterion.

    Args:
        model (nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        tuple: (accuracy, loss, recall, precision, f1_score, roc_auc, specificity)
    """

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

    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)

    logging.info(f"Model validation completed: {epoch_time:.1f}s | Loss: {epoch_loss:.4f} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")
    logging.debug(f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | Specificity={metrics['specificity']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")

    return (metrics["accuracy"], epoch_loss, metrics["recall"], metrics["precision"], metrics["f1"], metrics["roc_auc"], metrics["specificity"])

def evaluate_hard_voting(model: nn.Module, 
                         dataloader: DataLoader, 
                         device: torch.device) -> dict:
    """
    Evaluate a hard voting ensemble on a dataset.

    Args:
        model (nn.Module): Hard voting ensemble model.
        dataloader (DataLoader): Dataloader for evaluation.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        dict: Dictionary of evaluation metrics
    """

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            preds = model(inputs)  # Already final class predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = compute_classification_metrics(all_labels, all_preds)

    return metrics

def evaluate_ensemble(model:nn.Module, 
                      dataloader: DataLoader, 
                      device: torch.device) -> dict:
    """
    Evaluate a soft or weighted soft voting ensemble.

    Args:
        model (nn.Module): Ensemble model returning logits.
        dataloader (DataLoader): Dataloader for evaluation.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        dict: Dictionary of evaluation metrics
    """

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    eval_loss = total_loss / len(dataloader)

    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = eval_loss

    return metrics

def compute_classification_metrics(labels: list, 
                                   preds: list, 
                                   probs: list = None) -> dict:
    """
    Compute evaluation metrics for classification results.

    Args:
        labels (list): Ground truth class labels.
        preds (list): Predicted class labels.
        probs (list, optional): Predicted probabilities for the positive class. 
                                Required for ROC-AUC calculation.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1 score, 
              ROC-AUC (if probs provided), and specificity.
    """

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds),
        "precision": precision_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

    if probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    except ValueError:
        metrics["specificity"] = float("nan")

    return metrics

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