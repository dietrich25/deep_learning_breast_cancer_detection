# sources used:
# https://datascientistsdiary.com/fine-tuning-resnet-50-for-custom-image-classification/
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
# https://www.tutorialexample.com/understand-pytorch-model-named_parameters-with-examples-pytorch-tutorial/
# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
# https://discuss.pytorch.org/t/why-auxiliary-logits-set-to-false-in-train-mode/40705/7
# https://scikit-learn.org/stable/modules/model_evaluation.html
# https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
# https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/

from torch.utils.data import DataLoader
from torch import optim
import torch 
import torch.nn as nn
import torchvision.models as models
import logging
import time

# Model evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,roc_auc_score, confusion_matrix

# code adapted from https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
class Earlystopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):

        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            logging.debug(f"EarlyStop counter reseted to: {self.counter}")
        else:
            self.counter += 1
            logging.debug(f"EarlyStop counter increased to: {self.counter}")

        return self.counter >= self.patience
            
def train_epoch(model, dataloader, criterion, optimizer, device):

    start_time = time.time()

    # Set the model into training mode
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    num_batches = len(dataloader)
    logging.debug(f"Starting model training epoch with {num_batches} batches...")

    # Iterate through all the batches in the dataset
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Position data to the same device where the model is
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calc correct predictions
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)
    
    epoch_time = time.time() - start_time

    epoch_loss = running_loss / num_batches
    epoch_acc = correct_predictions / total_predictions
    logging.info(f"Training epoch completed: {epoch_time:.1f}s | Training Loss:{epoch_loss:.4f} | Training Accuracy={epoch_acc:.4f}")

    return epoch_acc, epoch_loss
    
def validate_epoch(model, dataloader, criterion, device):

    start_time = time.time()
    
    # Set the model into evaluation mode
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    num_batches = len(dataloader)
    logging.debug(f"Starting model validation epoch with {num_batches} batches...")

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

    logging.info(f"Validation epoch completed: {epoch_time:.1f}s | Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f} | Acc: {epoch_accuracy:.4f}")
    logging.debug(f"Precision={epoch_precision:.4f} | Recall={epoch_recall:.4f} | Specificity={epoch_specificity:.4f} | ROC-AUC={epoch_roc_auc:.4f}")

    return epoch_accuracy, epoch_loss, epoch_recall, epoch_precision, epoch_f1, epoch_roc_auc, epoch_specificity

def prepare_model(model_name: str, num_classes: int, training_depth: str) -> torch.nn.Module:
    """
    Prepare a pre-trained CNN model for transfer learning with a specific training depth

    Args:
        model_name(str): Name of the model ("resnet, "densenet", "inception")
        num_classes(int): Number of output classes
        training_depth(str): Parameter unfreeze strategy ("classifier_only", "last_layer", "last_2_layers")
        
    Returns:
        torch.nn.Module: Prepared model
    """

    logging.debug(f"Preparing {model_name} with {num_classes} classes and {training_depth} depth.")

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        classifier_name = "fc"
        logging.debug("ResNet50 model loaded with ImageNet weights")
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        classifier_name = "classifier"
        logging.debug("DenseNet121 model loaded with ImageNet weights")
    elif model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.aux_logits = False
        classifier_name = "fc"
        logging.debug("Inception_V3 model loaded with ImageNet weights")
    else:
        logging.error(f"Invalid model_name parameter: {model_name}. Exception raised...")
        raise ValueError("Invalid model_name. Supported models: 'resnet50', 'densenet121', 'inception_v3'")
    
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    logging.debug("All parameters frozen...")

    # Replace classifier
    classifier = getattr(model, classifier_name)
    in_features = classifier.in_features
    setattr(model, classifier_name, nn.Linear(in_features, num_classes))
    logging.debug(f"Classifier replaced from {in_features} to {num_classes} features.")

    # Unfreeze classifier layer
    classifier = getattr(model, classifier_name)
    for param in classifier.parameters():
        param.requires_grad = True
    logging.debug("Classifier layer unfrozen...")
    
    # Unfreeze model specific layers for training
    if training_depth == "last_layer":
        if model_name == "resnet50":
            for param in model.layer4.parameters():
                param.requires_grad = True
            logging.debug("Resnet50 layer4 unfrozen for training...")
        elif model_name == "densenet121":
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True
            logging.debug("Densenet121 denseblock4 and norm5 unfrozen for training...")   
        elif model_name == "inception_v3":
            for param in model.Mixed_7c.parameters():
                param.requires_grad = True
            logging.debug("Inception_V3 mixed_7c unfrozen for training...")

    elif training_depth == "last_2_layers":
        if model_name == "resnet50":
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            logging.debug("Resnet50 layer4 and layer3 unfrozen for training...")
        elif model_name == "densenet121":
            for param in model.features.denseblock3.parameters():
                param.requires_grad = True
            for param in model.features.transition3.parameters():
                param.requires_grad = True
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True
            logging.debug("DenseNet121 denseblock3, transition3, denseblock4, and norm5 unfrozen for training...")
        elif model_name == "inception_v3":
            for param in model.Mixed_7b.parameters():
                param.requires_grad = True
            for param in model.Mixed_7c.parameters():
                param.requires_grad = True
            logging.debug("Inception_V3 mixed_7b and mixed_7c unfrozen for training...")
    else:
        if training_depth !="classifier_only":
            logging.error(f"Incorrect parameter passed for model training depth: {training_depth}")
            raise ValueError(f"Incorrect parameter passed for model training depth: {training_depth}")
        
    logging.debug(f"Model {model_name} prepared for training...")
    return model

# code adapted from https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model_name: str,
                learning_rate: int,
                optimizer_name: str,
                training_depth: str,
                model: torch.nn.Module,
                device: torch.device,
                train_dataloader: DataLoader,
                validation_dataloader: DataLoader,
                config: dict):
    """
    Trains a Pytorch model using a specific training and validation data loader.

    Args:
        model_name(str): Name of the model used ("resnet", "densenet", "inception")
        model(torch.nn.Module): Pytorch model to train
        train_loader(torch.utils.data.Dataloader): dataloader for the training dataset
        validation_dataloader(torch.utils.data.Dataloader): dataloader for the validation dataset
        config(dict): configuration dictionary that contains device, learning rate, number of training epochs and filepath to save mode checkpoints.
    
    Returns:
        history(dict): Dictionary containing the training and validation losses and accuracies. 
    """

    total_start_time = time.time()
    logging.debug(f"**Initialising training for {model_name} with LR={learning_rate}, optimizer={optimizer_name}, depth={training_depth}.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-4)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), learning_rate, momentum = 0.9) 
    else:
        logging.error(f"Invalid optimizer parameter:{optimizer_name}.")
        raise ValueError(f"Invalid optimizer selected: {optimizer_name}")
    logging.debug(f"Optimizer {optimizer_name} initialised with {learning_rate} learning rate.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stop = Earlystopping(patience=10, delta=0.01)
    logging.debug("Learning rate scheduler and early stopping initialised.")

    history = {
        "model_name": [], "LR": [], "training_depth": [], "optimizer": [],
        "train_loss": [], "train_acc": [],
        "val_acc": [], "val_loss": [], "val_recall": [], "val_precision": [],
        "val_f1": [], "val_roc_auc": [], "val_specificity": []
    }

    best_val_f1 = 0.0 
    best_model_state_dict = None
    best_optimizer_state_dict = None
    best_epoch = 0

    logging.debug(f"Starting training {model_name} on {device} for {config['epochs']} epochs.")

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        logging.debug(f"Epoch progress: {epoch+1}/{config['epochs']}")

        train_acc, train_loss = train_epoch(model, train_dataloader,criterion,optimizer,device)

        val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = validate_epoch(model, validation_dataloader, criterion, device)

        epoch_time = time.time() - epoch_start_time

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(1.0 - val_f1)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            logging.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")

        history["model_name"].append(model_name)
        history["LR"].append(learning_rate)
        history["training_depth"].append(training_depth)
        history["optimizer"].append(optimizer_name)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(val_recall)
        history["val_precision"].append(val_precision)
        history["val_f1"].append(val_f1)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_specificity"].append(val_specificity)

        logging.info(f"Epoch {epoch+1}/{config['epochs']} completed in {epoch_time:.1f}s | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state_dict = model.state_dict()
            best_optimizer_state_dict = optimizer.state_dict()
            best_epoch = epoch
            logging.debug(f"**New best model saved with F1-score: {val_f1:.4f}.**")
        
        if early_stop(val_loss):
            logging.warning(f"Early stopping triggered at epoch {epoch + 1}.")
            return history, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_val_f1
    
    total_time = time.time() - total_start_time
    logging.info(f"**Training completed. Total time: {total_time/60:.1f} minutes | Best F1-score {val_f1:.4f}")
    return history, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_val_f1

