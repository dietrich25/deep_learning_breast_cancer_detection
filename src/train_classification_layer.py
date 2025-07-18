# sources used:
# https://datascientistsdiary.com/fine-tuning-resnet-50-for-custom-image-classification/
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
# https://www.tutorialexample.com/understand-pytorch-model-named_parameters-with-examples-pytorch-tutorial/
# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
# https://discuss.pytorch.org/t/why-auxiliary-logits-set-to-false-in-train-mode/40705/7
# https://scikit-learn.org/stable/modules/model_evaluation.html
# https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
# https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/


from datasets import CBISDDSMDataset, MIASDataset
from preprocessing import apply_transforms
from torch.utils.data import DataLoader
from torch import optim
import torch 
import torch.nn as nn
import torchvision.models as models
import os
import pandas as pd

# Model evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,roc_auc_score, confusion_matrix

""" # Configuration
config = {
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.001,
    "num_classes": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "checkpoints_path": "./checkpoints",
} """

# code adapted from https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
class Earlystopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        current_loss = val_loss

        if self.best_loss is None:
            self.best_loss = current_loss
        if current_loss < self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0
        return self.early_stop
            
def train_epoch(model, dataloader, criterion, optimizer, device):

    # Set the model into training mode
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    num_batches = len(dataloader)

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
    
    epoch_loss = running_loss / num_batches
    epoch_acc = correct_predictions / total_predictions

    return epoch_acc, epoch_loss
    
def validate_epoch(model, dataloader, criterion, device):
    
    # Set the model into evaluation mode
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []


    num_batches = len(dataloader)

    # Run validation without gradient calculation
    with torch.no_grad():
        for bacth_idx, (inputs, labels) in enumerate(dataloader):
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
    
    epoch_loss = running_loss / num_batches

    epoch_accuracy = accuracy_score(all_labels, all_preds)
    epoch_recall = recall_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_roc_auc = roc_auc_score(all_labels, all_probs)

    #specificity 
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    epoch_specificity = tn / (tn+fp) if (tn+fp) != 0 else float("nan")

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

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        classifier_name = "fc"
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        classifier_name = "classifier"
    elif model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.aux_logits = False
        classifier_name = "fc"
    else:
        raise ValueError("Invalid model_name. Supported models: 'resnet50', 'densenet121', 'inception_v3'")
    
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    classifier = getattr(model, classifier_name)
    in_features = classifier.in_features
    setattr(model, classifier_name, nn.Linear(in_features, num_classes))

    # Unfreeze classifier layer
    classifier = getattr(model, classifier_name)
    for param in classifier.parameters():
        param.requires_grad = True
    
    # Unfreeze model specific layers for training
    if training_depth == "last_layer":
        if model_name == "resnet50":
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif model_name == "densenet121":
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True
        elif model_name == "inception_v3":
            for param in model.Mixed_7c.parameters():
                param.requires_grad = True

    elif training_depth == "last_2_layers":
        if model_name == "resnet50":
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif model_name == "densenet121":
            for param in model.features.denseblock3.parameters():
                param.requires_grad = True
            for param in model.features.transition3.parameters():
                param.requires_grad = True
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True
        elif model_name == "inception_v3":
            for param in model.Mixed_7b.parameters():
                param.requires_grad = True
            for param in model.Mixed_7c.parameters():
                param.requires_grad = True
    else:
        if training_depth !="classifier_only":
            raise ValueError(f"Incoreect parameter passed for model training depth: {training_depth}")

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
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), learning_rate, momentum = 0.9) 
    else:
        raise ValueError(f"Invalid optimizer selected: {optimizer_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stop = Earlystopping(patience=5, delta=0.001)

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


    print(f"\nStarting training {model_name} on {device}.")
    print("---------------------------------------------------")

    for epoch in range(config["epochs"]):
        print(f"\nEpoch progress: {epoch+1}/{config["epochs"]}")
        print("---------------------------------------------------")

        train_loss, train_acc = train_epoch(model, train_dataloader,criterion,optimizer,device)
        print(f"\nTraining loss:  {train_loss:.2f}, Training accuracy: {train_acc:.2f}")

        val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = validate_epoch(model, validation_dataloader, criterion, device)
        print(f"\nValidation loss:  {val_loss:.2f}, Validation accuracy: {val_acc:.2f}, Validation recall: {val_recall:.2f}, Validation precision: {val_precision:.2f}, Validation F1-score: {val_f1:.2f}, Validation ROC-AUC: {val_roc_auc:.2f}, Validation Specificity: {val_specificity:.2f}")

        scheduler.step(1.0 - val_f1)

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

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state_dict = model.state_dict()
            best_optimizer_state_dict = optimizer.state_dict()
            best_epoch = epoch
            #output_path = os.path.join(config["checkpoints_path"], f"{model_name}_best.pth")

            """ torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config
            },output_path)
            print(f"Model with best accuracy {val_acc:.2f} saved.") """
        
        if early_stop(val_acc):
            print(f"Early stopping forced at epoch {epoch + 1}")
            return history, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_val_f1
    
    return history, best_model_state_dict, best_optimizer_state_dict, best_epoch, best_val_f1

""" def main():

    model_names = ["resnet", "densenet", "inception"]

    for model_name in model_names:
        print(f"\n{"----------------------------------"}")
        print(f"Training {model_name} starting...")
        print(f"\n{"----------------------------------"}")

        train_transform, val_transform = apply_transforms(model_name)

        train_mass_df = pd.read_csv("./data/processed/mass_case_description_train_set_mapped.csv")
        val_mass_df = pd.read_csv("./data/processed/mass_case_description_test_set_mapped.csv")

        train_mass_dataset = CBISDDSMDataset(train_mass_df, transform=train_transform)
        val_mass_dataset = CBISDDSMDataset(val_mass_df, transform=val_transform)

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
        
        model = prepare_model(model_name, config["num_classes"])

        history = train_model(model_name, model, config["device"],
                              train_loader, val_loader, config)
        
        torch.save(history, f"{model_name}_initial_training_history.pth")

if __name__ == "__main__":
    main()
    
 """
