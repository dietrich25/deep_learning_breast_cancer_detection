# sources used:
# https://datascientistsdiary.com/fine-tuning-resnet-50-for-custom-image-classification/
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
# https://www.tutorialexample.com/understand-pytorch-model-named_parameters-with-examples-pytorch-tutorial/
# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
# https://discuss.pytorch.org/t/why-auxiliary-logits-set-to-false-in-train-mode/40705/7


from datasets import CBISDDSMDataset, MIASDataset
from preprocessing import apply_transforms
from torch.utils.data import DataLoader
from torch import optim
import torch 
import torch.nn as nn
import torchvision.models as models
import os
import pandas as pd

# Configuration
config = {
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 0.001,
    "num_classes": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "checkpoints_path": "./checkpoints"
}

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
    correct_predictions = 0
    total_predictions = 0

    num_batches = len(dataloader)

    # Run validation without gradient calculation
    with torch.no_grad():
        for bacth_idx, (inputs, labels) in enumerate(dataloader):
            # Position data to the same device where the model is
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Calc correct predictions
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)
    
    epoch_loss = running_loss / num_batches
    epoch_acc = correct_predictions / total_predictions

    return epoch_acc, epoch_loss

def prepare_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Initialises and prepares a pre-trained model for transfer learning.
    1. Loads the pretrained model
    2. Freezes all layers except the last classifier layer
    3. Replaces classifier for binary classification task

    Args:
        model_name(str): Name of the model ("resnet, "densenet", "inception")
        num_classes(int): Number of output classes

    Returns:
        torch.nn.Module: Prepared model
    """
    if model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        classifier_name = "fc"
    elif model_name == "densenet":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        classifier_name = "classifier"
    elif model_name == "inception":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        classifier_name = "fc"
    else:
        raise ValueError("Invalid model_name. Supported models: 'resnet', 'densenet', 'inception'")
    
    # 1. Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Replace classifier
    classifier = getattr(model, classifier_name)
    in_features = classifier.in_features
    setattr(model, classifier_name, nn.Linear(in_features, num_classes))

    # 3. Unfreeze classifier layer
    classifier = getattr(model, classifier_name)
    for param in classifier.parameters():
        param.requires_grad = True
    
    return model

# code adapted from https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model_name: str,
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
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma = 0.1)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_acc = 0.0

    print(f"\nStarting training {model_name} on {device}.")
    print("---------------------------------------------------")

    for epoch in range(config["epochs"]):
        print(f"\nEpoch progress: {epoch+1}/{config["epochs"]}")
        print("---------------------------------------------------")

        train_loss, train_acc = train_epoch(model, train_dataloader,criterion,optimizer,device)
        print(f"\nTraining loss:  {train_loss:.2f}, Training accuracy: {train_acc:.2f}")

        val_loss, val_acc = validate_epoch(model, validation_dataloader, criterion, device)
        print(f"\nValidation loss:  {val_loss:.2f}, Validation accuracy: {val_acc:.2f}")

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = os.path.join(config["checkpoints_path"], f"{model_name}_best.pth")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config
            },output_path)
            print(f"Model with best accuracy {val_acc:.2f} saved.")
    
    return history

def main():
    """
    Main function to run the pipeline.
    """

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
    

