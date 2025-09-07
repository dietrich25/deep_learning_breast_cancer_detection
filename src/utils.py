import random
import numpy as np
import pandas as pd
import torch
import os
import logging
import torch.nn as nn
import torchvision.models as models
from torch import optim

### General support functions ###
def set_random_seeds(seed):
    """ Set random seed for used components to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_logging(path):
    """Set up logging for the project/module (on debug level)."""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers = [logging.FileHandler(path), logging.StreamHandler()])
    logging.info(f"Logging initialised, log file: {path}")

def create_dirs(config):
    """Create directories for checkpoints, results and logs."""
    os.makedirs(config["checkpoints"], exist_ok=True)
    os.makedirs(config["results_path"], exist_ok=True)
    os.makedirs(config["logs_path"], exist_ok=True)

### Dataset related functions ###
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column naming convention across dataframes and to use underscores in column names instead of spaces.

    Parameters:
        df(pd.DataFrame): Input dataframe that might contain column names with non-standard value.
    
    Returns:
        pd.DataFrame: Copy of the input dataframe with standardised column names.
    
    """
    df = df.copy()
    new_colum_names = {col: col.replace(" ", "_") for col in df.columns}
    df = df.rename(columns=new_colum_names)
    
    return df

def create_case_id(file_path:str) -> str:
    """
    Extract the case ID from the file path (assuming the path structure uses '/' and the cse ID is the first directory in the path).

    Parameters:
        file_path(str): File path string.
    
    Returns:
        str: Extracted case ID.
    """
    return os.path.dirname(file_path).split("/")[0]

def get_largest_dcm_file(case_id: str, root_dir:str) -> str:
    """
    Return the file path to the largest Dicom file (full mammogram image) for a given case ID.

    Parameters:
        case_id (str): Case ID used as an unique identifier for a patient case.
        root_dir (str): Directiry that contains all CBIS-DDSM cases.
    
    Returns:
        largest_file (str): Full image file path to the largest .dcm file connected to a case, or None if no file is found.
    """

    case_folder = os.path.join(root_dir, case_id)
    largest_file = None
    largest_size = -1
    
    for dirpath, dirnames, filenames in os.walk(case_folder):
        for filename in filenames:
            if filename.lower().endswith(".dcm"):
                filepath = os.path.join(dirpath, filename)
                size = os.path.getsize(filepath)
                if size > largest_size:
                    largest_size = size
                    largest_file = filepath
                
    return largest_file.replace("\\", "/")

def prepare_for_merge(df: pd.DataFrame, type: str) -> pd.DataFrame:
    """
    Prepare dataframe to have consistent column structure for merge. Missing columns are added with NaN values.

    Parameters:
        df (pd.Dataframe): Input dataframe either for the mass or the calcifications subset

    Returns:
        df (pd.Dataframe): Dataframe with standardised column structure.

    """
    df = df.copy()

    if type == "mass":
        df["calc_type"] = None
        df["calc_distribution"] = None
    else:
        df["mass_shape"] = None
        df["mass_margins"] = None
    
    return df

def save_dataframe(df: pd.DataFrame, output_csv_path: str):
    """
    Saves a pd.DataFrame to csv file.

    Parameters:
        df (pd.DataFrame): Dataframe to be saved.
        output_csv_path (str): Full file path to be used to save the csv.

    Returns:
        None
    """
    df.to_csv(output_csv_path, index=False)
    print(f"Saved: {output_csv_path}")

### DL related support function ###

def adjust_optimizer(model: torch.nn.Module,
                      model_name: str,
                      optimizer_name: str,
                      classifier_lr: float,
                      backbone_lr: float) -> torch.nn.Module:
    
    """
    Creates an optimizer with different learning rates for classifier and backbone parameters.

    Args:
        model (torch.nn.Module): The model containing trainable parameters.
        model_name (str): Name of the model architecture.
        optimizer_name (str): Optimizer to use ('adam', 'adamw', 'sgd').
        classifier_lr (float): Learning rate for classifier layer.
        backbone_lr (float): Learning rate for backbone layers.

    Returns:
        torch.nn.Module: Initialized optimizer.
    """
    
    # Separate parameters
    classifier_params = []
    backbone_params = []
    
    if model_name == "resnet50":
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "fc" in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
    
    elif model_name == "densenet121":
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "classifier" in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
    
    elif model_name == "inception_v3":
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "fc" in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
    
    # Group parameters for different learning rates
    param_groups = []
    
    # classifier lr to be reduced during backbone training
    if classifier_params:
        param_groups.append({
            'params': classifier_params,
            'lr': classifier_lr * 0.5,  
            'name': 'classifier'
        })
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })
    
    # Create optimizer with parameter groups
    if optimizer_name == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=1e-5, betas=(0.9, 0.999))
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=0.05)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-5, nesterov=True)
    else:
        logging.error(f"Invalid optimizer: {optimizer_name}")
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    
    logging.info(f"Created {optimizer_name} with adjusted learning rate per parameter group.")

    return optimizer

def optimizer_add_new_params(optimizer: torch.optim.Optimizer,
                            model: torch.nn.Module,
                            model_name: str,
                            classifier_lr: float,
                            backbone_lr: float) -> torch.optim.Optimizer:
    
    """
    Adds newly trainable parameters to the optimizer (e.g. after unfreezing layers).

    Args:
        optimizer (torch.optim.Optimizer): Existing optimizer.
        model (torch.nn.Module): Model with updated trainable parameters.
        model_name (str): Name of the model architecture.
        classifier_lr (float): Learning rate for classifier.
        backbone_lr (float): Learning rate for backbone.

    Returns:
        torch.optim.Optimizer: Updated optimizer.
    """
    
    # Identify new parameters that need to be added
    existing_params = {id(p) for group in optimizer.param_groups for p in group['params']}
    all_trainable_params = {id(p) for p in model.parameters() if p.requires_grad}
    new_param_ids = all_trainable_params - existing_params
    
    if new_param_ids:
        new_params = [p for p in model.parameters() if id(p) in new_param_ids]
        
        # Add new trainable parameters
        optimizer.add_param_group({
            'params': new_params,
            'lr': backbone_lr,
            'name': 'newly_unfrozen'
        })
        logging.info(f"Added {len(new_params)} new parameters to the optimizer")
        
        # Adjust existing learning rates for second phase
        for group in optimizer.param_groups:
            if group.get('name') == 'classifier' or 'fc' in str(group.get('params', [])):
                group['lr'] = classifier_lr * 0.5  
                logging.info(f"Reduced classifier learning rate to {group['lr']}")
    
    return optimizer

def unfreeze_layer(model: torch.nn.Module, 
                   model_name: str, 
                   num_depth:int) -> torch.nn.Module:
    """
    Unfreezes a specified number of top blocks in a Pytorch model. 

    Parameters:
        model(torch.nn.Module): Pytorch model
        model_name(str): name of the model. 
            Supported values are 'resnet50', 'densenet121', 'inception_v3'.
        num_depth: controls the number of top blocks to unfreeze.
            Supported values:
                1: Unfreeze the last block
                2. Unfreeze the last two blocks

    Returns:
        model(torch.nn.Module): Pytorch model with selected layers unfrozen for training.
    """

    if num_depth == 0:
        pass
    elif num_depth == 1:
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

    elif num_depth == 2:
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
        logging.error(f"Invalid model training depth {num_depth}. Valid values are 0, 1 or 2.")
        raise ValueError("Invalid model training depth passed on to unfreeze_layer().")

    return model

def save_best_model(model_name: str, 
                    model_states: dict, 
                    metrics: dict, 
                    model_best_config: dict, 
                    config: dict) -> None:
    """
    Saves the best-performing model checkpoint to disk.

    Args:
        model_name (str): Model name.
        model_states (dict): State dict from model.
        metrics (dict): Evaluation metrics.
        model_best_config (dict): Configuration used for best run.
        config (dict): General config including 'checkpoints' directory.
    """

    output_path = os.path.join(config["checkpoints"], f"{model_name}_best.pth")

    torch.save({
        "model_name": model_name,
        "model_state_dict": model_states,
        "metrics": metrics,
        "model_best_config": model_best_config,
        "config": config
    }, output_path)

    logging.info(f"Best model version for {model_name} saved.")

def load_model(model_name: str, 
               checkpoint: str, 
               device: torch.device, 
               num_classes: int) -> torch.nn.Module:
    """
    Loads a saved model checkpoint and sends it to the specified device.

    Parameters:
        model_name(str): name of the Pytorch model to load. Supported values: 'resnet50', 'densenet121', 'inception_v3'.
        checkpoint(str): file path to the checkpoint, that contains the saved model dictionary
        device(torch.device): Device to load the model to (such as 'cuda' or 'cpu')
        num_classes(int): Number of output classes
        
    Returns:
        torch.nn.Module: Pytorch model loaded from a checkpoint and sent to device
        
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes))
    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes))
    if model_name == "inception_v3":
        #model = models.inception_v3(weights=None)
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.aux_logits = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes))
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.debug(f"{model_name} model loaded from checkpoint.")
    model.to(device)
    logging.debug(f"Model sent to device {device}.")
    
    return model
      