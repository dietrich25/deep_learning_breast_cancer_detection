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
from utils import unfreeze_layer, adjust_optimizer, optimizer_add_new_params
from evaluation import evaluate_model_performance

# code adapted from https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
class Earlystopping:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, current_score):

        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if current_score > self.best_score + self.delta:
            self.best_score = current_score
            self.counter = 0
            logging.debug(f"EarlyStop counter reseted to: {self.counter}")
        else:
            self.counter += 1
            logging.debug(f"EarlyStop counter increased to: {self.counter}")

        return self.counter >= self.patience
     
def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps = 4):

    start_time = time.time()

    # Set the model into training mode
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    num_batches = len(dataloader)
    logging.debug(f"Starting model training epoch with {num_batches} batches...")

    optimizer.zero_grad()

    # Iterate through all the batches in the dataset
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Position data to the same device where the model is
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss = loss/accumulation_steps
        
        loss.backward()
      

        if(batch_idx+1)%accumulation_steps == 0 or (batch_idx+1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients

            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

        # Calc correct predictions
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)
    
    epoch_time = time.time() - start_time

    epoch_loss = running_loss / num_batches
    epoch_acc = correct_predictions / total_predictions
    logging.info(f"Training epoch completed: {epoch_time:.1f}s | Training Loss:{epoch_loss:.4f} | Training Accuracy={epoch_acc:.4f}")

    return epoch_acc, epoch_loss
    
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

    logging.debug(f"Initialising {model_name} with {num_classes} classes.")

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
    if model_name == "resnet50" or model_name == "inception_v3":
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "densenet121":
        model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
        )
    logging.debug(f"Classifier replaced from {in_features} to {num_classes} features.")

    # Unfreeze classifier layer
    classifier = getattr(model, classifier_name)
    for param in classifier.parameters():
        param.requires_grad = True
    logging.debug("Classifier layer unfrozen...")
    
    logging.debug(f"Model {model_name} prepared for training...")
    return model

# code adapted from https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model_phase(model: torch.nn.Module,
                model_name:str,
                phase: str,
                learning_rate: float,
                optimizer:torch.optim.Optimizer,
                criterion: torch.nn.Module,
                train_dataloader: DataLoader,
                validation_dataloader: DataLoader,
                config: dict) -> tuple:
    
    total_start_time = time.time()
    logging.debug(f"Initialising {phase} phase training for {model_name}...")

    accumulation_steps = 8

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    early_stop = Earlystopping(patience=10, delta=0.01)

    history = {
        "model_name": [], "LR": [], "training_depth": [], "optimizer": [],
        "train_loss": [], "train_acc": [],
        "val_acc": [], "val_loss": [], "val_recall": [], "val_precision": [],
        "val_f1": [], "val_roc_auc": [], "val_specificity": []
    }

    best_f1 = 0.0
    best_roc_auc = 0.0
    best_model_state_dict = None
    best_optimizer_state_dict = None
    best_metrics = {}

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        logging.info(f"Epoch progress: {epoch+1}/{config['epochs']}")

        train_acc, train_loss = train_epoch(model, train_dataloader,criterion,optimizer,config["device"], accumulation_steps=accumulation_steps)

        val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = evaluate_model_performance(model, validation_dataloader, criterion, config["device"])

        epoch_time = time.time() - epoch_start_time

        history["model_name"].append(model_name)
        history["LR"].append(learning_rate)
        history["training_depth"].append(phase)
        history["optimizer"].append(type(optimizer).__name__)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(val_recall)
        history["val_precision"].append(val_precision)
        history["val_f1"].append(val_f1)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_specificity"].append(val_specificity)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            logging.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")

        logging.info(f"Epoch {epoch+1}/{config['epochs']} completed in {epoch_time:.1f}s | Val F1: {val_f1:.4f} | Val ROC_AUC: {val_roc_auc:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state_dict = model.state_dict().copy()
            best_optimizer_state_dict = optimizer.state_dict().copy()
            best_metrics = {
                "epoch": epoch,
                "f1": val_f1,
                "accuracy": val_acc,
                "recall": val_recall,
                "precision": val_precision,
                "roc_auc": val_roc_auc,
                "specificity": val_specificity
            }
            logging.info(f"**New best model saved with F1: {val_f1:.4f}.**")
        
        if early_stop(val_f1):
            logging.warning(f"Early stopping triggered at epoch {epoch + 1}.")
            break
    
    total_time = time.time() - total_start_time
    logging.info(f"Training phase {phase} completed. Total time: {total_time/60:.1f} minutes | Best F1-score {best_f1:.4f}")

    return history, best_model_state_dict, best_optimizer_state_dict, best_metrics

def progressive_model_training_test(model_name: str,
                            classifier_lr: float,
                            backbone_lr: float,
                            optimizer_name: str,
                            training_depth: int,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            config: dict,
                            class_weigths: torch.FloatTensor):
            
    prog_start_time = time.time()

    # Record training and validation results
    full_history = {
        "phase": [],
        "model_name": [], "LR": [], "training_depth": [], "optimizer": [],
        "train_loss": [], "train_acc": [],
        "val_acc": [], "val_loss": [], "val_recall": [], "val_precision": [],
        "val_f1": [], "val_roc_auc": [], "val_specificity": []
    }

    # General performance trackers
    overall_best_f1 = 0.0
    overall_best_ROC_AUC = 0.0
    overall_best_state = None
    overall_best_metrics = {}

    ### Phase 1 - Replace and train classifier layer ####
    logging.info("-- Phase 1: Classifier layer training --")

    # Load model with pre-trained weights and replaced, unfrozen classifier
    model = prepare_model(model_name, config["num_classes"], "classifier_only")
    model = model.to(config["device"])
    # Set model to training state
    model.train()
    logging.debug("Model training mode activated...")


    # Initialise loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weigths,label_smoothing=0.05)

    # Initialise optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=classifier_lr, weight_decay=1e-5, betas=(0.9, 0.999))
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=classifier_lr, weight_decay=0.05)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=classifier_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
    else:
        logging.error(f"Invalid optimizer passed to progressive_model_training(): {optimizer_name}")
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
            
    phase1_history, phase1_best_state, phase1_best_optimizer_state, phase1_best_metrics = train_model_phase(
        model=model,
        model_name=model_name,
        phase="classifier",
        learning_rate = classifier_lr,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        config=config
    )

    # Append training history
    for key in phase1_history:
        if key == "phase":
            full_history[key].extend(["classifier"] * len(phase1_history["train_loss"]))
        elif key in full_history:
            full_history[key].extend(phase1_history[key])

    # Update best performance trackers
    if phase1_best_metrics["f1"] > overall_best_f1:
        overall_best_f1 = phase1_best_metrics["f1"]
        overall_best_state = phase1_best_state
        overall_best_metrics = phase1_best_metrics
        overall_best_metrics["phase"] = "classifier"

    if training_depth > 0 and phase1_best_state is not None:
        try:
            model.load_state_dict(phase1_best_state)
            logging.info("Best performing model state passed to phase 2.")
        except Exception as e:
            logging.error(f"Failed to load previous best model state: {e}")
            raise RuntimeError("Failed to load previous best model state") from e
        model = unfreeze_layer(model, model_name, training_depth)

        if training_depth == 1:
            try:
                optimizer.load_state_dict(phase1_best_optimizer_state)
                optimizer = optimizer_add_new_params(optimizer, model, model_name, classifier_lr, backbone_lr)
            except Exception as e:
                logging.error(f"Failed to load and adjust previous best optimizer state: {e}")
                raise RuntimeError("Failed to load and adjust previous best optimizer state") from e
        else:
            # Initialise new optimizer with adjusted learning rates
            optimizer = adjust_optimizer(model=model,
                model_name=model_name,
                optimizer_name=optimizer_name,
                classifier_lr=classifier_lr, 
                backbone_lr=backbone_lr
            )

        logging.info("-- Phase 2: Progressive backbone training --")
        phase2_history, phase2_best_state, phase2_best_optimizer_state, phase2_best_metrics = train_model_phase(
            model=model,
            model_name=model_name,
            phase=f"backbone_{training_depth}",
            learning_rate = classifier_lr,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_loader,
            validation_dataloader=val_loader,
            config=config
        )

        # Update history
        for key in phase2_history:
            if key == "phase":
                full_history[key].extend([f"backbone_{training_depth}"] * len(phase2_history["train_loss"]))
            elif key in full_history:
                full_history[key].extend(phase2_history[key])
        
        # Update performance trackers
        if phase2_best_metrics["f1"] > overall_best_f1:
            overall_best_f1 = phase2_best_metrics["f1"]
            overall_best_state = phase2_best_state
            overall_best_metrics = phase2_best_metrics
            overall_best_metrics["phase"] = f"backbone_depth_{training_depth}"
        
    prog_end_time = time.time()
    total_time = prog_end_time - prog_start_time
    logging.info(f"--- Progressive model training finished in {total_time/60:.1f} minutes. Best ROC_AUC: {overall_best_ROC_AUC:.4f} ---")

    return full_history, overall_best_state, overall_best_metrics


def progressive_model_training(model_name: str,
                                    classifier_lr: float,
                                    backbone_lr: float,
                                    optimizer_name: str,
                                    training_depth: int,
                                    train_loader: DataLoader,
                                    val_loader: DataLoader,
                                    config: dict,
                                    class_weights: torch.FloatTensor):

    prog_start_time = time.time()
    
    full_history = {
        "phase": [],
        "model_name": [], "LR": [], "training_depth": [], "optimizer": [],
        "train_loss": [], "train_acc": [],
        "val_acc": [], "val_loss": [], "val_recall": [], "val_precision": [],
        "val_f1": [], "val_roc_auc": [], "val_specificity": []
    }

    # General performance trackers
    overall_best_f1 = 0.0
    overall_best_state = None
    overall_best_metrics = {}

    ### Phase 1 - Replace and train classifier layer ####
    logging.info("-- Phase 1: Classifier layer training --")

    model = prepare_model(model_name, config["num_classes"], "classifier_only")
    model = model.to(config["device"])
    model.train()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Optimizer for phase 1 (with only classifier parameters)
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=classifier_lr, weight_decay=1e-5, betas=(0.9, 0.999))
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=classifier_lr, weight_decay=0.05)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=classifier_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
            
    phase1_history, phase1_best_state, phase1_best_optimizer_state, phase1_best_metrics = train_model_phase(
        model=model,
        model_name=model_name,
        phase="classifier",
        learning_rate=classifier_lr,  # Fixed: use actual classifier_lr
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        config=config
    )

    # Append training history
    for key in phase1_history:
        if key in full_history:
            full_history[key].extend(phase1_history[key])
    # Add phase info
    full_history["phase"].extend(["classifier"] * len(phase1_history["train_loss"]))

    # Update best performance trackers
    if phase1_best_metrics["f1"] > overall_best_f1:
        overall_best_f1 = phase1_best_metrics["f1"]
        overall_best_state = phase1_best_state
        overall_best_metrics = phase1_best_metrics.copy()
        overall_best_metrics["phase"] = "classifier"

    ### Phase 2 - Progressive backbone training ###
    if training_depth > 0 and phase1_best_state is not None:
        logging.info("-- Phase 2: Progressive backbone training --")
        
        # Load best model from phase 1
        try:
            model.load_state_dict(phase1_best_state)
            logging.info("Best performing model state loaded for phase 2.")
        except Exception as e:
            logging.error(f"Failed to load previous best model state: {e}")
            raise RuntimeError("Failed to load previous best model state") from e
        
        # Unfreeze backbone layers
        model = unfreeze_layer(model, model_name, training_depth)
        
        # Create new optimizer with differentiated learning rates
        optimizer = setup_differentiated_optimizer_phase2(
            model, model_name, optimizer_name, classifier_lr, backbone_lr
        )

        phase2_history, phase2_best_state, phase2_best_optimizer_state, phase2_best_metrics = train_model_phase(
            model=model,
            model_name=model_name,
            phase=f"backbone_{training_depth}",
            learning_rate=backbone_lr,  
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_loader,
            validation_dataloader=val_loader,
            config=config
        )

        # Update history
        for key in phase2_history:
            if key in full_history:
                full_history[key].extend(phase2_history[key])
        # Add phase info
        full_history["phase"].extend([f"backbone_{training_depth}"] * len(phase2_history["train_loss"]))
        
        # Update performance trackers
        if phase2_best_metrics["f1"] > overall_best_f1:
            overall_best_f1 = phase2_best_metrics["f1"]
            overall_best_state = phase2_best_state
            overall_best_metrics = phase2_best_metrics.copy()
            overall_best_metrics["phase"] = f"backbone_depth_{training_depth}"
        
    prog_end_time = time.time()
    total_time = prog_end_time - prog_start_time
    logging.info(f"--- Progressive model training finished in {total_time/60:.1f} minutes. Best F1: {overall_best_f1:.4f} ---")

    return full_history, overall_best_state, overall_best_metrics

def setup_differentiated_optimizer_phase2(model: torch.nn.Module,
                                          
                                         model_name: str,
                                         optimizer_name: str,
                                         classifier_lr: float,
                                         backbone_lr: float) -> torch.optim.Optimizer:
    """
    Create a fresh optimizer for phase 2 with differentiated learning rates
    """
    # Get classifier parameters
    if model_name in ["resnet50", "inception_v3"]:
        classifier_params = list(model.fc.parameters())
    elif model_name == "densenet121":
        classifier_params = list(model.classifier.parameters())
    
    # Get backbone parameters (excluding classifier)
    classifier_param_ids = set(id(p) for p in classifier_params)
    backbone_params = [p for p in model.parameters() 
                      if p.requires_grad and id(p) not in classifier_param_ids]
    
    # Create parameter groups
    param_groups = [
        {'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier'},
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'}
    ]
    
    # Create optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=1e-5, betas=(0.9, 0.999))
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=0.05)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-5, nesterov=True)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    
    logging.info(f"Phase 2 optimizer created with classifier LR: {classifier_lr:.6f}, backbone LR: {backbone_lr:.6f}")
    
    return optimizer

# simplified trainign pipeline to fix likely gradient issues between phases and make debugging eaiser
# also 2 phase training took due to 2x early stopping limits
# now classifier and lower layers are trained together

def prepare_model_simple(model_name: str, num_classes: int, training_depth: int) -> torch.nn.Module:
    """
    Prepare model with unfrozen layers based on training depth from the start
    """
    logging.debug(f"Initialising {model_name} with {num_classes} classes.")

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        classifier_name = "fc"
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        classifier_name = "classifier"
    elif model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False  # Disable auxiliary logits after model creation
        classifier_name = "fc"
    else:
        raise ValueError("Invalid model_name. Supported models: 'resnet50', 'densenet121', 'inception_v3'")
    
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    logging.debug("All parameters frozen...")
    
    # Replace classifier
    classifier = getattr(model, classifier_name)
    in_features = classifier.in_features
    
    if model_name in ["resnet50", "inception_v3"]:
        new_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        model.fc = new_classifier
    elif model_name == "densenet121":
        new_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        model.classifier = new_classifier
    
    # Always unfreeze classifier (new classifier is unfrozen by default)
    classifier = getattr(model, classifier_name)
    for param in classifier.parameters():
        param.requires_grad = True
    logging.debug("Classifier layer unfrozen...")
    
    # Unfreeze backbone layers based on training_depth
    if training_depth > 0:
        model = unfreeze_layer(model, model_name, training_depth)
        logging.debug(f"Backbone layers unfrozen (depth: {training_depth})")
    
    return model

def setup_differentiated_optimizer(model: torch.nn.Module, 
                                 model_name: str,
                                 optimizer_name: str,
                                 classifier_lr: float, 
                                 backbone_lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for classifier and backbone
    """
    # Get classifier parameters (convert generator to list and filter for requires_grad)
    if model_name in ["resnet50", "inception_v3"]:
        classifier_params = [p for p in model.fc.parameters() if p.requires_grad]
    elif model_name == "densenet121":
        classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    
    # Get backbone parameters (only those that require grad, excluding classifier)
    classifier_param_ids = set(id(p) for p in classifier_params)
    backbone_params = [p for p in model.parameters() 
                      if p.requires_grad and id(p) not in classifier_param_ids]
    
    # Create parameter groups
    param_groups = []
    
    if classifier_params:
        param_groups.append({
            'params': classifier_params, 
            'lr': classifier_lr, 
            'name': 'classifier'
        })
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params, 
            'lr': backbone_lr, 
            'name': 'backbone'
        })
    
    if not param_groups:
        raise ValueError("No trainable parameters found!")
    
    # Create optimizer
    if optimizer_name == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=1e-5, betas=(0.9, 0.999))
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-5, nesterov=True)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    
    # Log parameter groups
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'])
        logging.info(f"Parameter group {group['name']}: {num_params:,} parameters, LR: {group['lr']:.6f}")
    
    return optimizer

def single_phase_training(model_name: str,
                         classifier_lr: float,
                         backbone_lr: float,
                         optimizer_name: str,
                         training_depth: int,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         config: dict,
                         class_weights: torch.FloatTensor):
    """
    Single-phase training with differentiated learning rates
    """
    start_time = time.time()
    
    # Prepare model with appropriate layers unfrozen
    model = prepare_model_simple(model_name, config["num_classes"], training_depth)
    model = model.to(config["device"])
    
    # Setup optimizer with differentiated learning rates AFTER unfreezing
    optimizer = setup_differentiated_optimizer(
        model, model_name, optimizer_name, classifier_lr, backbone_lr
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss() # removed class weigths
    
    # Training setup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    early_stop = Earlystopping(patience=5, delta=0.01)
    
    # Training history
    history = {
        "model_name": [], "classifier_lr": [], "backbone_lr": [], "training_depth": [], 
        "optimizer": [], "train_loss": [], "train_acc": [],
        "val_acc": [], "val_loss": [], "val_recall": [], "val_precision": [],
        "val_f1": [], "val_roc_auc": [], "val_specificity": []
    }
    
    best_f1 = 0.0
    best_model_state = None
    best_metrics = {}
    
    accumulation_steps = 8
    
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        logging.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Training
        train_acc, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config["device"], accumulation_steps
        )
        
        # Validation
        val_acc, val_loss, val_recall, val_precision, val_f1, val_roc_auc, val_specificity = \
            evaluate_model_performance(model, val_loader, criterion, config["device"])
        
        # Record history
        history["model_name"].append(model_name)
        history["classifier_lr"].append(classifier_lr)
        history["backbone_lr"].append(backbone_lr)
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
        
        # Learning rate scheduling
        old_lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler.step(val_f1)
        new_lrs = [group['lr'] for group in optimizer.param_groups]
        
        if old_lrs != new_lrs:
            for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                group_name = optimizer.param_groups[i]['name']
                logging.info(f"LR changed for {group_name}: {old_lr:.6f} -> {new_lr:.6f}")
        
        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch completed in {epoch_time:.1f}s | Val F1: {val_f1:.4f} | Val ROC_AUC: {val_roc_auc:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            best_metrics = {
                "epoch": epoch,
                "f1": val_f1,
                "accuracy": val_acc,
                "recall": val_recall,
                "precision": val_precision,
                "roc_auc": val_roc_auc,
                "specificity": val_specificity
            }
            logging.info(f"**New best model saved with F1: {val_f1:.4f}**")
        
        # Early stopping
        if early_stop(val_f1):
            logging.warning(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time/60:.1f} minutes. Best F1: {best_f1:.4f}")
    
    return history, best_model_state, best_metrics