# Sources used:
# https://docs.pytorch.org/vision/stable/transforms.html
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://docs.pytorch.org/vision/stable/models.html

import torchvision.transforms as transforms

# Normalisation with the use of ImageNet mean and standard
imgNetMean = [0.485, 0.456, 0.406]
imgNetStd = [0.229, 0.224, 0.225]

def apply_transforms(model_type:str) -> tuple:
    """
    Create training and validation image transforms based on the model type.

    Args:
        model_type (str): Name of the model architecture. Supported: 'resnet50', 'densenet121', 'inception_v3'.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: A tuple containing (training_transform, validation_transform).

    Raises:
        ValueError: If the model_type is not recognized.
    """

    if model_type in ["resnet50" , "densenet121"]:
        img_size = 512
    elif model_type == "inception_v3":
        img_size = 512
    else:
        raise ValueError ("Invalid model_type value: " + model_type)
    
    training_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean = imgNetMean, std = imgNetStd)
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean = imgNetMean, std = imgNetStd)
    ])

    return training_transform, validation_transform