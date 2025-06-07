# Sources used:
# https://docs.pytorch.org/vision/stable/transforms.html
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://docs.pytorch.org/vision/stable/models.html

import torchvision.transforms as transforms
from PIL import Image

# Preprocessing process:
# Resize images
# Greyscale to RGB conversion - models expect 3 channels
# Data normalization
# Data augmentations - for training only
# Tensor conversion

# Normalisation with the use of ImageNet mean and standard
imgNetMean = [0.485, 0.456, 0.406]
imgNetStd = [0.229, 0.224, 0.225]

def apply_transforms(model_type):
    if model_type in ["resnet" , "densenet"]:
        img_size = 224
    elif model_type == "inception":
        img_size = 299
    else:
        raise ValueError ("Invalid model_type value: " + model_type)
    
    training_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.RandomHorizontalFlip(p=0.5), # random flip with 50% probability
        transforms.RandomRotation(10), # random rotation -10 to +10 degrees
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