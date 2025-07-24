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
    if model_type in ["resnet50" , "densenet121"]:
        # img_size = 224
        img_size = 512
    elif model_type == "inception_v3":
        img_size = 299
    else:
        raise ValueError ("Invalid model_type value: " + model_type)
    
    training_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
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