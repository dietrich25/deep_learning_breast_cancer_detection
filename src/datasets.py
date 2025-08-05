# sources used:
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
# https://www.geeksforgeeks.org/reading-image-opencv-using-python/

from torch.utils.data import Dataset
import os
import pydicom
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

def load_dicom_as_pil_image(file_path: str) -> Image.Image:
    """
    Read a DICOM file and convert it to an RGB PIL image.

    Args:
        file_path(string): Path to the .dcm file.
    
    Returns:
        PIL.Image.Image with 3-channels (RGB).
    """
    try:
        dcm = pydicom.dcmread(file_path)
        img_array = dcm.pixel_array.astype(np.float32)
        # Normalise values to 0-255 range
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
        img_array = img_array.astype(np.uint8)

        img = Image.fromarray(img_array)

        return img.convert("RGB") # models expect 3 channels

    except Exception as e:
        logging.error(f"Failed to load DICOM file {file_path}: {e}")
        print(f"Failed to load DICOM file {file_path}: {e}")
        return None

def load_pgm_as_pil_image(file_path: str) -> Image.Image:
    """
    Read a PGM file and convert it to an RGB PIL image.

    Args:
        file_path(string): Path to the .pgm file.
    
    Returns:
        PIL.Image.Image with 3-channels (RGB).
    """
    try:
        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            raise RuntimeError(f"Could not read pgm image: {file_path}")
        img = Image.fromarray(img_array)
        return img
    
    except Exception as e:
        logging.error(f"Failed to load PGM file {file_path}: {e}")
        print(f"Failed to load PGM file {file_path}: {e}")
        return None

def load_cbis_ddsm_split(train_df_path: str, 
                         test_df_path: str, 
                         val_split_ratio = 0.25, 
                         random_state = 42):
    """
    Load CBIS-DDSM datasets and create stratified (by pathology) train/validation/test splits.

    Parameters:
        train_df_path(str): Path to the csv file containing metadata about the CBIS-DDSM official training set
        test_df_path(str): Path to the csv file containing metadata about the CBIS-DDSM official test set
        val_split_ratio(float): proportion of the training dataset to use for validation (0.0 - 1.0)
        random_state(int): Random seed to make dataset split reproducible

    Returns:
        tuple(train_df, val_df, test_df):
        - train_df(Dataframe): training subset after stratified split
        - val_df(Dataframe): validation subset after stratified split
        - test_df(Dataframe): official CBIS-DDSM test dataset
    """
    
    training_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    ### Test for data leak, duplicated cases across sets
    train_case_ids = set(training_df["case_id"])
    test_case_ids = set(test_df["case_id"])
    case_duplications = train_case_ids.intersection(test_case_ids)
    if case_duplications:
        logging.error(f"Data duplications detected between training and test datasets: {len(case_duplications)} cases can be found in both.")
        raise ValueError("Data duplications detected between training and test datasets")
    else:
        logging.info("No data duplications detected between the training and test dataset.")

    train_df, val_df = train_test_split(training_df, 
                                    test_size=val_split_ratio,
                                    stratify=training_df["pathology"], 
                                    random_state=random_state)
    
    # Class distribution in the training dataset
    logging.info(f"--- CBIS-DDSM official training dataset content: {len(training_df)} cases ---")
    train_class_counts = Counter(training_df["pathology"])
    for pathology, count in sorted(train_class_counts.items()):
        perc = (count/len(training_df)) * 100
        logging.info(f"Training dataset class distribution: Pathology: {pathology} | Count: {count} | Percentage: {perc:.1f}%")

    # Class distribution in the training stratified split
    logging.info(f"CBIS-DDSM training subset content: {len(train_df)} cases.")
    sub_train_class_counts = Counter(train_df["pathology"])
    for pathology, count in sorted(sub_train_class_counts.items()):
        perc = (count/len(train_df)) * 100
        logging.info(f"Training subset class distribution: Pathology: {pathology} | Count: {count} | Percentage: {perc:.1f}%")
    
    # Class distribution in the validation stratified split
    logging.info(f"CBIS-DDSM training subset content: {len(val_df)} cases.")
    val_class_counts = Counter(val_df["pathology"])
    for pathology, count in sorted(val_class_counts.items()):
        perc = (count/len(val_df)) * 100
        logging.info(f"Validation subset class distribution: Pathology: {pathology} | Count: {count} | Percentage: {perc:.1f}%")
    
    # Class distribution in the test dataset
    logging.info(f"--- CBIS-DDSM official training dataset content: {len(test_df)} cases ---")
    test_class_counts = Counter(test_df["pathology"])
    for pathology, count in sorted(test_class_counts.items()):
        perc = (count/len(test_df)) * 100
        logging.info(f"Test dataset class distribution: Pathology: {pathology} | Count: {count} | Percentage: {perc:.1f}%")

    return train_df, val_df, test_df

def get_dataset_labels(df):
    """Get the pathology labels from the CBIS-DDSM cases."""
    label_map = {
        "BENIGN": 0,
        "BENIGN_WITHOUT_CALLBACK": 0,
        "MALIGNANT": 1
    }

    labels = []
    for _, row in df.iterrows():
        label_str = row["pathology"]
        label = label_map.get(label_str, -1)
        if label != -1:
            labels.append(label)
    
    return labels

def balance_cbis_ddsm_class_weights(dataset_labels, device):
    
    class_counts = Counter(dataset_labels)
    unique_classes = sorted(class_counts.keys())

    class_weights = compute_class_weight("balanced",
                                         classes = np.array(unique_classes),
                                         y = dataset_labels)
    
    logging.debug(f"Class weights: {class_weights}")

    weight_tensor = torch.FloatTensor(class_weights).to(device)

    return weight_tensor

def load_mias_dataset(file_path:str) -> pd.DataFrame:

    df = pd.read_csv(file_path)
    
    logging.info("\n%s", df.head())

    # Class distribution in the external test set dataset
    logging.info(f"--- Mini-MIAS external validation dataset content: {len(df)} cases ---")
    class_counts = Counter(df["severity"])
    for severity, count in sorted(class_counts.items()):
        perc = (count/len(df)) * 100
        logging.info(f"Training dataset class distribution: Severity: {severity} | Count: {count} | Percentage: {perc:.1f}%")
    
    # Abnormality distribution in the external test set dataset
    logging.info(f"--- Mini-MIAS external validation dataset abnormality subclasses ---")
    class_counts = Counter(df["abnormality"])
    for abnorm, count in sorted(class_counts.items()):
        perc = (count/len(df)) * 100
        logging.info(f"Training dataset class distribution: Abnormality: {abnorm} | Count: {count} | Percentage: {perc:.1f}%")
    
    return df


class CBISDDSMDataset(Dataset):
    def __init__(self, dataframe, transform = None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_map = {
            "BENIGN": 0,
            "BENIGN_WITHOUT_CALLBACK": 0,
            "MALIGNANT": 1
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img_path = os.path.normpath(self.data.loc[idx, "image file path"])

        # read dicom images
        img = load_dicom_as_pil_image(img_path)

        # get and map the label
        label_str = self.data.loc[idx, "pathology"]
        label = self.label_map.get(label_str, -1) # -1 = unknown, 0 - benign, 1 - malignant

        # apply transformation
        if self.transform:
            img = self.transform(img)

        return img, label

class MIASDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.data["severity"] = self.data["severity"].astype(str).str.strip()
        self.transform = transform
        self.label_map = {
            "Na": 0,
            "B": 0,
            "M": 1
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.normpath(self.data.loc[idx, "filepath"])

        # read pgm image
        img = load_pgm_as_pil_image(img_path)

        # get and map the label
        label_str = self.data.loc[idx, "severity"]
        label = self.label_map.get(label_str, -1)
        if label == -1:
            logging.error(f"Invalid label at index {idx}: severity='{label_str}', filepath='{img_path}'")
            logging.error(f"Available keys in label_map: {list(self.label_map.keys())}")
            logging.error(f"Label string representation: {repr(label_str)}")

        # apply transformation
        if self.transform:
            img = self.transform(img)

        return img, label