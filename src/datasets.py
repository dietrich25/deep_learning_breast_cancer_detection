# sources used:
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
# https://www.geeksforgeeks.org/reading-image-opencv-using-python/

from torch.utils.data import Dataset
import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

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


        img = Image.fromarray(img_array)

        return img.convert("RGB") # models expect 3 channels

    except Exception as e:
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
            return None
        img = Image.fromarray(img_array)
        return img
    
    except Exception as e:
        print(f"Failed to load PGM file {file_path}: {e}")
        return None

def load_cbis_ddsm_split(train_df_path: str, 
                         test_df_path: str, 
                         val_split_ratio = 0.25, 
                         random_state = 42):
    
    training_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    train_df, val_df = train_test_split(training_df, 
                                        test_size=val_split_ratio,
                                        stratify=training_df["pathology"], 
                                        random_state=random_state)
    
    return train_df, val_df, test_df


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
        #cropped_img_path = os.path.normpath(self.data.loc[idx, "cropped image file path"])
        img_path = os.path.normpath(self.data.loc[idx, "image file path"])

        # read dicom images
        #cropped_img = load_dicom_as_pil_image(cropped_img_path)
        img = load_dicom_as_pil_image(img_path)

        # get and map the label
        label_str = self.data.loc[idx, "pathology"]
        label = self.label_map.get(label_str, -1) # -1 = unknown, 0 - benign, 1 - malignant

        # apply transformation
        if self.transform:
            #cropped_img = self.transform(cropped_img)
            img = self.transform(img)

        # return image and label
        return img, label

class MIASDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label = {
            "": 0,
            "B": 0,
            "M": 1
        }
    
    def __len__(self):
        return len(self.data)
    
    def __get_item__(self, idx):
        #TBD
        return 0