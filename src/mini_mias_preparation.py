import pandas as pd
import logging
import os
from utils import save_dataframe

log_path = os.path.join("./logs", f"mias_dataset_preparation_log.log")
filepath = "./data/raw/MIAS_metadata/metadata.txt"
image_folder = "./data/raw/MIAS"

columns = [
    "reference", "background_tissue", "abnormality", "severity",
    "center_x", "center_y", "radius_in_pixel"
]

### Initialise logging ###
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler(log_path), logging.StreamHandler()])
    
logging.info(f"Logging initialised for MIAS dataset preparation, log file: {log_path}")

### Read resource file ###
with open(filepath, "r") as file:
    lines = file.readlines()
parsed_rows = []
for line in lines[1:]: 
    tokens = line.strip().split()
    tokens += ["Na"] * (len(columns) - len(tokens))
    parsed_rows.append(tokens[:len(columns)]) 
df = pd.DataFrame(parsed_rows, columns=columns)

logging.info("================= MIAS Dataset =================")
### Examine metadata content ###
logging.debug(df.head())
logging.debug(df.shape)
logging.debug(df.describe())
logging.info(f"Metadata file contains {len(df)} cases.")

### Check for duplicates ###
duplicates = df[df.duplicated(keep=False)]
if not duplicates.empty:
    logging.error(f"Duplicated entries found: {len(duplicates)}")
    logging.error(duplicates)
else:
    logging.info("All cases are unique, no duplicates found.")

### Check that all case files exist ###
for case in df["reference"]:
    file_path = os.path.join(image_folder,f"{case}.pgm")
    if not os.path.exists(file_path):
        logging.error(f"Missing image file for {case}.")

### Add filepath to the dataframe ###
df["filepath"] = df["reference"].apply(lambda ref: os.path.join(image_folder, f"{ref}.pgm"))

### Drop irrelevant columns ###
df.drop(columns=["center_x", "center_y", "radius_in_pixel"], inplace=True)

### Handle missing severity value and whitespaces ###
df["severity"] = df["severity"].apply(lambda x: str(x).strip() if pd.notna(x) else "Na")

### Examine classes ###
logging.info(f"Abnormality classes in the dataset: {df['abnormality'].unique()}")
logging.info(f"Severity classes in the dataset: {df['severity'].unique()}")

### Check for missing label ###
nan_count = df["severity"].isna().sum()
logging.info(f"Missing label count: {nan_count}")

### Create cleaned dataset csv ###
output_path = "./data/processed/mias_external_verification_set.csv"
save_dataframe(df=df,output_csv_path=output_path)

