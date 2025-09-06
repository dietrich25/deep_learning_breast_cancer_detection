import pandas as pd
import os
import logging
from utils import setup_logging, standardise_columns, create_case_id, get_largest_dcm_file, prepare_for_merge, save_dataframe

# Load original dataset metadata files
Mass_Test = pd.read_csv("./data/raw/CBIS-DDSM_metadata/mass_case_description_test_set.csv")
Mass_Training = pd.read_csv("./data/raw/CBIS-DDSM_metadata/mass_case_description_train_set.csv")
Calc_Test = pd.read_csv("./data/raw/CBIS-DDSM_metadata/calc_case_description_test_set.csv")
Calc_Training = pd.read_csv("./data/raw/CBIS-DDSM_metadata/calc_case_description_train_set.csv")

log_path = os.path.join("./logs", f"cbis_dataset_preparation_log.log")
setup_logging(log_path)

# Output path
output_folder = "./data/processed/"
# Dataset root directory
root_dir ="./data/raw/CBIS-DDSM"

# Check Column names
logging.info("================= Mass Test Columns =================")
logging.info(Mass_Test.columns)
logging.info("================= Mass Training Columns =================")
logging.info(Mass_Training.columns)
logging.info("================= Calc Test Columns =================")
logging.info(Calc_Test.columns)
logging.info("================= Calc Training Columns =================")
logging.info(Calc_Training.columns)

# Fix Breast Density inconsistency
Mass_Test = standardise_columns(Mass_Test)
Mass_Training = standardise_columns(Mass_Training)
Calc_Test = standardise_columns(Calc_Test)
Calc_Training = standardise_columns(Calc_Training)
logging.info("Column names standardised across dataframes...")

# Add case id to each dataframe
Mass_Test["case_id"] = Mass_Test["image_file_path"].apply(create_case_id)
Mass_Training["case_id"] = Mass_Training["image_file_path"].apply(create_case_id)
Calc_Test["case_id"] = Calc_Test["image_file_path"].apply(create_case_id)
Calc_Training["case_id"] = Calc_Training["image_file_path"].apply(create_case_id)
logging.info("Case ID extracted from the file path and added as a column to each dataframe...")

# drop binary ROI file path and cropped file path
columns_to_drop = ["cropped_image_file_path", "ROI_mask_file_path"]
Mass_Test = Mass_Test.drop(columns=columns_to_drop)
Mass_Training = Mass_Training.drop(columns=columns_to_drop)
Calc_Test = Calc_Test.drop(columns=columns_to_drop)
Calc_Training = Calc_Training.drop(columns=columns_to_drop)
logging.info("Binary ROI file path and cropped image path dropped from the dataframes (out of scope)...")

logging.info("================= Starting metadata adjustment =================")

# Update file paths
Mass_Test["image file path"] = Mass_Test["case_id"].apply(get_largest_dcm_file)
Mass_Training["image file path"] = Mass_Training["case_id"].apply(get_largest_dcm_file)
Calc_Test["image file path"] = Calc_Test["case_id"].apply(get_largest_dcm_file)
Calc_Training["image file path"] = Calc_Training["case_id"].apply(get_largest_dcm_file)
logging.info("File paths for full image mammograms have been updated to match local storage structure.")

# Adjust column structure
Mass_Test = prepare_for_merge(Mass_Test, "mass")
Mass_Training = prepare_for_merge(Mass_Training, "mass")
Calc_Test = prepare_for_merge(Calc_Test, "calc")
Calc_Training = prepare_for_merge(Calc_Training, "calc")
logging.info("Column structure standardised across dataframes...")

logging.info("================= Combine mass and calcification datasets =================")

Combined_Training = pd.concat([Mass_Training, Calc_Training], ignore_index=True)
Combined_Test = pd.concat([Mass_Test, Calc_Test], ignore_index=True)
logging.info("Datasets successfully merged...")

logging.info("================= Inspect class distribution =================")
training_distribution = Combined_Training['pathology'].value_counts()
test_distribution = Combined_Test['pathology'].value_counts()
perc_training = Combined_Training['pathology'].value_counts(normalize=True) * 100
perc_test = Combined_Test['pathology'].value_counts(normalize=True) * 100

logging.info(f"Training set class value distribution: {training_distribution}")
logging.info(f"Training set percentage distribution: {perc_training.round(2)}%")
logging.info(f"Test set class value distribution: {test_distribution}")
logging.info(f"Test set percentage distribution: {perc_test.round(2)}%")

logging.info("================= Save dataframes =================")
save_dataframe(Combined_Training, os.path.join(output_folder, "combined_training_set_mapped.csv"))
save_dataframe(Combined_Test, os.path.join(output_folder, "combined_test_set_mapped.csv"))
logging.info(f"Prepared dataset information saved to {output_folder}.")
