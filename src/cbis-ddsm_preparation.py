import pandas as pd
import os
import logging

# Load original dataset metadata files
Mass_Test = pd.read_csv("./data/raw/CBIS-DDSM_metadata/mass_case_description_test_set.csv")
Mass_Training = pd.read_csv("./data/raw/CBIS-DDSM_metadata/mass_case_description_train_set.csv")
Calc_Test = pd.read_csv("./data/raw/CBIS-DDSM_metadata/calc_case_description_test_set.csv")
Calc_Training = pd.read_csv("./data/raw/CBIS-DDSM_metadata/calc_case_description_train_set.csv")

log_path = os.path.join("./logs", f"cbis_dataset_preparation_log.log")
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler(log_path), logging.StreamHandler()])
    
logging.info(f"Logging initialised, log file: {log_path}")

# Output path
output_folder = "./data/processed/"
# Dataset root directory
root_dir ="./data/raw/CBIS-DDSM"

# Check Column names
logging.info("\n================= Mass Test Columns =================")
logging.info(Mass_Test.columns)
logging.info("\n================= Mass Training Columns =================")
logging.info(Mass_Training.columns)
logging.info("\n================= Calc Test Columns =================")
logging.info(Calc_Test.columns)
logging.info("\n================= Calc Training Columns =================")
logging.info(Calc_Training.columns)

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

# Fix Breast Density inconsistency
Mass_Test = standardise_columns(Mass_Test)
Mass_Training = standardise_columns(Mass_Training)
Calc_Test = standardise_columns(Calc_Test)
Calc_Training = standardise_columns(Calc_Training)
logging.info("Column names standardised across dataframes...")

# Get case id from file path
def create_case_id(file_path:str) -> str:
    """
    Extract the case ID from the file path (assuming the path structure uses '/' and the cse ID is the first directory in the path).

    Parameters:
        file_path(str): File path string.
    
    Returns:
        str: Extracted case ID.
    """
    return os.path.dirname(file_path).split("/")[0]

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

logging.info("\n================= Starting metadata adjustment =================")

def get_largest_dcm_file(case_id: str) -> str:
    """
    Return the file path to the largest Dicom file (full mammogram image) for a given case ID.

    Parameters:
        case_id (str): Case ID used as an unique identifier for a patient case.
    
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

# Update file paths
Mass_Test["image file path"] = Mass_Test["case_id"].apply(get_largest_dcm_file)
Mass_Training["image file path"] = Mass_Training["case_id"].apply(get_largest_dcm_file)
Calc_Test["image file path"] = Calc_Test["case_id"].apply(get_largest_dcm_file)
Calc_Training["image file path"] = Calc_Training["case_id"].apply(get_largest_dcm_file)
logging.info("File paths for full image mammograms have been updated to match local storage structure.")

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

# Adjust column structure
Mass_Test = prepare_for_merge(Mass_Test, "mass")
Mass_Training = prepare_for_merge(Mass_Training, "mass")
Calc_Test = prepare_for_merge(Calc_Test, "calc")
Calc_Training = prepare_for_merge(Calc_Training, "calc")
logging.info("Column structure standardised across dataframes...")

logging.info("\n================= Combine mass and calcification datasets =================")

Combined_Training = pd.concat([Mass_Training, Calc_Training], ignore_index=True)
Combined_Test = pd.concat([Mass_Test, Calc_Test], ignore_index=True)
logging.info("Datasets successfully merged...")

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

logging.info("\n================= Save dataframes =================")
save_dataframe(Combined_Training, os.path.join(output_folder, "combined_training_set_mapped.csv"))
save_dataframe(Combined_Test, os.path.join(output_folder, "combined_test_set_mapped.csv"))
logging.info(f"Prepared dataset information saved to {output_folder}.")
