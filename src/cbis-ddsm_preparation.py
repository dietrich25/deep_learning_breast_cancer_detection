import pandas as pd
import os
import re
import glob

# Load original dataset metadata files
Mass_Test = pd.read_csv("./data/raw/CBIS-DDSM_metadata/mass_case_description_test_set.csv")
Mass_Training = pd.read_csv("./data/raw/CBIS-DDSM_metadata/mass_case_description_train_set.csv")
Calc_Test = pd.read_csv("./data/raw/CBIS-DDSM_metadata/calc_case_description_test_set.csv")
Calc_Training = pd.read_csv("./data/raw/CBIS-DDSM_metadata/calc_case_description_train_set.csv")

# Output path
output_folder = "./data/processed/"

# Dataset root directory
root_dir ="./data/raw/CBIS-DDSM"

# Check Column names
print("\n================= Mass Test Columns =================")
print(Mass_Test.columns)
print("\n================= Mass Training Columns =================")
print(Mass_Training.columns)
print("\n================= Calc Test Columns =================")
print(Calc_Test.columns)
print("\n================= Calc Training Columns =================")
print(Calc_Training.columns)

# Get case id from file path
def create_case_id(file_path):
    return os.path.dirname(file_path).split("/")[0]

# Add case id to each dataframe
Mass_Test["case_id"] = Mass_Test["image file path"].apply(create_case_id)
Mass_Training["case_id"] = Mass_Training["image file path"].apply(create_case_id)
Calc_Test["case_id"] = Calc_Test["image file path"].apply(create_case_id)
Calc_Training["case_id"] = Calc_Training["image file path"].apply(create_case_id)

# drop binary ROI file path and cropped file path
columns_to_drop = ["cropped image file path", "ROI mask file path"]
Mass_Test = Mass_Test.drop(columns=columns_to_drop)
Mass_Training = Mass_Training.drop(columns=columns_to_drop)
Calc_Test = Calc_Test.drop(columns=columns_to_drop)
Calc_Training = Calc_Training.drop(columns=columns_to_drop)

print("\n================= Starting metadata adjustment =================")

def get_largest_dcm_file(case_id):
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

def save_dataframe(df, output_csv_path):
    df.to_csv(output_csv_path, index=False)
    print(f"Saved: {output_csv_path}")

save_dataframe(Mass_Test, os.path.join(output_folder, "mass_case_description_test_set_mapped.csv"))
save_dataframe(Mass_Training, os.path.join(output_folder, "mass_case_description_train_set_mapped.csv"))
save_dataframe(Calc_Test, os.path.join(output_folder, "calc_case_description_test_set_mapped.csv"))
save_dataframe(Calc_Training, os.path.join(output_folder, "calc_case_description_train_set_mapped.csv"))
