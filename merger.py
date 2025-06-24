import os
import pandas as pd

# Function to merge all outputs to a single excel file

def merge_csv_to_excel(root_folder, output_excel):
    # Get subfolders
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    
    # Create an Excel writer
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        for subfolder in subfolders:
            sheet_name = os.path.basename(subfolder)
            all_dfs = []
            
            # Get all CSV files in the subfolder
            for file in os.listdir(subfolder):
                if file.endswith('.csv'):
                    file_path = os.path.join(subfolder, file)
                    df = pd.read_csv(file_path)
                    df["Source File"] = file  # Add source file column
                    all_dfs.append(df)
            
            # Concatenate all dataframes if there are any
            if all_dfs:
                merged_df = pd.concat(all_dfs, ignore_index=True)
                merged_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Excel file saved at {output_excel}")

# Example usage
root_folder = "D:\\desktop\\wfh\\outputs\\actual"
output_excel = "merged_output.xlsx"
merge_csv_to_excel(root_folder, output_excel)